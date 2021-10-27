# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoyi Zhu,Hao-Shu Fang
# -----------------------------------------------------

"""Script for single-image demo."""
import argparse
import torch
import os
import platform
import sys
import math
import time

import cv2
import numpy as np

from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.presets import SimpleTransform
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.models import builder
from alphapose.utils.config import update_config
from detector.apis import get_detector
from alphapose.utils.vis import getTime
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient,__version__

app = Flask(__name__)

default_config = "configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml"
default_model = "pretrained_models/halpe26_fast_res50_256x192.pth"
default_detector = "yolo"
default_inputimg = ""
save_image = False
vis = False
showbox = False
profile = False
default_format = None
default_min_box_area = 0
default_eval = False
defaul_gpus = "0"
default_flip = False
default_debug = False
default_vis_fast = False
default_pose_flow = False
defaul_pose_track = False
default_config = "configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml"
default_model = "pretrained_models/halpe26_fast_res50_256x192.pth"

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=False,default=default_config,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=False, default=default_model,
                    help='checkpoint file name')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)

args = parser.parse_args()

cfg = update_config(default_config)

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
gpus = [int(i) for i in defaul_gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
print("gpu " + str(gpus))
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
device = torch.device("cuda:" + str(gpus[0]) if gpus[0] >= 0 else "cpu")
print("cuda:" + str(gpus[0]) if gpus[0] >= 0 else "cpu")
tracking = defaul_pose_track or default_pose_flow or default_detector=='tracker'

class DetectionLoader():
    def __init__(self, detector, cfg):
        self.cfg = cfg
        self.device = device
        self.detector = detector

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = cfg.DATA_PRESET.SIGMA

        pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        if cfg.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                pose_dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.device)

        self.image = (None, None, None, None)
        self.det = (None, None, None, None, None, None, None)
        self.pose = (None, None, None, None, None, None, None)

    def process(self, im_name, image):
        # start to pre process images for object detection
        self.image_preprocess(im_name, image)
        print('image_preprocess')
        # start to detect human in images
        self.image_detection()
        print('image_detection')
        # start to post process cropped human image for pose estimation
        self.image_postprocess()
        return self

    def image_preprocess(self, im_name, image):
        # expected image shape like (1,3,h,w) or (3,h,w)
        img = self.detector.image_preprocess(image)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        # add one dimension at the front for batch if image shape (3,h,w)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        orig_img = image # scipy.misc.imread(im_name_k, mode='RGB') is depreciated
        im_dim = orig_img.shape[1], orig_img.shape[0]

        im_name = os.path.basename(im_name)

        with torch.no_grad():
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        self.image = (img, orig_img, im_name, im_dim)

    def image_detection(self):
        imgs, orig_imgs, im_names, im_dim_list = self.image
        if imgs is None:
            self.det = (None, None, None, None, None, None, None)
            return

        with torch.no_grad():
            print("117")
            dets = self.detector.images_detection(imgs, im_dim_list)
            print("load dets")
            if isinstance(dets, int) or dets.shape[0] == 0:
                self.det = (orig_imgs, im_names, None, None, None, None, None)
                return
            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            ids = torch.zeros(scores.shape)

        boxes = boxes[dets[:, 0] == 0]
        if isinstance(boxes, int) or boxes.shape[0] == 0:
            self.det = (orig_imgs, im_names, None, None, None, None, None)
            return
        inps = torch.zeros(boxes.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes.size(0), 4)

        self.det = (orig_imgs, im_names, boxes, scores[dets[:, 0] == 0], ids[dets[:, 0] == 0], inps, cropped_boxes)

    def image_postprocess(self):
        with torch.no_grad():
            (orig_img, im_name, boxes, scores, ids, inps, cropped_boxes) = self.det
            if orig_img is None:
                self.pose = (None, None, None, None, None, None, None)
                return
            if boxes is None or boxes.nelement() == 0:
                self.pose = (None, orig_img, im_name, boxes, scores, ids, None)
                return

            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            self.pose = (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes)

    def read(self):
        return self.pose


class DataWriter():
    def __init__(self, cfg):
        self.cfg = cfg

        self.eval_joints = list(range(cfg.DATA_PRESET.NUM_JOINTS))
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        self.item = (None, None, None, None, None, None, None)

    def start(self):
        # start to read pose estimation results
        return self.update()

    def update(self):
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE

        # get item
        (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.item
        if orig_img is None:
            return None
        # image channel RGB->BGR
        orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
        self.orig_img = orig_img
        if boxes is None or len(boxes) == 0:
            return None
        else:
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            assert hm_data.dim() == 4
            if hm_data.size()[1] == 136:
                self.eval_joints = [*range(0,136)]
            elif hm_data.size()[1] == 26:
                self.eval_joints = [*range(0,26)]
            pose_coords = []
            pose_scores = []

            for i in range(hm_data.shape[0]):
                bbox = cropped_boxes[i].tolist()
                pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)

            boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area)

            _result = []
            for k in range(len(scores)):
                _result.append(
                    {
                        'keypoints':preds_img[k],
                        'kp_score':preds_scores[k],
                        'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        'idx':ids[k],
                        'bbox':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
                    }
                )

            result = {
                'imgname': im_name,
                'result': _result
            }

            if hm_data.size()[1] == 49:
                from alphapose.utils.vis import vis_frame_dense as vis_frame
            elif self.opt.vis_fast:
                from alphapose.utils.vis import vis_frame_fast as vis_frame
            else:
                from alphapose.utils.vis import vis_frame
            self.vis_frame = vis_frame

        return result

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        self.item = (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name)

class SingleImageAlphaPose():
    def __init__(self, cfg):
        self.cfg = cfg

        # Load pose model
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

        print(f'Loading pose model from {default_model}...')
        self.pose_model.load_state_dict(torch.load(default_model, map_location=device))
        self.pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
        if len(gpus) > 1 and gpus[0] >= 1:
            self.pose_model = torch.nn.DataParallel(self.pose_model, device_ids=gpus).to(device)
        else:
            print("get there to cpu")
            self.pose_model.to(device)
        self.pose_model.eval()
        self.det_loader = DetectionLoader(get_detector(args), self.cfg)

    def process(self, im_name, image):
        # Init data writer
        self.writer = DataWriter(self.cfg)

        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }
        pose = None
        try:
            start_time = getTime()
            with torch.no_grad():
                print("262")
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = self.det_loader.process(im_name, image).read()
                if orig_img is None:
                    raise Exception("no image is given")
                if boxes is None or boxes.nelement() == 0:
                    print("267")
                    if profile:
                        print("269")
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    self.writer.save(None, None, None, None, None, orig_img, im_name)
                    if profile:
                        print("274")
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    pose = self.writer.start()
                    print("get called 270")
                    if profile:
                        print("280")
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)
                else:
                    print("284")
                    if profile:
                        print("286")
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    # Pose Estimation
                    inps = inps.to(device)
                    if default_flip:
                        inps = torch.cat((inps, flip(inps)))
                    hm = self.pose_model(inps)
                    if default_flip:
                        hm_flip = flip_heatmap(hm[int(len(hm) / 2):], self.pose_dataset.joint_pairs, shift=True)
                        hm = (hm[0:int(len(hm) / 2)] + hm_flip) / 2
                    if profile:
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    hm = hm.cpu()
                    self.writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                    pose = self.writer.start()
                    print("get called 292")
                    if profile:
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)

            if profile:
                print(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )
            print('===========================> Finish Model Running.')
        except Exception as e:
            print(repr(e))
            print(e)
            print('An error as above occurs when processing the images, please check it')
            raise e
        except KeyboardInterrupt:
            print('===========================> Finish Model Running.')
        print("get called 309")
        return pose

    def getImg(self):
        return self.writer.orig_img

    def vis(self, image, pose):
        if pose is not None:
            image = self.writer.vis_frame(image, pose, self.writer.opt)
        return image

    def writeJson(self, final_result, outputpath, form='coco', for_eval=False):
        from alphapose.utils.pPose_nms import write_json
        write_json(final_result, outputpath, form=form, for_eval=for_eval)
        print("Results have been written to json.")

def load_model():
    return SingleImageAlphaPose(cfg)

def download_img(container_name, blob_name):
    try:
        print("Azure Blob Storage v" + __version__ + " - Python quickstart sample")
        connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        # Create a blob client using the local file name as the name for the blob
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        return blob_client.download_blob()
    except Exception as ex:
        print('Exception:')
        print(ex)
        return ex


demo = load_model()

def blob_to_array(blob):
    arr = np.asarray(bytearray(blob), dtype=np.uint8)
    return arr

@app.route('/predict', methods=['POST'])
def predit():
    print("get called")
    try:
        input_img = request.get_json(silent=True)
        container_name = input_img['containerName']
        blob_name = input_img['blobName']
        print(container_name + "  " + blob_name)
        img = download_img(container_name, blob_name)
        # im_name = args.inputimg    # the path to the target image
        # cv2.imread(im_name)
        img_content = img.readall()
        x = np.fromstring(img_content, dtype='uint8')
        cv2_img = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pose = demo.process(blob_name, image)
        return str(pose)
    except Exception as err:
        print(err)
        return 'unable to process img'



