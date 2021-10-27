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


@app.route('/predict', methods=['POST'])
def predit():
    print("get called")
    input_img = request.get_json(silent=True)
    container_name = input_img['containerName']
    blob_name = input_img['blobName']
    print(container_name + "  " + blob_name)
    # img = download_img(container_name, blob_name)
    # im_name = args.inputimg    # the path to the target image
    # cv2.imread(im_name)
    # image = cv2.cvtColor(img.readall(), cv2.COLOR_BGR2RGB)
    # pose = demo.process(blob_name, image)
    return blob_name

# demo = load_model()

