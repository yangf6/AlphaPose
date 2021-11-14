from .fastpose import FastPose
# from .fastpose_duc import FastPose_DUC
from .hrnet import PoseHighResolutionNet
from .simplepose import SimplePose
# from .fastpose_duc_dense import FastPose_DUC_Dense
from .hardnet import HarDNetPose
from .criterion import L1JointRegression

__all__ = ['FastPose', 'SimplePose', 'PoseHighResolutionNet','HarDNetPose', 'L1JointRegression']
