from __future__ import print_function
import tensorflow as tf
import os
# import torch
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# import argparse
# import torch.utils.data as data
# from data import AnnotationTransform, VOCDetection, detection_collate, preproc, cfg
# from layers.modules import MultiBoxLoss
# from layers.functions.prior_box import PriorBox
# import time
# import datetime
# import math
# from models.faceboxes import FaceBoxes

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")