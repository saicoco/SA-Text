# coding=utf-8
import numpy as np
from PIL import Image
import numpy as np
import cv2
import tvm
import numpy as np
from tvm.contrib import util, ndk, graph_runtime
import os
import topi
import matplotlib.pyplot as plt
import time
import glob
import sys

def preprocess(im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    im = cv2.resize(im, dsize=(512, 512))
    im = (im / 255. - np.array(mean)) / np.array(std)
    return im.transpose((2, 0, 1)).astype(np.float32)

def sigmoid(data):
    return 1. / (1. + np.exp(-1. * data))

def inference(mod, data):
    data = preprocess(data)
    t1 = time.time()
    mod.set_input('input', tvm.nd.array([data]))
    mod.run()
    gaussian_map = mod.get_output(0).asnumpy()[0, 0]
    border_map = mod.get_output(1).asnumpy()[0, 0]
    gaussian_map = sigmoid(gaussian_map)
    border_map = sigmoid(border_map)
    border_map = np.where(border_map > 0.7, np.ones_like(border_map), np.zeros_like(border_map)) * 255
    gaussian_map = np.where(gaussian_map > 0., gaussian_map, np.zeros_like(gaussian_map))* 255
    dur = time.time() - t1
    print("dur_time:", dur)
    cv2.imwrite("gaussian_map.jpg", gaussian_map.astype(np.uint8))
#     plt.imshow(gaussian_map.astype(np.uint8))
#     plt.show()

loaded_json = open("tvm_optimize/deploy_graph.json").read()
loaded_lib = tvm.runtime.load_module('tvm_optimizes/deploy_lib.tar')
loaded_params = bytearray(open('tvm_optimize/deploy_param.params', "rb").read())

target = tvm.target.cuda()
ctx = tvm.context(str(target), 0)
m = graph_runtime.create(loaded_json, loaded_lib, ctx)
m.load_params(loaded_params)
ts = []
# ims = glob.glob(sys.argv[1] + '*g')
# for im_name in ims:
/samples/1.jpg'
img = cv2.imread(im_name)[:, :, ::-1]
inference(m, img)