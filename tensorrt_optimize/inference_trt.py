# coding=utf-8
from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw
import cv2
import math
import sys, os
import common


def load_engine(trt_runtime, plan_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]
    
def scale(img, long_size=512):
    """
    resize with long-size
    """
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    new_H, new_W = img.shape[:2]
    img_padd = np.zeros((long_size, long_size, 3), dtype=np.uint8)
    img_padd[:new_H, :new_W, :] = img
    return img_padd, scale

def draw_polys(image, polys):
    for i, poly in enumerate(polys):
        pts = np.array(poly).reshape((4, 2)).astype(np.int32)
        cv2.drawContours(image, [pts], 0, color=(0, 255, 0), thickness=2)
    return image

def preprocess(im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    im = (im / 255. - np.array(mean)) / np.array(std)
    im = im.transpose((2, 0, 1)).astype(np.float32)
    shape = im.shape[1:]
    im = np.array(im, dtype=np.float32, order='C')
    return im, shape

def postprocess(gaussian_map, text_thres=0.3):
    prediction_map = np.where(gaussian_map > text_thres, 1, 0)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(prediction_map.astype(np.uint8), connectivity=4)
    bboxes = []
    img_h, img_w = prediction_map.shape[:2]
    scores = []
    for k in range(1, nLabels):
    
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 4: continue
        # make segmentation map
        segmap = np.zeros(prediction_map.shape, dtype=np.uint8)
        segmap[labels==k] = 255

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]

        if size*1./(w*h) >0.4:
            niter = int(math.sqrt(size * min(w, h) / (w * h)) * 4.3)
        else:
            new_w = math.sqrt(w**2 + h**2)
            niter = int(math.sqrt(size * 1.0 / new_w) * 3.3)

        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle) * 4
        
        mask = np.zeros_like(gaussian_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        box_score = cv2.mean(gaussian_map, mask)[0]
        if box_score < 0:
            continue
        scores.append(box_score)
        bboxes.append(box)
    return np.array(bboxes) if len(bboxes)>0 else None, scores

def inference(data_dir, engine_path, long_side_size=1024):
    filenames = glob.glob(data_root + '/*g')[:10]
    times = []
    TRT_LOGGER = trt.Logger()
    trt_runtime = trt.Runtime(TRT_LOGGER)
    engine = load_engine(trt_runtime, engine_path)
    
    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        for filename in tqdm(filenames):
            ori_image = cv2.imread(filename)
            image, im_scales = scale(ori_image, long_size=long_side_size)
            image, shape = preprocess(image)
            inputs[0].host = image
            
            t1 = time.time()
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            dur = time.time() - t1
            times.append(dur)
            gaussian_map = trt_outputs[0].reshape((shape[0]//4, shape[1]//4))
            boxes, scores = postprocess(gaussian_map)
            
            polys = []
            if boxes is not None:
                boxes = boxes[:, :8].reshape((-1, 4, 2)) * 1. /im_scales
                for box in boxes:
                    box = sort_poly(box.astype(np.int32))
                    polys.append(box)
                polys = np.array(polys, dtype=np.float32).reshape((-1, 8))
                result_im = draw_polys(ori_image, polys)
                cv2.imwrite("res/{}.jpg".format(filename.split('/')[-1].split('.')[0]), result_im)
                
    print("mean_time:", np.mean(times))

            
if __name__ == '__main__':
    import sys, glob, time
    from tqdm import tqdm
    
    data_root = sys.argv[1]
    engine_path = sys.argv[2]
    inference(data_root, engine_path, long_side_size=512)