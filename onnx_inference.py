import os
import cv2
import sys
import time
import collections
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils import data

from dataset import IC15TestLoader
import models
import util
from dataset.icdar2015_loader import shrink
import pyclipper
import Polygon as plg
import torch.onnx
import math
import onnxruntime

import matplotlib.pyplot as plt


def extend_3c(img):
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.concatenate((img, img, img), axis=2)
    return img

def debug(idx, img_paths, imgs, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    col = []
    for i in range(len(imgs)):
        row = []
        for j in range(len(imgs[i])):
            # img = cv2.copyMakeBorder(imgs[i][j], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            row.append(imgs[i][j])
        res = np.concatenate(row, axis=1)
        col.append(res)
    res = np.concatenate(col, axis=0)
    img_name = img_paths[idx].split('/')[-1]
    cv2.imwrite(output_root + img_name, res)

def write_result_as_txt(image_name, bboxes, path):
    filename = util.io.join_path(path, 'res_%s.txt'%(image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
        lines.append(line)
    util.io.write_lines(filename, lines)

def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

def unshrink(bboxes, rate=1.5, max_shr=20):
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        if area < 900:
            continue
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = (int)(area * rate / (peri + 0.001) + 0.5)
        
        shrinked_bbox = pco.Execute(offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue
        shrinked_bbox = np.array(shrinked_bbox)[0]
        
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bboxes.append(shrinked_bbox)
    
    return np.array(shrinked_bboxes)

def test(args):
    import torch
    data_loader = IC15TestLoader(root_dir=args.root_dir, long_size=args.long_size)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    # onnx inference
    model = onnxruntime.InferenceSession(args.resume)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    total_frame = 0.0
    total_time = 0.0
    for idx, (org_img, img, scale_val) in enumerate(test_loader):
        print('progress: %d / %d'%(idx, len(test_loader)))
        sys.stdout.flush()
        if args.gpus > 0:
            img = Variable(img.cuda(), volatile=True)
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()
        if args.gpus > 0:
            torch.cuda.synchronize()
        start = time.time()

        # outputs = model(img)
        inputs = {model.get_inputs()[0].name: to_numpy(img)}
        outputs = model.run(None, inputs)
        probability_map, threshold_map, binarization_map = outputs

        score = binarization_map[0, 0]
        text = np.where(score > 0.2, np.ones_like(score), np.zeros_like(score))
        text = text.astype(np.uint8)

        prob_map = probability_map[0, 0] * 255
        thre_map = threshold_map[0, 0]
        bin_map = binarization_map[0, 0] * 255
        out_path = 'outputs/vis_ic15/'
        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
        print("im_name:", image_name)
        # cv2.imwrite(out_path + image_name + '_prob.png', prob_map.astype(np.uint8))
        # cv2.imwrite(out_path + image_name + '_thre.png' , thre_map.astype(np.uint8))
        # cv2.imwrite(out_path + image_name + '_bin.png', bin_map.astype(np.uint8))
        
        scale = (org_img.shape[1] * 1.0 / img.shape[1], org_img.shape[0] * 1.0 / img.shape[0])
        print("[shape_info:]", text.shape, img.shape, org_img.shape, scale, scale_val)
        bboxes = []
        scale_val = scale_val.cpu().numpy()

        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text.astype(np.uint8), connectivity=4)
        img_h, img_w = text.shape

        for k in range(1, nLabels):
            # size filtering
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 20:
                continue
            
            # make segmentation map
            segmap = np.zeros(text.shape, dtype=np.uint8)
            segmap[labels==k] = 255
            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            # niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
            # sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
            # # boundary check
            # if sx < 0 : sx = 0
            # if sy < 0 : sy = 0
            # if ex >= img_w: ex = img_w
            # if ey >= img_h: ey = img_h
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
            # segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
            np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle) * 4
            box = box / scale_val
            box = box.astype('int32')
            bboxes.append(box)


        # find contours
        bboxes = np.array(bboxes)
        num_box = bboxes.shape[0]
        try:
            unshrink_bboxes = unshrink(bboxes.reshape((num_box, -1, 2)))
        except:
            continue
        for i in range(unshrink_bboxes.shape[0]):
            cv2.drawContours(text_box, [unshrink_bboxes[i]], -1, (0, 255, 255), 2)
        if args.gpus > 0:
            torch.cuda.synchronize()
        end = time.time()
        total_frame += 1
        total_time += (end - start)
        print('fps: %.2f'%(total_frame / total_time))
        sys.stdout.flush()

        for bbox in bboxes:
            cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)

        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
        write_result_as_txt(image_name, bboxes.reshape((-1, 8)), 'outputs/submit_ic15/')
        
        # text_box = cv2.resize(text_box, (text.shape[1], text.shape[0]))
        debug(idx, data_loader.img_paths, [[text_box]], 'outputs/vis_ic15/')

    # cmd = 'cd %s;zip -j %s %s/*'%('./outputs/', 'submit_ic15.zip', 'submit_ic15');
    # print(cmd)
    # sys.stdout.flush()
    # util.cmd.cmd(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--root_dir', nargs='?', type=str, default=None)
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=1.0,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=7,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--scale', nargs='?', type=int, default=1,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs='?', type=int, default=640,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=5.0,
                        help='min kernel area')
    parser.add_argument('--min_area', nargs='?', type=float, default=800.0,
                        help='min area')
    parser.add_argument('--min_score', nargs='?', type=float, default=0.93,
                        help='min score')
    parser.add_argument('--gpus', nargs='?', type=int, default=-1,
                        help='gpu device')
    parser.add_argument('--onnx', nargs='?', type=bool, default='',
                        help='wether to export onnx model')
    args = parser.parse_args()
    test(args)
