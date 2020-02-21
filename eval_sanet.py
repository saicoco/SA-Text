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

import pyclipper
import Polygon as plg
import torch.onnx
import math
from textfill import textfill

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
    filename = util.io.join_path(path, '%s.txt'%(image_name))
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

def dilated_kernel(S):
    kernel_size = int(np.ceil(35/4.))
    if S < 20000:
        kernel_size = int(np.ceil(int(8. + S * 1.0 / 750.)/4.))
    return np.ones((kernel_size, kernel_size), dtype=np.uint8)

def test(args):
    import torch
    data_loader = IC15TestLoader(root_dir=args.root_dir, long_size=args.long_size)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    # Setup Model
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=False, num_classes=1, scale=args.scale, train_mode=False)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=1, scale=args.scale)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=1, scale=args.scale)
    
    for param in model.parameters():
        param.requires_grad = False

    if args.gpus > 0:
        model = model.cuda()
    
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            device = torch.device('cpu') if args.gpus < 0 else None
            checkpoint = torch.load(args.resume, map_location=device)
            
            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)

            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            sys.stdout.flush()
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            sys.stdout.flush()
    model.eval()
    if args.onnx:
        import torch.onnx.symbolic_opset9 as onnx_symbolic
        def upsample_nearest_2d(g, input, output_size):
            scales = g.op('Constant', value_t=torch.Tensor([1., 1., 2., 2.]))
            return g.op("Upsample", input, scales, mode_s='nearest')
        onnx_symbolic = upsample_nearest_2d
        dummy_input = torch.autograd.Variable(torch.randn(1, 3, 640, 640)).cpu()
        torch.onnx.export(model, dummy_input, 'sanet.onnx', verbose=False,
        input_names=["input"],	
        output_names=["gaussian_map", 'border_map'],
        dynamic_axes = {'input':{0:'b', 2:'h', 3:'w'}, 'gaussian_map':{0:'b', 2:'h', 3:'w'}, 'border_map':{0:'b', 2:'h', 3:'w'}}
        )
        return 0
    total_frame = 0.0
    total_time = 0.0
    
    for idx, (org_img, img, scale_val) in enumerate(test_loader):
        print('progress: %d / %d'%(idx, len(test_loader)))
        sys.stdout.flush()
        if args.gpus > 0:
            img = Variable(img.cuda(), volatile=True)
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()
        resize_img = img.cpu().numpy().astype('uint8')[0].transpose((1, 2, 0)).copy()
        if args.gpus > 0:
            torch.cuda.synchronize()
        start = time.time()

        outputs = model(img)
        infer_time = time.time()
        probability_map, border_map = outputs[0].sigmoid(), outputs[1].sigmoid()
#         print(probability_map.max(), probability_map.min())
        score = probability_map[0, 0]
        border_score = border_map[0, 0]
#         prediction_map = textfill(score.cpu().numpy(), border_score, top_threshold=0.7, end_thershold=0.2)
        post_time = time.time()
        center_text = torch.where(score > 0.7, torch.ones_like(score), torch.zeros_like(score))
        center_text = center_text.data.cpu().numpy().astype(np.uint8)

        text_region = torch.where(score > 0.5, torch.ones_like(score), torch.zeros_like(score))
        border_region = torch.where(border_score > 0.9, torch.ones_like(border_score), torch.zeros_like(border_score))
        prediction_map = text_region.data.cpu().numpy()
        border_region = border_region.data.cpu().numpy()
        prediction_map[border_region==1] = 0
        
        prob_map = probability_map.cpu().numpy()[0, 0] * 255
        bord_map = border_map[0, 0].cpu().numpy() * 255
        out_path = 'outputs/vis_ic15/'
        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
#         cv2.imwrite(out_path + image_name + '_prob.png', prob_map.astype(np.uint8))
#         cv2.imwrite(out_path + image_name + '_bd.png', bord_map.astype(np.uint8))
#         cv2.imwrite(out_path + image_name + '_tr.png', text_region.astype(np.uint8) * 255)
#         cv2.imwrite(out_path + image_name + '_fl.png', prediction_map.astype(np.uint8) * 255)
    
        scale = (org_img.shape[1] * 1.0 / img.shape[1], org_img.shape[0] * 1.0 / img.shape[0])
        bboxes = []
        scale_val = scale_val.cpu().numpy()
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(prediction_map.astype(np.uint8), connectivity=4)
        t5 = time.time()
#         nLabels = prediction_map.max()
#         print("nLabels:", nLabels)
        img_h, img_w = prediction_map.shape[:2]
        for k in range(1, nLabels):
        
            size = stats[k, cv2.CC_STAT_AREA]
#             if size < 10: continue
            # make segmentation map
            segmap = np.zeros(prediction_map.shape, dtype=np.uint8)
            segmap[labels==k] = 255
#             segmap[np.logical_and(border_score > 0.7, score.cpu().numpy() < 0.05)] = 0   # remove link area
            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
#             print("xywh:", x, y, w, h, " size:", size)
            niter = int(math.sqrt(size * min(w, h) / (w * h)) * 4.3)
            sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
#             print("info:", sy, ey, sx, ex)
            # boundary check
            if sx < 0 : sx = 0
            if sy < 0 : sy = 0
            if ex >= img_w: ex = img_w
            if ey >= img_h: ey = img_h
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))

            segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
            ############### original postprocess ################
#             # make segmentation map
#             segmap = np.zeros(score.shape, dtype=np.uint8)
#             segmap[prediction_map==k] = 255

#             # contourexpand
#             text_area = np.sum(segmap)
#             kernel = dilated_kernel(text_area)
            ############## original postprocess ################
            
#             segmap = cv2.dilate(segmap, kernel, iterations=1)
            np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle) * 4
            box = box / scale_val
            box = box.astype('int32')
            bboxes.append(box)
        t6 = time.time()
        print("infer_time:{}, post_time:{}, expand_time:{}".format(infer_time-start, post_time-infer_time, t6-t5))
        # find contours
        bboxes = np.array(bboxes)
        num_box = bboxes.shape[0]
        if args.gpus > 0:
            torch.cuda.synchronize()
        end = time.time()
        total_frame += 1
        total_time += (end - start)
        sys.stdout.flush()

        for bbox in bboxes:
            cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)
        image_name = ".".join(data_loader.img_paths[idx].split('/')[-1].split('.')[:-1])
        write_result_as_txt(image_name, bboxes.reshape((-1, 8)), 'outputs/submit_ic15/')
        
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
    parser.add_argument('--long_size', nargs='?', type=int, default=784,
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
