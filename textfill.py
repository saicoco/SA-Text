# coding=utf-8

import numpy as np
import cv2
import queue as Queue
import threading

def textfill(gaussian_map, min_area=10, top_threshold=0.6, end_thershold=0.5):

    """
    Args:
        kernals: center_region and text_region
    """

    center_region = np.where(gaussian_map > top_threshold, np.ones_like(gaussian_map), np.zeros_like(gaussian_map))
    pred = np.zeros(gaussian_map.shape, dtype='int32')
    
    label_num, label = cv2.connectedComponents(center_region.astype(np.uint8), connectivity=4)
    
    # for label_idx in range(1, label_num):
    #     if np.sum(label == label_idx) < min_area:
    #         label[label == label_idx] = 0

    queue = Queue.Queue(maxsize = 0)
    next_queue = Queue.Queue(maxsize = 0)
    points = np.array(np.where(label > 0)).transpose((1, 0))
    
    for point_idx in range(points.shape[0]):
        x, y = points[point_idx, 0], points[point_idx, 1]
        l = label[x, y]
        queue.put((x, y, l))
        pred[x, y] = l

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    kernal = np.where(gaussian_map.copy() > end_thershold, gaussian_map, np.zeros_like(gaussian_map))
    # kernal_mask = np.where(gaussian_map.copy() > end_thershold, np.ones_like(gaussian_map)*0.5, np.zeros_like(gaussian_map))
    while not queue.empty():
        (x, y, l) = queue.get()

        for j in range(4):
            tmpx = x + dx[j]
            tmpy = y + dy[j]

            if tmpx < 0 or tmpx >= kernal.shape[0] or tmpy < 0 or tmpy >= kernal.shape[1]:
                continue

            if kernal[tmpx, tmpy] < end_thershold or pred[tmpx, tmpy] > 0:
                continue
            
            if kernal[tmpx, tmpy] >= end_thershold or (kernal[tmpx, tmpy] >= end_thershold and kernal[tmpx, tmpy] <= kernal[x, y]):

                queue.put((tmpx, tmpy, l))
                pred[tmpx, tmpy] = l
                pred[x, y] = l
                kernal_mask[x, y] = l

    # print(np.where(kernal_mask==0.5))
    # pred[kernal_mask==0.5] = label_num
    return pred



def connect_conpoent(label, kernal, pred, label_idx, end_thershold, min_area=10):
    
    if np.sum(label == label_idx) < min_area:
        label[label == label_idx] = 0

    queue = Queue.Queue(maxsize = 0)
    points = np.array(np.where(label == label_idx)).transpose((1, 0))
    
    for point_idx in range(points.shape[0]):
        x, y = points[point_idx, 0], points[point_idx, 1]
        l = label[x, y]
        queue.put((x, y, l))
        pred[x, y] = l

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    
    while not queue.empty():
        (x, y, l) = queue.get()

        is_edge = True
        lowest = 0

        for j in range(4):
            tmpx = x + dx[j]
            tmpy = y + dy[j]

            if tmpx < 0 or tmpx >= kernal.shape[0] or tmpy < 0 or tmpy >= kernal.shape[1]:
                continue

            if kernal[tmpx, tmpy] < end_thershold/2. or pred[tmpx, tmpy] > 0:
                continue
            
            if kernal[tmpx, tmpy] >= end_thershold/2. or (kernal[tmpx, tmpy] >= end_thershold and kernal[tmpx, tmpy] <= kernal[x, y]):

                queue.put((tmpx, tmpy, l))
                pred[tmpx, tmpy] = l
                is_edge = False
    return pred

def textfill_v2(gaussian_map, min_area=10, top_threshold=0.7, end_thershold=0.2):

    """
    Args:
        kernals: center_region and text_region
    """

    center_region = np.where(gaussian_map > top_threshold, np.ones_like(gaussian_map), np.zeros_like(gaussian_map))
    label_num, label = cv2.connectedComponents(center_region.astype(np.uint8), connectivity=4)
    pred = np.zeros(gaussian_map.shape, dtype='int32')
    kernal = gaussian_map.copy()
    for label_idx in range(1, label_num):
        # threading.Thread(target=connect_conpoent, args=(label, kernal, pred, label_idx, end_thershold, min_area))
        pred = connect_conpoent(label, kernal, pred, label_idx, end_thershold, min_area=10)

    return pred  
