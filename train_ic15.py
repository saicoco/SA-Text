import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import shutil

from torch.autograd import Variable
from torch.utils import data
import os

from dataset import IC15Loader
from metrics import runningScore
import models
from util import Logger, AverageMeter
import time
import util

from tensorboardX import SummaryWriter

def weighted_regression(gaussian_map, gaussian_gt, training_mask, border_map=None):
    """
    Weighted MSE-loss
    Args:
        gaussian_map: gaussian_map from network outputs
        gaussian_gt: gt for gaussian_map
        training_mask: 
    """
    gaussian_map = torch.sigmoid(gaussian_map)
    text_map = torch.where(gaussian_gt > 0.2, torch.ones_like(gaussian_gt), torch.zeros_like(gaussian_gt))
    center_map = torch.where(gaussian_gt > 0.7, torch.ones_like(gaussian_gt), torch.zeros_like(gaussian_gt))
    center_gt = torch.where(gaussian_gt > 0.7, gaussian_gt, torch.zeros_like(gaussian_gt))
    text_gt = torch.where(gaussian_gt > 0.2, gaussian_gt, torch.zeros_like(gaussian_gt))
    bg_map = 1. - text_map

    pos_num = torch.sum(text_map)
    neg_num = torch.sum(bg_map)

    pos_weight = neg_num * 1. / (pos_num + neg_num)
    neg_weight = 1. - pos_weight

#     mse_loss = F.mse_loss(gaussian_map, gaussian_gt, reduce='none')
    mse_loss = F.smooth_l1_loss(gaussian_map, gaussian_gt, reduce='none')
    weighted_mse_loss = mse_loss * (text_map * pos_weight + bg_map * neg_weight) * training_mask
    
    center_region_loss = torch.sum(center_gt * mse_loss * training_mask) / center_gt.sum()
#     border_loss = torch.sum(border_map * mse_loss * training_mask) / border_map.sum()
    
    return weighted_mse_loss.mean(), torch.sum(text_gt * mse_loss * training_mask) / text_map.sum(), center_region_loss

def ohem_single(score, gt_text, training_mask):
    pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))
    
    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask
    
    neg_num = (int)(np.sum(gt_text <= 0.5))
    neg_num = (int)(min(pos_num * 3, neg_num))
    
    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted = np.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
    return selected_mask

def ohem_batch(scores, gt_texts, training_masks):
    scores = scores.data.cpu().numpy()
    gt_texts = gt_texts.data.cpu().numpy()
    training_masks = training_masks.data.cpu().numpy()
    scores = np.squeeze(scores, axis=1)
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = np.concatenate(selected_masks, 0)
    selected_masks = torch.from_numpy(selected_masks).float()

    return selected_masks

def dice_loss(input, target, mask):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    mask = mask.contiguous().view(mask.size()[0], -1)
    
    input = input * mask
    target = target * mask

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
#     a = torch.sum(input * target, 1)
#     b = torch.sum(input  + target, 1) + 0.001
#     d = (2 * a + 1.0) / (b + 1.0)
#     dice_loss = torch.mean(d)
    return 1 - dice_loss

def cal_text_score(texts, gt_texts, training_masks, running_metric_text, low_thres=0.05, high_thres=0.2):
    training_masks = training_masks.data.cpu().numpy()
    
    pred_text = torch.sigmoid(texts[:, 0, :, :]).data.cpu().numpy() * training_masks
    pred_text[pred_text <= low_thres] = 0
    pred_text[pred_text >= high_thres] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text[gt_text <= low_thres] = 0
    gt_text[gt_text >= high_thres] = 1
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text

def train(train_loader, model, criterion, optimizer, epoch, summary_writer):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    running_metric_text = runningScore(2)
    L1_loss = torch.nn.L1Loss()
    end = time.time()
    for batch_idx, (imgs, gt_texts, training_masks, ori_imgs, border_map) in enumerate(train_loader):
        data_time.update(time.time() - end)

        imgs = Variable(imgs.cuda())
        gt_texts = Variable(gt_texts[:, ::4, ::4].cuda())
        training_masks = Variable(training_masks[:, ::4, ::4].cuda())
        border_map = Variable(border_map.cuda())
        outputs = model(imgs)
        gaussian_map = outputs
#         gaussian_map, center_map, region_map = outputs
        weighted_mse_loss, mse_region_loss, loss_center = weighted_regression(gaussian_map, gt_texts, training_masks)

        center_gt = torch.where(gt_texts > 0.7, gt_texts, torch.zeros_like(gt_texts))
#         center_mask = torch.where(gt_texts > 0.7, torch.ones_like(gt_texts), torch.zeros_like(gt_texts))
        
        region_gt = torch.where(gt_texts > 0.4, gt_texts, torch.zeros_like(gt_texts))
#         region_mask = torch.where(gt_texts > 0.4, torch.ones_like(gt_texts), torch.zeros_like(gt_texts))
        
        # loss for center_map
#         loss_center_dice = criterion(gaussian_map, center_gt, training_masks)
        
        
        # loss for region_map
        loss_region_dice = criterion(gaussian_map, region_gt, training_masks)
        
        
        # loss for border_map
#         border_mask = 1. - (center_other - border_map)
#         loss_border = criterion(gaussian_map, gt_texts, training_masks)
        
        loss = loss_center +  weighted_mse_loss +  mse_region_loss + loss_region_dice
#         print("loss:", loss_center, "loss_region:", loss_region, "weighted_mse_loss:", weighted_mse_loss)
        losses.update(loss.item(), imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        score_center = cal_text_score(gaussian_map, gt_texts, training_masks, running_metric_text, 0, 0.8)
#         score_region = cal_text_score(gaussian_map, gt_texts, training_masks * region_mask, running_metric_text, 0, 0.2)

        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 20 == 0:
            # visualization
            global_step = epoch * len(train_loader) + batch_idx
            maps = torch.sigmoid(gaussian_map[0:1])
            center_map = torch.where(maps > 0.8, maps, torch.zeros_like(maps))
            text_map = torch.where(maps > 0.4, maps, torch.zeros_like(maps))
            summary_writer.add_images('gt/img', ori_imgs[0:1], global_step=global_step)
            summary_writer.add_images('gt/score_map', torch.unsqueeze(gt_texts[0:1], 1), global_step=global_step)    
            summary_writer.add_images('gt/center_map', torch.unsqueeze(center_gt[0:1], 1), global_step=global_step)
            summary_writer.add_images('gt/region_map', torch.unsqueeze(region_gt[0:1], 1), global_step=global_step)
#             summary_writer.add_images('gt/border_map', torch.unsqueeze(border_mask[0:1], 1), global_step=global_step)
            summary_writer.add_images('predicition/score_map', torch.sigmoid(gaussian_map[0:1]), global_step=global_step)
            summary_writer.add_images('predicition/center_map', torch.sigmoid(center_map[0:1]), global_step=global_step)
            summary_writer.add_images('predicition/region_map', torch.sigmoid(text_map[0:1]), global_step=global_step)

            summary_writer.add_scalar('loss/reg_loss', weighted_mse_loss, global_step=global_step)
            summary_writer.add_scalar('loss/reg_center_loss', loss_center, global_step=global_step)
#             summary_writer.add_scalar('loss/center_dice_loss', loss_center_dice, global_step=global_step)   
            summary_writer.add_scalar('loss/region_dice_loss', loss_region_dice, global_step=global_step)   
#             summary_writer.add_scalar('loss/border_loss', loss_border, global_step=global_step)
            summary_writer.add_scalar('loss/text_region_loss', mse_region_loss, global_step=global_step)

            summary_writer.add_scalar('metric/acc_c', score_center['Mean Acc'], global_step=global_step)
            summary_writer.add_scalar('metric/iou_c', score_center['Mean IoU'], global_step=global_step)
#             summary_writer.add_scalar('metric/acc_t', score_region['Mean Acc'], global_step=global_step)
#             summary_writer.add_scalar('metric/iou_t', score_region['Mean IoU'], global_step=global_step)
            
            output_log  = '({batch}/{size}) Batch: {bt:.3f}s | TOTAL: {total:.0f}min | ETA: {eta:.0f}min | Loss: {loss:.4f} | Acc_c: {acc_c: .4f} | IOU_c: {iou_c: .4f} '.format(
                batch=batch_idx + 1,
                size=len(train_loader),
                bt=batch_time.avg,
                total=batch_time.avg * batch_idx / 60.0,
                eta=batch_time.avg * (len(train_loader) - batch_idx) / 60.0,
                loss=losses.avg,
                acc_c=score_center['Mean Acc'],
                iou_c=score_center['Mean IoU'],
#                 acc_t=score_region['Mean Acc'],
#                 iou_t=score_region['Mean IoU'],
            )
            print(output_log)
            sys.stdout.flush()

    return (losses.avg, score_center['Mean Acc'], score_center['Mean IoU'])

def adjust_learning_rate(args, optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def main(args):
    if args.checkpoint == '':
        args.checkpoint = "checkpoints/ic15_%s_bs_%d_ep_%d"%(args.arch, args.batch_size, args.n_epoch)
    if args.pretrain:
        if 'synth' in args.pretrain:
            args.checkpoint += "_pretrain_synth"
        else:
            args.checkpoint += "_pretrain_ic17"

    print ('checkpoint path: %s'%args.checkpoint)
    print ('init lr: %.8f'%args.lr)
    print ('schedule: ', args.schedule)
    sys.stdout.flush()

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    kernel_num = 1
    min_scale = 0.4
    start_epoch = 0

    data_loader = IC15Loader(root_dir=args.root_dir, is_transform=True, img_size=args.img_size, kernel_num=kernel_num, min_scale=min_scale)
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=3,
        drop_last=True,
        pin_memory=True)

    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=kernel_num)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=kernel_num)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=kernel_num)
    elif args.arch == "resnet18":
        model = models.resnet18(pretrained=True, num_classes=kernel_num)
    model = torch.nn.DataParallel(model).cuda()
    
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)

    title = 'icdar2015'
    if args.pretrain:
        print('Using pretrained model.')
        assert os.path.isfile(args.pretrain), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['state_dict'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss','Train Acc.', 'Train IOU.'])
    elif args.resume:
        print('Resuming from checkpoint.')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        print('Training from scratch.')
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss','Train Acc.', 'Train IOU.'])

    writer = SummaryWriter(logdir=args.checkpoint)
    for epoch in range(start_epoch, args.n_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.n_epoch, optimizer.param_groups[0]['lr']))

        train_loss, train_te_acc, train_te_iou = train(train_loader, model, dice_loss, optimizer, epoch, writer)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'lr': args.lr,
                'optimizer' : optimizer.state_dict(),
            }, checkpoint=args.checkpoint)

#         logger.append([optimizer.param_groups[0]['lr'], train_loss, train_te_acc])
#     logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--img_size', nargs='?', type=int, default=640, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=600, 
                        help='# of the epochs')
    parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16, 
                        help='Batch Size')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--pretrain', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--root_dir', default='', type=str, metavar='PATH',
                    help='path to dataset (default: checkpoint)')                    
    args = parser.parse_args()

    main(args)
