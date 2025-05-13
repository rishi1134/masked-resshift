#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-08-13 21:37:58

'''
Calculate LPIPS, and FID.
'''

import os, sys, math
import lpips
import pyiqa
import pickle
import argparse
import numpy as np
from scipy import linalg
from pathlib import Path
from loguru import logger as base_logger

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import util_image

def load_im_tensor(im_path):
    """
    Load image and normalize to [-1, 1]
    """
    im = util_image.imread(im_path, chn='rgb', dtype='float32')
    im = torch.from_numpy(im).permute(2,0,1).unsqueeze(0).cuda()
    im = (im - 0.5) / 0.5

    return im

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, default="", help="Path to save the HQ images")
    parser.add_argument("--sr_dir", type=str, default="", help="Path to save the SR images")
    args = parser.parse_args()

    # setting logger
    log_path = str(Path(args.sr_dir).parent / 'metrics.log')
    logger = base_logger
    logger.remove()
    logger.add(log_path, format="{time:YYYY-MM-DD(HH:mm:ss)}: {message}", mode='w', level='INFO')
    logger.add(sys.stderr, format="{message}", level='INFO')

    for key in vars(args):
        value = getattr(args, key)
        logger.info(f'{key}: {value}')

    lpips_metric_vgg = lpips.LPIPS(net='vgg').cuda()
    lpips_metric_alex = lpips.LPIPS(net='alex').cuda()
    clipiqa_metric = pyiqa.create_metric('clipiqa')
    musiq_metric = pyiqa.create_metric('musiq')

    info_path = Path(args.gt_dir).parent / 'infos' / 'mask_split.pkl'
    with open(str(info_path), mode='rb') as ff:
        mask_split = pickle.load(ff)

    mean_lpips_vgg = 0
    mean_lpips_alex = 0
    mean_clipiqa = 0
    mean_musiq = 0
    num_mask_types = 0
    for mask_type in mask_split.keys():
        num_mask_types += 1
        im_path_list = [(Path(args.sr_dir) / im_name) for im_name in mask_split[mask_type]]

        logger.info(f"Mask types: {mask_type}, images: {len(im_path_list)}")

        features = []
        current_lpips_vgg = 0
        current_lpips_alex = 0
        current_clipiqa = 0
        current_musiq = 0
        for im_path_sr in im_path_list:
            im_sr = load_im_tensor(im_path_sr)

            im_path_gt = Path(args.gt_dir) / im_path_sr.name
            im_gt = load_im_tensor(im_path_gt)

            with torch.no_grad():
                # calculate lpips
                current_lpips_vgg += lpips_metric_vgg(im_gt, im_sr).sum().item()
                current_lpips_alex += lpips_metric_alex(im_gt, im_sr).sum().item()
                # calculate clipiqa
                current_clipiqa += clipiqa_metric(im_sr).sum().item()
                # calculate musiq
                current_musiq += musiq_metric(im_sr).sum().item()

        # calculate average lpips score
        current_lpips_vgg /= len(im_path_list)
        mean_lpips_vgg += current_lpips_vgg
        current_lpips_alex /= len(im_path_list)
        mean_lpips_alex += current_lpips_alex

        # calculate average clipiqa score
        current_clipiqa /= len(im_path_list)
        mean_clipiqa += current_clipiqa

        # calculate average musiq score
        current_musiq /= len(im_path_list)
        mean_musiq += current_musiq

        logger.info(f"  LPIPS-VGG: {current_lpips_vgg:6.4f}")
        logger.info(f"  LPIPS-Alex: {current_lpips_alex:6.4f}")
        logger.info(f"  CLIPIQA: {current_clipiqa:6.4f}")
        logger.info(f"  MUSIQ: {current_musiq:5.2f}")

    mean_lpips_vgg /= num_mask_types
    mean_lpips_alex /= num_mask_types
    mean_clipiqa /= num_mask_types
    mean_musiq /= num_mask_types

    logger.info(f"MEAN LPIPS-VGG: {mean_lpips_vgg:6.4f}")
    logger.info(f"MEAN LPIPS-Alex: {mean_lpips_alex:6.4f}")
    logger.info(f"MEAN CLIPIQA: {mean_clipiqa:6.4f}")
    logger.info(f"MEAN MUSIQ: {mean_musiq:5.2f}")

if __name__ == "__main__":
    main()
