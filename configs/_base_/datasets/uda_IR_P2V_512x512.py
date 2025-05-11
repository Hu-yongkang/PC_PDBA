# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
 


dataset_type = 'ISPRSPDataset'         
data_root = ' '
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)  # 
crop_size = (512, 512)
LoveDAR_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(720,720),ratio_range=(0.75, 1.5)),    #  
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
LoveDAU_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(720,720),ratio_range=(0.75, 1.5)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']), 
]

test_pipeline = [    
    dict(type='LoadImageFromFile'),
 
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512,512),
 
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
    
]
 
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='ISPRSPDataset',
            data_root=' ',
            img_dir=' ',
            ann_dir=' ', 
            pipeline=LoveDAR_train_pipeline),
        target=dict(
            type='ISPRSVDataset',
            data_root=' ',
            img_dir=' ',
            ann_dir=' ', 
            pipeline=LoveDAU_train_pipeline)),
    val=dict(
        type='ISPRSVDataset',
        data_root=' ',
        img_dir=' ',
        ann_dir=' ', 
        pipeline=test_pipeline),
    test=dict(
        type='ISPRSVDataset',
        data_root=' ',
        img_dir=' ',
        ann_dir=' ',
        pipeline=test_pipeline))
