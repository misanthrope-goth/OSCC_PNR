_base_ = [
    '../../_base_/models/ircsn_r152.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(
        norm_eval=True,
        bn_frozen=True,
        pretrained='pretrain/ircsn_from_scratch_r152_ig65m.pth'
    ),
    cls_head=dict(num_classes=2))
# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/video'
data_root_val = 'data/video'
ann_file_train = 'data/oscc_train.txt'
ann_file_val = 'data/oscc_val.txt'
ann_file_test = 'data/oscc_test.txt'


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=32),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 320)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatVideoShape', input_format='NCTHW',sparse_sample=True, num_sparse_clips=1),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=32,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 320)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatVideoShape', input_format='NCTHW',sparse_sample=True, num_sparse_clips=1),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=32,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 320)),
    dict(type='ThreeCrop', crop_size=320),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatVideoShape', input_format='NCTHW',sparse_sample=True, num_sparse_clips=1),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])


optimizer = dict(
    type='AdamW', lr=0.00001,
    weight_decay=0.0001)  
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=1)

total_epochs = 10

find_unused_parameters = True
