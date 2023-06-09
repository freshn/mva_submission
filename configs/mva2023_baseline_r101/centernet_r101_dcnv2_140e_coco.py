_base_ = [
    # '../_base_/datasets/coco_detection.py',
    '../mva2023_baseline/drone_dataset.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='CenterNet',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        frozen_stages=-1,
        norm_eval=False,
        norm_cfg=dict(type='BN'),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='ckpt/resnext101_64x4d-ee2c6f71.pth')),
    neck=dict(
        type='CTResNetNeck',
        in_channel=2048,
        num_deconv_filters=(1024, 512, 256),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=1,  # 80
        in_channel=256,
        feat_channel=256,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))

# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[90, 120])
runner = dict(max_epochs=140)

# Avoid evaluation and saving weights too frequently
evaluation = dict(interval=5, metric='bbox')
checkpoint_config = dict(interval=5)
