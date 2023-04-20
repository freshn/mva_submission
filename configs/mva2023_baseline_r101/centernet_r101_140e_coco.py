_base_ = './centernet_r101_dcnv2_140e_coco.py'

model = dict(neck=dict(use_dcn=False))

data = dict(workers_per_gpu=4)
# resume_from = 'work_dirs/centernet_r101_140e_coco/latest.pth'