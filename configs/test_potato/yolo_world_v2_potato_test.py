# configs/potato_simple_test.py
_base_ = '../pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_cc3mlite_train_lvis_minival.py'
# _base_ = '../pretrain/yolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py'
# _base_ = '../pretrain/yolo_world_v2_m_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py'

data_root = 'potato_dataset/'
num_potato_classes = 4

test_dataloader = dict(
    dataset=dict(
        # Adding inner dataset for MultiModalDataset
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root=data_root,
            ann_file='annotations/test_coco.json',
            data_prefix=dict(img='images/test/'),
            metainfo=dict(classes=('Bud-Sprouted', 'Defected potato',
                                   'Diseased-fungal-damaged', 'Good')),
        ),
        class_text_path='potato_dataset/class_texts.json'
    )
)

model = dict(
    # Freezing backbone
    backbone=dict(
        image_model=dict(frozen_stages=4),  # Freezing the first 4 stages
        text_model=dict(frozen_modules=['all'])  # Тext model is already frozen
    ),
    
    # Modifying bbox_head for 5 classes
    # bbox_head=dict(
    #     head_module=dict(
    #         num_classes=num_potato_classes,
    #         use_bn_head=True,
    #         embed_dims=512,
    #     )
    # ),
    test_cfg=dict(
        max_per_img=300,
        # score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7)
    )
)

test_evaluator = dict(
    type='mmdet.CocoMetric', 
    ann_file='potato_dataset/annotations/test_coco.json',
    metric='bbox',
    classwise=True
)