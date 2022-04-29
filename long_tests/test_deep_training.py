import os
import uuid

from google.protobuf import struct_pb2
import numpy as np

from tests.common import DOG_IMAGE_URL
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from tests.common import (
    raise_on_failure,
    wait_for_model_trained,
    post_model_outputs_and_maybe_allow_retries,
)


def pat_key_metadata():
    return (("authorization", "Key %s" % os.environ.get("CLARIFAI_PAT_KEY")),)


def test_mmdetection():
    channel = ClarifaiChannel.get_json_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    app_id = "coco-2017"
    template_name = "MMDetection"
    model_type_id = "visual-detector"
    concepts = []
    for i in range(1, 91):
        concepts.append(resources_pb2.Concept(id=str(i)))
    model_id = "mmdet-coco-" + uuid.uuid4().hex[:15]
    user_app_id = resources_pb2.UserAppIDSet(user_id="me", app_id=app_id)

    train_info_params = struct_pb2.Struct()
    train_info_params.update(
        {
            "template": template_name,
            "custom_config": get_mmdet_config(),
            "seed": 12345
        }
    )

    post_models_response = stub.PostModels(
        service_pb2.PostModelsRequest(
            user_app_id=user_app_id,
            models=[
                resources_pb2.Model(
                    id=model_id,
                    model_type_id=model_type_id,
                    train_info=resources_pb2.TrainInfo(params=train_info_params),
                    output_info=resources_pb2.OutputInfo(
                        data=resources_pb2.Data(concepts=concepts),
                    ),
                )
            ],
        ),
        metadata=pat_key_metadata(),
    )
    raise_on_failure(post_models_response)

    try:
        post_model_versions_response = stub.PostModelVersions(
            service_pb2.PostModelVersionsRequest(
                user_app_id=user_app_id,
                model_id=model_id,
            ),
            metadata=pat_key_metadata(),
        )
        raise_on_failure(post_model_versions_response)
        model_version_id = post_model_versions_response.model.model_version.id

        wait_for_model_trained(
            stub,
            pat_key_metadata(),
            model_id,
            model_version_id,
            user_app_id=user_app_id,
        )

        post_model_outputs_request = service_pb2.PostModelOutputsRequest(
            user_app_id=user_app_id,
            model_id=model_id,
            version_id=model_version_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(url=DOG_IMAGE_URL)
                    )
                )
            ],
        )

        post_model_outputs_response = post_model_outputs_and_maybe_allow_retries(
            stub,
            post_model_outputs_request,
            metadata=pat_key_metadata(),
        )
        raise_on_failure(post_model_outputs_response)

        first_region = post_model_outputs_response.outputs[0].data.regions[0]

        bounding_box = first_region.region_info.bounding_box
        assert calculate_iou([bounding_box.top_row, bounding_box.left_col, bounding_box.bottom_row, bounding_box.right_col], [0.2289, 0.1643, 1, 0.8023]) > .9

        assert len(first_region.data.concepts) == 1
        first_region_concept = first_region.data.concepts[0]
        assert first_region_concept.id == "18"
        assert first_region_concept.name == "dog"
        assert first_region_concept.value > 0.5

        for i in range(1, len(post_model_outputs_response.outputs[0].data.regions)):
            assert len(post_model_outputs_response.outputs[0].data.regions[i].data.concepts) == 1
            region_concept = post_model_outputs_response.outputs[0].data.regions[i].data.concepts[0]
            assert region_concept.value < 0.5
    finally:
        delete_model_response = stub.DeleteModel(
            service_pb2.DeleteModelRequest(
                user_app_id=user_app_id,
                model_id=model_id,
            ),
            metadata=pat_key_metadata(),
        )
        raise_on_failure(delete_model_response)


def get_mmdet_config():
    return """
_base_ = '/mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco.py'
model=dict(
  bbox_head=dict(num_classes=0),
  )
optimizer = dict(
  lr=0.06
  )
data=dict(
  samples_per_gpu=32,
  workers_per_gpu=4,
  train=dict(
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[1.0, 1.0, 1.0],
            to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
      ], 
    ann_file='',
    img_prefix='',
    classes=''
    ),
  val=dict(
    pipeline=[
      dict(type='LoadImageFromFile'),
      dict(
          type='MultiScaleFlipAug',
          img_scale=(512, 512),
          flip=False,
          transforms=[
              dict(type='Resize', keep_ratio=True),
              dict(type='RandomFlip'),
              dict(
                  type='Normalize',
                  mean=[123.675, 116.28, 103.53],
                  std=[1.0, 1.0, 1.0],
                  to_rgb=True),
              dict(type='Pad', size_divisor=32),
              dict(type='ImageToTensor', keys=['img']),
              dict(type='Collect', keys=['img'])
          ])
      ],
    ann_file='',
    img_prefix='',
    classes=''))

img_norm_cfg=dict(mean=[123.675, 116.28, 103.53], to_rgb=True)
runner = dict(type='EpochBasedRunner', max_epochs=1)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[1.0, 1.0, 1.0],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
"""

def calculate_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=0)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=0)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou