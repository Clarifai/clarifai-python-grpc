from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import (
    DOG_IMAGE_URL,
    APPAREL_MODEL_ID,
    COLOR_MODEL_ID,
    FACE_MODEL_ID,
    FOOD_MODEL_ID,
    GENERAL_EMBEDDING_MODEL_ID,
    GENERAL_MODEL_ID,
    LANDSCAPE_QUALITY_MODEL_ID,
    LOGO_MODEL_ID,
    MODERATION_MODEL_ID,
    NSFW_MODEL_ID,
    PORTRAIT_QUALITY_MODEL_ID,
    TEXTURES_AND_PATTERNS_MODEL_ID,
    TRAVEL_MODEL_ID,
    WEDDING_MODEL_ID,
    LOGO_V2_MODEL_ID,
    PEOPLE_DETECTION_YOLOV5_MODEL_ID,
    GENERAL_ENGLISH_IMAGE_CAPTION_CLIP_MODEL_ID,
    IMAGE_SUBJECT_SEGMENTATION_MODEL_ID,
    EASYOCR_ENGLISH_MODEL_ID,
    PADDLEOCR_ENG_CHINESE_MODEL_ID,
    both_channels,
    metadata,
    raise_on_failure,
    BEER_VIDEO_URL,
    post_model_outputs_and_maybe_allow_retries,
)


MODEL_TITLE_AND_ID_PAIRS = [
    ("apparel", APPAREL_MODEL_ID),
    ("color", COLOR_MODEL_ID),
    ("face", FACE_MODEL_ID),
    ("food", FOOD_MODEL_ID),
    ("general embedding", GENERAL_EMBEDDING_MODEL_ID),
    ("general", GENERAL_MODEL_ID),
    ("landscape quality", LANDSCAPE_QUALITY_MODEL_ID),
    ("logo", LOGO_MODEL_ID),
    ("moderation", MODERATION_MODEL_ID),
    ("nsfw", NSFW_MODEL_ID),
    ("portrait quality", PORTRAIT_QUALITY_MODEL_ID),
    ("textures and patterns", TEXTURES_AND_PATTERNS_MODEL_ID),
    ("travel", TRAVEL_MODEL_ID),
    ("wedding", WEDDING_MODEL_ID),
    ("logo v2", LOGO_V2_MODEL_ID),
    ("people detection yolov5", PEOPLE_DETECTION_YOLOV5_MODEL_ID),
    ("caption", GENERAL_ENGLISH_IMAGE_CAPTION_CLIP_MODEL_ID),
    ("subject segmenter", IMAGE_SUBJECT_SEGMENTATION_MODEL_ID),
    ("easyocr english", EASYOCR_ENGLISH_MODEL_ID),
    ("paddleocr english chinese", PADDLEOCR_ENG_CHINESE_MODEL_ID),
]


@both_channels
def test_image_predict_on_public_models(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    for title, model_id in MODEL_TITLE_AND_ID_PAIRS:
        request = service_pb2.PostModelOutputsRequest(
            model_id=model_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL))
                )
            ],
        )
        response = post_model_outputs_and_maybe_allow_retries(stub, request, metadata=metadata())
        raise_on_failure(
            response,
            custom_message=f"Image predict failed for the {title} model (ID: {model_id}).",
        )


@both_channels
def test_video_predict_on_public_models(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    models_with_no_video_support = {
        "color",
    }

    for title, model_id in MODEL_TITLE_AND_ID_PAIRS:
        if title in models_with_no_video_support:
            continue
        request = service_pb2.PostModelOutputsRequest(
            model_id=model_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(video=resources_pb2.Video(url=BEER_VIDEO_URL))
                )
            ],
        )
        response = post_model_outputs_and_maybe_allow_retries(stub, request, metadata=metadata())
        raise_on_failure(
            response,
            custom_message=f"Video predict failed for the {title} model (ID: {model_id}).",
        )
