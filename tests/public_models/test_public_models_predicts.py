import os

from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc

from common import (
    APPAREL_MODEL_ID,
    BEER_VIDEO_URL,
    COLOR_MODEL_ID,
    DOG_IMAGE_URL,
    FACE_MODEL_ID,
    FOOD_MODEL_ID,
    GENERAL_EMBEDDING_MODEL_ID,
    GENERAL_MODEL_ID,
    LANDSCAPE_QUALITY_MODEL_ID,
    LOGO_MODEL_ID,
    MODERATION_MODEL_ID,
    NER_ENGLISH_MODEL_ID,
    NSFW_MODEL_ID,
    PORTRAIT_QUALITY_MODEL_ID,
    TEXT_GEN_MODEL_ID,
    TEXT_MULTILINGUAL_MODERATION_MODEL_ID,
    TEXT_SENTIMENT_MODEL_ID,
    TEXT_SUM_MODEL_ID,
    TEXTURES_AND_PATTERNS_MODEL_ID,
    TRAVEL_MODEL_ID,
    WEDDING_MODEL_ID,
    LOGO_V2_MODEL_ID,
    PEOPLE_DETECTION_YOLOV5_MODEL_ID,
    GENERAL_ENGLISH_IMAGE_CAPTION_CLIP_MODEL_ID,
    IMAGE_SUBJECT_SEGMENTATION_MODEL_ID,
    EASYOCR_ENGLISH_MODEL_ID,
    PADDLEOCR_ENG_CHINESE_MODEL_ID,
    ENGLISH_AUDIO_URL,
    ENGLISH_ASR_MODEL_ID,
    GENERAL_ASR_NEMO_JASPER_MODEL_ID,
    OBJECT_DETECTION_MODELS,
    TRANSLATION_TEST_DATA,
    TRANSLATION_MODELS,
    both_channels,
    metadata,
    post_model_outputs_and_maybe_allow_retries,
    raise_on_failure,
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

# Add models in object_detection_models dict to model_id_pairs list
for _, values in OBJECT_DETECTION_MODELS.items():
    MODEL_TITLE_AND_ID_PAIRS.append(tuple(values))

TEXT_MODEL_TITLE_IDS_TUPLE = [
    ("text summarization", TEXT_SUM_MODEL_ID, "summarization", "hcs"),
    ("text generation", TEXT_GEN_MODEL_ID, "text-generation", "textgen"),
    ("text sentiment", TEXT_SENTIMENT_MODEL_ID, "text-classification", "nlptownres"),
    (
        "text multilingual moderation",
        TEXT_MULTILINGUAL_MODERATION_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "ner english",
        NER_ENGLISH_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
]

# Map corresponding test data to each model in translation_models dict
# append app_id and user_id vars to model data and then add the data to
# the text_model_title_ids list of tuple
for key, values in TRANSLATION_MODELS.items():
    language = key.split("_")[0]
    values.append(TRANSLATION_TEST_DATA[language])
    app_credentials = [
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ]
    values += app_credentials
    TEXT_MODEL_TITLE_IDS_TUPLE.append(tuple(values))

AUDIO_MODEL_TITLE_IDS_TUPLE = [
    ("english audio transcription", ENGLISH_ASR_MODEL_ID, "asr", "facebook")(
        "general-asr-nemo_jasper",
        GENERAL_ASR_NEMO_JASPER_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    )
]

@both_channels
def test_audio_predict_on_public_models(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    for title, model_id, app_id, user_id in AUDIO_MODEL_TITLE_IDS_TUPLE:
        request = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
            model_id=model_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        audio=resources_pb2.Audio(url=ENGLISH_AUDIO_URL)
                    )
                )
            ],
        )
        response = post_model_outputs_and_maybe_allow_retries(
            stub, request, metadata=metadata(pat=True)
        )
        raise_on_failure(
            response,
            custom_message=f"Audio predict failed for the {title} model (ID: {model_id}).",
        )

@both_channels
def test_text_predict_on_public_models(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    for title, model_id, text, app_id, user_id in TEXT_MODEL_TITLE_IDS_TUPLE:
        request = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
            model_id=model_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(text=resources_pb2.Text(raw=text))
                )
            ],
        )
        response = post_model_outputs_and_maybe_allow_retries(
            stub, request, metadata=metadata(pat=True)
        )
        raise_on_failure(
            response,
            custom_message=f"Text predict failed for the {title} model (ID: {model_id}).",
        )

@both_channels
def test_image_predict_on_public_models(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    for title, model_id in MODEL_TITLE_AND_ID_PAIRS:
        request = service_pb2.PostModelOutputsRequest(
            model_id=model_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(url=DOG_IMAGE_URL)
                    )
                )
            ],
        )
        response = post_model_outputs_and_maybe_allow_retries(
            stub, request, metadata=metadata()
        )
        raise_on_failure(
            response,
            custom_message=f"Image predict failed for the {title} model (ID: {model_id}).",
        )

@both_channels
def test_video_predict_on_public_models(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    title = "general"
    model_id = GENERAL_MODEL_ID

    request = service_pb2.PostModelOutputsRequest(
        model_id=model_id,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(video=resources_pb2.Video(url=BEER_VIDEO_URL))
            )
        ],
    )
    response = post_model_outputs_and_maybe_allow_retries(
        stub, request, metadata=metadata()
    )
    raise_on_failure(
        response,
        custom_message=f"Video predict failed for the {title} model (ID: {model_id}).",
    )
