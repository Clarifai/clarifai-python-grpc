import os

from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc

from tests.common import (
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
    HELSINKINLP_TRANSLATION_MODELS,
    FACEBOOK_TRANSLATION_MODELS,
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

DETECTION_MODEL_TITLE_AND_IDS = []

# Add models in object_detection_models dict to model_id_pairs list
# older image tests use different model-ids not from the platform
for _, values in OBJECT_DETECTION_MODELS.items():
    DETECTION_MODEL_TITLE_AND_IDS.append(tuple(values))

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

TEXT_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE = []

# Map corresponding test data to each model in translation_models dict
# append app_id and user_id vars to model data and then add the data to
# the text_translation_model_title_id_data list of tuples
for key, values in FACEBOOK_TRANSLATION_MODELS.items():
    language = key.split("_")[0]
    values.append(TRANSLATION_TEST_DATA[language])
    app_credentials = [
        "translation",
        "facebook",
    ]
    values += app_credentials
    TEXT_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE.append(tuple(values))

for key, values in HELSINKINLP_TRANSLATION_MODELS.items():
    language = key.split("_")[0]
    values.append(TRANSLATION_TEST_DATA[language])
    app_credentials = [
        "translation",
        "helsinkinlp",
    ]
    values += app_credentials
    TEXT_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE.append(tuple(values))

AUDIO_MODEL_TITLE_IDS_TUPLE = [
    ("english audio transcription", ENGLISH_ASR_MODEL_ID, "asr", "facebook"),
    (
        "general-asr-nemo_jasper",
        GENERAL_ASR_NEMO_JASPER_MODEL_ID,
        "asr",
        "nvidia",
    ),
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
                    data=resources_pb2.Data(audio=resources_pb2.Audio(url=ENGLISH_AUDIO_URL))
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
    """Test non translation text/nlp models.
    All these models can take the same test text input.
    """
    stub = service_pb2_grpc.V2Stub(channel)

    for title, model_id, app_id, user_id in TEXT_MODEL_TITLE_IDS_TUPLE:
        request = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
            model_id=model_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(raw=TRANSLATION_TEST_DATA["EN"])
                    )
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
def test_text_translation_predict_on_public_models(channel):
    """Test language translation models.
    Each language-english translation has its own text input while
    all en-language translations use the same english text.
    """
    stub = service_pb2_grpc.V2Stub(channel)

    for (
        title,
        model_id,
        text,
        app_id,
        user_id,
    ) in TEXT_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE:
        request = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
            model_id=model_id,
            inputs=[
                resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=text)))
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
def test_image_detection_predict_on_public_models(channel):
    """Test object detection models using clarifai platform user
    and app id access credentials.
    """
    stub = service_pb2_grpc.V2Stub(channel)

    for title, model_id, app_id, user_id in DETECTION_MODEL_TITLE_AND_IDS:
        request = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
            model_id=model_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL))
                )
            ],
        )
        response = post_model_outputs_and_maybe_allow_retries(
            stub, request, metadata=metadata(pat=True)
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
    response = post_model_outputs_and_maybe_allow_retries(stub, request, metadata=metadata())
    raise_on_failure(
        response,
        custom_message=f"Video predict failed for the {title} model (ID: {model_id}).",
    )
