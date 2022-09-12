import os

import pytest as pytest

from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc

from tests.common import (
    APPAREL_MODEL_ID,
    BEER_VIDEO_URL,
    COLOR_MODEL_ID,
    DOG_IMAGE_URL,
    ENGLISH_TEXT,
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
    SPANISH_TEXT,
    TEXT_GEN_MODEL_ID,
    TEXT_MULTILINGUAL_MODERATION_MODEL_ID,
    TEXT_SENTIMENT_MODEL_ID,
    TEXT_SUM_MODEL_ID,
    TEXTURES_AND_PATTERNS_MODEL_ID,
    TRANSLATE_ROMANCE_MODEL_ID,
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
    YOLOV6_S_MODEL_ID,
    YOLOV6_NANO_MODEL_ID,
    YOLOV6_TINY_MODEL_ID,
    YOLOV7_MODEL_ID,
    YOLOV7_E6_MODEL_ID,
    YOLOV7_W6_MODEL_ID,
    YOLOV7_D6_MODEL_ID,
    YOLOV7_E6E_MODEL_ID,
    YOLOV7_X_MODEL_ID,
    TRANSLATE_EN_SPANISH_MODEL_ID,
    TRANSLATE_GERMAN_EN_MODEL_ID,
    TRANSLATE_CHINESE_EN_MODEL_ID,
    TRANSLATE_ARABIC_EN_MODEL_ID,
    TRANSLATE_WELSH_EN_MODEL_ID,
    TRANSLATE_GERMAN_EN_FB_MODEL_ID,
    TRANSLATE_EN_GERMAN_FB_MODEL_ID,
    TRANSLATE_SPANISH_EN_FB_MODEL_ID,
    TRANSLATE_EN_SPANISH_FB_MODEL_ID,
    TRANSLATE_CHINESE_EN_FB_MODEL_ID,
    TRANSLATE_EN_CHINESE_FB_MODEL_ID,
    TRANSLATE_ARABIC_EN_FB_MODEL_ID,
    TRANSLATE_EN_ARABIC_FB_MODEL_ID,
    TRANSLATE_WELSH_EN_FB_MODEL_ID,
    TRANSLATE_EN_WELSH_FB_MODEL_ID,
    TRANSLATE_RUSSIAN_EN_MODEL_ID,
    TRANSLATE_EN_RUSSIAN_MODEL_ID,
    TRANSLATE_TURKISH_EN_MODEL_ID,
    TRANSLATE_EN_TURKISH_MODEL_ID,
    TRANSLATE_FRENCH_EN_MODEL_ID,
    TRANSLATE_EN_FRENCH_MODEL_ID,
    TRANSLATE_INDONESIAN_EN_MODEL_ID,
    TRANSLATE_EN_INDONESIAN_MODEL_ID,
    TRANSLATE_PORTUGESE_EN_MODEL_ID,
    TRANSLATE_EN_PORTUGESE_MODEL_ID,
    TRANSLATE_CZECH_EN_MODEL_ID,
    TRANSLATE_JAPANESE_EN_MODEL_ID,
    TRANSLATE_DANISH_EN_MODEL_ID,
    TRANSLATE_CATALAN_EN_MODEL_ID,
    TRANSLATE_BULGARIAN_EN_MODEL_ID,
    TRANSLATE_AFRIKAANS_EN_MODEL_ID,
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
    (
        "general-detector-yolov6s-coco",
        YOLOV6_S_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID")
    ),
    (
        "general-detector-yolov6n-coco",
        YOLOV6_NANO_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID")
    ),
    (
        "general-detector-yolov6tiny-coco",
        YOLOV6_TINY_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID")
    ),
    (
        "general-detector-yolov7-coco",
        YOLOV7_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID")
    ),
    (
        "general-image-detector-yolov7-e6-coco",
        YOLOV7_E6_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID")
    ),
    (
        "general-image-detector-yolov7-w6-coco",
        YOLOV7_W6_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID")
    ),
    (
        "general-image-detector-yolov7-d6-coco",
        YOLOV7_D6_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID")
    ),
    (
        "general-image-detector-yolov7-e6e-coco",
        YOLOV7_E6E_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID")
    ),
    (
        "general-image-detector-yolov7-x-coco",
        YOLOV7_X_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID")
    )
]


TEXT_MODEL_TITLE_IDS_TUPLE = [
    ("text summarization", TEXT_SUM_MODEL_ID, "summarization", "huggingface-research"),
    ("text generation", TEXT_GEN_MODEL_ID, "text-generation", "huggingface-research"),
    ("text sentiment", TEXT_SENTIMENT_MODEL_ID, "text-classification", "huggingface-research"),
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
    (
        "text-translation-romance-lang-english",
        TRANSLATE_ROMANCE_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "text-translation-english-spanish",
        TRANSLATE_EN_SPANISH_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "text-translation-german-english",
        TRANSLATE_GERMAN_EN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "text-translation-chinese-english",
        TRANSLATE_CHINESE_EN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "text-translation-arabic-english",
        TRANSLATE_ARABIC_EN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "text-translation-welsh-english",
        TRANSLATE_WELSH_EN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-russian-to-english-text",
        TRANSLATE_RUSSIAN_EN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-english-to-russian-text",
        TRANSLATE_EN_RUSSIAN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-turkish-to-english-text",
         TRANSLATE_TURKISH_EN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-english-to-turkish-text",
        TRANSLATE_EN_TURKISH_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-french-to-english-text",
        TRANSLATE_FRENCH_EN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-english-to-french-text",
        TRANSLATE_EN_FRENCH_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-indonesian-to-english-text",
        TRANSLATE_INDONESIAN_EN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-english-to-indonesian-text",
        TRANSLATE_EN_INDONESIAN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-german-to-english-text",
        TRANSLATE_GERMAN_EN_FB_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-english-to-german-text",
        TRANSLATE_EN_GERMAN_FB_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-spanish-to-english-text",
        TRANSLATE_SPANISH_EN_FB_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-english-to-spanish-text",
        TRANSLATE_EN_SPANISH_FB_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-chinese-to-english-text",
        TRANSLATE_CHINESE_EN_FB_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-english-to-chinese-text",
        TRANSLATE_EN_CHINESE_FB_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-arabic-to-english-text",
        TRANSLATE_ARABIC_EN_FB_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-english-to-arabic-text",
        TRANSLATE_EN_ARABIC_FB_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-welsh-to-english-text",
        TRANSLATE_WELSH_EN_FB_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-english-to-welsh-text",
        TRANSLATE_EN_WELSH_FB_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-portuguese-to-english-text",
        TRANSLATE_PORTUGESE_EN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "translation-english-to-portuguese-text",
        TRANSLATE_EN_PORTUGESE_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "text-translation-czech-english",
        TRANSLATE_CZECH_EN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "text-translation-japanese-english",
        TRANSLATE_JAPANESE_EN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "text-translation-danish-english",
        TRANSLATE_DANISH_EN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "text-translation-catalan-english",
        TRANSLATE_CATALAN_EN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "text-translation-bulgarian-english",
        TRANSLATE_BULGARIAN_EN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),
    (
        "text-translation-afrikaans-english",
        TRANSLATE_AFRIKAANS_EN_MODEL_ID,
        os.environ.get("CLARIFAI_APP_ID"),
        os.environ.get("CLARIFAI_USER_ID"),
    ),

]


AUDIO_MODEL_TITLE_IDS_TUPLE = [
    ("english audio transcription", ENGLISH_ASR_MODEL_ID, "asr", "facebook")
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


@pytest.mark.skip(
    reason="On Github Actions there's 'Model training had no data' error for some reason"
)
@both_channels
def test_text_predict_on_public_models(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    for title, model_id, app_id, user_id in TEXT_MODEL_TITLE_IDS_TUPLE:
        if title == "translate romance":
            request = service_pb2.PostModelOutputsRequest(
                user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
                model_id=model_id,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(text=resources_pb2.Text(raw=SPANISH_TEXT))
                    )
                ],
            )
        else:
            request = service_pb2.PostModelOutputsRequest(
                user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
                model_id=model_id,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(text=resources_pb2.Text(raw=ENGLISH_TEXT))
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


@pytest.mark.skip(
    reason="On Github Actions there's 'Model training had no data' error for some reason"
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
