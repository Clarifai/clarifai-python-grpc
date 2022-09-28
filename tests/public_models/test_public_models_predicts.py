import os
import pytest

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
    both_channels,
    metadata,
    post_model_outputs_and_maybe_allow_retries,
    raise_on_failure,
)


TRANSLATION_TEST_DATA = {
    "ROMANCE": "No me apetece nada estudiar esta noche",
    "EN": "I dont feel like studying tonight but I must study",
    "SPANISH": "No me apetece nada estudiar esta noche",
    "GERMAN": "Ich habe heute Abend keine Lust zu lernen",
    "CHINESE": "我今晚不想學習",
    "ARABIC": "لا أشعر بالرغبة في الدراسة الليلة",
    "WELSH": "Dydw i ddim yn teimlo fel astudio heno",
    "FRENCH": "Je n'ai pas envie d'étudier ce soir",
    "RUSSIAN": "Я не хочу учиться сегодня вечером",
    "TURKISH": "bu gece ders çalışmak istemiyorum",
    "INDONESIAN": "Saya tidak merasa ingin belajar malam ini",
    "PORTUGESE": "Eu não sinto vontade de estudar esta noite",
    "CZECH": "Dnes večer se mi nechce učit",
    "JAPANESE": "今夜は勉強したくない",
    "DANISH": "Jeg har ikke lyst til at studere i aften",
}

# general visual detection models (yolo, detic)
# Data Structure: {MODEL_NAME: [<clarifai-id>, <clarifai-name>, <app_id>, <user_id>]}
OBJECT_DETECTION_MODELS = {
    "YOLOV6_S": [
        "general-detector-yolov6s-coco",
        "yolov6s-coco",
        "yolov6",
        "meituan",
    ],
    "YOLOV6_NANO": [
        "general-detector-yolov6n-coco",
        "yolov6n-coco",
        "yolov6",
        "meituan",
    ],
    "YOLOV7": ["general-detector-yolov7-coco", "yolov7", "yolov7", "wongkinyiu"],
    "YOLOV7_E6": [
        "general-image-detector-yolov7-e6-coco",
        "yolov7-e6",
        "yolov7",
        "wongkinyiu",
    ],
    "YOLOV7_X": [
        "general-image-detector-yolov7-x-coco",
        "yolov7-x",
        "yolov7",
        "wongkinyiu",
    ],
    "BLAZE_FACE_DETECTOR": [
        "general-image-detector-blazeface_ssh-widerface",
        "general-image-detector-blazeface_ssh-widerface",
        "face",
        "paddlepaddle",
    ],
    "DETIC_CLIP_R50": [
        "general-image-detector-detic_clipR50Caption-coco",
        "detic-clip-r50-1x_caption-CPU",
        "detic",
        "facebook",
    ],
    "DETIC_C2_SWINB_LVIS": [
        "general-image-detector-detic_C2_SwinB_896_lvis",
        "general-image-detector-detic_C2_SwinB_896_lvis",
        "detic",
        "facebook",
    ],
    "DETIC_C2_SWINB_COCO": [
        "general-image-detector-detic_C2_SwinB-21K_COCO",
        "general-image-detector-detic_C2_SwinB-21K_COCO",
        "detic",
        "facebook",
    ],
    "DETIC_C2_IN_L_SWINB_LVIS": [
        "general-image-detector-detic_C2_IN_L_SwinB_lvis",
        "general-image-detector-detic_C2_IN_L_SwinB_lvis",
        "detic",
        "facebook",
    ],
}
SHORT_OBJECT_DETECTION_MODEL_KEYS = ["YOLOV6_S", "YOLOV7", "DETIC_CLIP_R50"]
OBJECT_DETECTION_MODELS_SHORT = {
    k: OBJECT_DETECTION_MODELS[k] for k in SHORT_OBJECT_DETECTION_MODEL_KEYS
}

## LANGUAGE TRANSLATION

# Store these in a dict with model_id as key and a list of the
# clarifai name, and clarifai-id as values
### Dictionary Structure: {MODEL_NAME: [<clarifai-id>, <clarifai-name>]}

HELSINKINLP_TRANSLATION_MODELS = {
    "ROMANCE_EN_MODEL": [
        "Text Translation: Romance to English",
        "text-translation-romance-lang-english",
    ],
    "EN_SPANISH_MODEL": [
        "Helsinki-NLP/opus-mt-en-es",
        "text-translation-english-spanish",
    ],
    "GERMAN_EN_MODEL": [
        "Helsinki-NLP/opus-mt-de-en",
        "text-translation-german-english",
    ],
    "CHINESE_EN_MODEL": [
        "Helsinki-NLP/opus-mt-zh-en",
        "text-translation-chinese-english",
    ],
    "ARABIC_EN_MODEL": [
        "Helsinki-NLP/opus-mt-ar-en",
        "text-translation-arabic-english",
    ],
    "WELSH_EN_MODEL": [
        "Helsinki-NLP/opus-mt-cy-en",
        "text-translation-welsh-english",
    ],
    "CZECH_EN_MODEL": [
        "Helsinki-NLP/opus-mt-cs-en",
        "text-translation-czech-english",
    ],
    "JAPANESE_EN_MODEL": [
        "Helsinki-NLP/opus-mt-jap-en",
        "text-translation-japanese-english",
    ],
    "DANISH_EN_MODEL": [
        "Helsinki-NLP/opus-mt-da-en",
        "text-translation-danish-english",
    ],
}

FACEBOOK_TRANSLATION_MODELS = {
    "GERMAN_EN_FB_MODEL": [
        "translation-german-to-english-text",
        "translation-german-to-english-text",
    ],
    "EN_GERMAN_FB_MODEL": [
        "translation-english-to-german-text",
        "translation-english-to-german-text",
    ],
    "SPANISH_EN_FB_MODEL": [
        "translation-spanish-to-english-text",
        "translation-spanish-to-english-text",
    ],
    "EN_SPANISH_FB_MODEL": [
        "translation-english-to-spanish-text",
        "translation-english-to-spanish-text",
    ],
    "CHINESE_EN_FB_MODEL": [
        "translation-chinese-to-english-text",
        "translation-chinese-to-english-text",
    ],
    "EN_CHINESE_FB_MODEL": [
        "translation-english-to-chinese-text",
        "translation-english-to-chinese-text",
    ],
    "RUSSIAN_EN_MODEL": [
        "translation-russian-to-english-text",
        "translation-russian-to-english-text",
    ],
    "EN_RUSSIAN_MODEL": [
        "translation-english-to-russian-text",
        "translation-english-to-russian-text",
    ],
    "TURKISH_EN_MODEL": [
        "translation-turkish-to-english-text",
        "translation-turkish-to-english-text",
    ],
    "EN_TURKISH_MODEL": [
        "translation-english-to-turkish-text",
        "translation-english-to-turkish-text",
    ],
    "FRENCH_EN_MODEL": [
        "translation-french-to-english-text",
        "translation-french-to-english-text",
    ],
    "EN_FRENCH_MODEL": [
        "translation-english-to-french-text",
        "translation-english-to-french-text",
    ],
    "INDONESIAN_EN_MODEL": [
        "translation-indonesian-to-english-text",
        "translation-indonesian-to-english-text",
    ],
    "EN_INDONESIAN_MODEL": [
        "translation-english-to-indonesian-text",
        "translation-english-to-indonesian-text",
    ],
    "ARABIC_EN_FB_MODEL": [
        "translation-arabic-to-english-text",
        "translation-arabic-to-english-text",
    ],
    "EN_ARABIC_FB_MODEL": [
        "translation-english-to-arabic-text",
        "translation-english-to-arabic-text",
    ],
    "WELSH_EN_FB_MODEL": [
        "translation-welsh-to-english-text",
        "translation-welsh-to-english-text",
    ],
    "EN_WELSH_FB_MODEL": [
        "translation-english-to-welsh-text",
        "translation-english-to-welsh-text",
    ],
    "PORTUGESE_EN_MODEL": [
        "translation-portuguese-to-english-text",
        "translation-portuguese-to-english-text",
    ],
    "EN_PORTUGESE_MODEL": [
        "translation-english-to-portuguese-text",
        "translation-english-to-portuguese-text",
    ],
}


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
for _, values in OBJECT_DETECTION_MODELS_SHORT.items():
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

# title, model_id, text, app, user
TEXT_FB_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE = []
TEXT_HELSINKI_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE = []

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
    TEXT_FB_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE.append(tuple(values))


for key, values in HELSINKINLP_TRANSLATION_MODELS.items():
    language = key.split("_")[0]
    values.append(TRANSLATION_TEST_DATA[language])
    app_credentials = [
        "translation",
        "helsinkinlp",
    ]
    values += app_credentials
    TEXT_HELSINKI_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE.append(tuple(values))

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


@pytest.mark.skip(reason="This test is ready, but will be added in time")
@both_channels
def test_text_fb_translation_predict_on_public_models(channel):
    """Test language translation models.
    Each language-english translation has its own text input while
    all en-language translations use the same english text.
    """
    stub = service_pb2_grpc.V2Stub(channel)
    for title, model_id, text, app_id, user_id in TEXT_FB_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE:
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


@pytest.mark.skip(reason="This test is ready, but will be added in time")
@both_channels
def test_text_helsinki_translation_predict_on_public_models(channel):
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
    ) in TEXT_HELSINKI_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE:
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
