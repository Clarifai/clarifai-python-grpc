import os

from tests.common import GENERAL_MODEL_ID

APPAREL_MODEL_ID = "e0be3b9d6a454f0493ac3a30784001ff"
COLOR_MODEL_ID = "eeed0b6733a644cea07cf4c60f87ebb7"
FOOD_MODEL_ID = "bd367be194cf45149e75f01d59f77ba7"
GENERAL_EMBEDDING_MODEL_ID = "bbb5f41425b8468d9b7a554ff10f8581"
LANDSCAPE_QUALITY_MODEL_ID = "bec14810deb94c40a05f1f0eb3c91403"
LOGO_MODEL_ID = "c443119bf2ed4da98487520d01a0b1e3"
MODERATION_MODEL_ID = "d16f390eb32cad478c7ae150069bd2c6"
NSFW_MODEL_ID = "e9576d86d2004ed1a38ba0cf39ecb4b1"
PORTRAIT_QUALITY_MODEL_ID = "de9bd05cfdbf4534af151beb2a5d0953"
TEXTURES_AND_PATTERNS_MODEL_ID = "fbefb47f9fdb410e8ce14f24f54b47ff"
TRAVEL_MODEL_ID = "eee28c313d69466f836ab83287a54ed9"
WEDDING_MODEL_ID = "c386b7a870114f4a87477c0824499348"
LOGO_V2_MODEL_ID = "006764f775d210080d295e6ea1445f93"
PEOPLE_DETECTION_YOLOV5_MODEL_ID = "23aa4f9c9767a2fd61e63c55a73790ad"
GENERAL_ENGLISH_IMAGE_CAPTION_CLIP_MODEL_ID = "86039c857a206810679f7f72b82fff54"
IMAGE_SUBJECT_SEGMENTATION_MODEL_ID = "6a3dc529acf3f720a629cdc8c6ad41a9"
PADDLEOCR_ENG_CHINESE_MODEL_ID = "dc09ac965f64826410fbd8fea603abe6"
MULTIMODAL_CLIP_EMBED_MODEL_ID = "6bfff3d825ad4fc790713b0fb593fc68"

# TODO: Why is this not being used? Is there a reason why this does not have a test?
DEMOGRAPHICS_MODEL_ID = "c0c0ac362b03416da06ab3fa36fb58e3"

ENGLISH_AUDIO_URL = "https://samples.clarifai.com/english_audio_sample.mp3"
ENGLISH_ASR_MODEL_ID = "asr-wav2vec2-base-960h-english"
GENERAL_ASR_NEMO_JASPER_MODEL_ID = "general-asr-nemo_jasper"

NER_ENGLISH_MODEL_ID = "ner_english_v2"
TEXT_SUM_MODEL_ID = "distilbart-cnn-12-6"
TEXT_SENTIMENT_MODEL_ID = "multilingual-uncased-sentiment"  # bert-based
TEXT_MULTILINGUAL_MODERATION_MODEL_ID = "bdcedc0f8da58c396b7df12f634ef923"


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
TEXT_GEN_PROMPT = "What are the main events that led to the American Revolution?"

# general visual detection models (yolo, detic)
# Data Structure: {MODEL_NAME: [title, model, app, user]}
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
        "Helsinki-NLP/opus-mt-ROMANCE-en",
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
    "JAPANESE_EN_MODEL": [
        "Helsinki-NLP/opus-mt-jap-en",
        "text-translation-japanese-english",
    ],
}

FACEBOOK_TRANSLATION_MODELS = {
    "SPANISH_EN_FB_MODEL": [
        "translation-spanish-to-english-text",
        "translation-spanish-to-english-text",
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
    "FRENCH_EN_MODEL": [
        "translation-french-to-english-text",
        "translation-french-to-english-text",
    ],
    "EN_FRENCH_MODEL": [
        "translation-english-to-french-text",
        "translation-english-to-french-text",
    ],
    "ARABIC_EN_FB_MODEL": [
        "translation-arabic-to-english-text",
        "translation-arabic-to-english-text",
    ],
    "EN_ARABIC_FB_MODEL": [
        "translation-english-to-arabic-text",
        "translation-english-to-arabic-text",
    ],
}

# Large Language Models
# Data Structure: [(model_id, model_name, app_id, user_id),...]
LLM_TITLE_ID_TUPLES = [
    (
        "Llama2-7b-chat",
        "Llama2-7b-chat",
        "Llama-2",
        "meta-llama"
    ),
    (
        "claude-v2",
        "claude-v2",
        "completion",
        "anthropic"
    ),
    (
        "tiiuae-falcon-7b-instruct",
        "tiiuae-falcon-7b-instruct",
        "LLM-OpenSource-Models-Training-Inference-Test",
        "clarifai"
    ),
    (
        "hkunlp_instructor-xl",
        "hkunlp_instructor-xl",
        "LLM-OpenSource-Models-Training-Inference-Test",
        "clarifai"
    )
]

MODEL_TITLE_AND_ID_PAIRS = [
    ("apparel", APPAREL_MODEL_ID),
    ("color", COLOR_MODEL_ID),
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
    ("paddleocr english chinese", PADDLEOCR_ENG_CHINESE_MODEL_ID),
]

# title, model, app, user
DETECTION_MODEL_TITLE_AND_IDS = []

MULTIMODAL_MODEL_TITLE_AND_IDS = [
    ("multimodal clip embed", MULTIMODAL_CLIP_EMBED_MODEL_ID),
]

# Add models in object_detection_models dict to model_id_pairs list
# older image tests use different model-ids not from the platform
for _, values in OBJECT_DETECTION_MODELS_SHORT.items():
    DETECTION_MODEL_TITLE_AND_IDS.append(tuple(values))

TEXT_MODEL_TITLE_IDS_TUPLE = [
    ("text summarization", TEXT_SUM_MODEL_ID, "summarization", "hcs"),
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
    # TODO: Will be added later
    # (
    #     "general-asr-nemo_jasper",
    #     GENERAL_ASR_NEMO_JASPER_MODEL_ID,
    #     "asr",
    #     "nvidia",
    # ),
]
