# Enterprise models to keep
APPAREL_MODEL_ID = "e0be3b9d6a454f0493ac3a30784001ff"
FOOD_MODEL_ID = "bd367be194cf45149e75f01d59f77ba7"
LOGO_MODEL_ID = "c443119bf2ed4da98487520d01a0b1e3"
LOGO_V2_MODEL_ID = "006764f775d210080d295e6ea1445f93"
TEXTURES_AND_PATTERNS_MODEL_ID = "fbefb47f9fdb410e8ce14f24f54b47ff"

MODEL_TITLE_AND_ID_PAIRS = [
    ("apparel", APPAREL_MODEL_ID),
    ("food", FOOD_MODEL_ID),
    ("logo", LOGO_MODEL_ID),
    ("logo v2", LOGO_V2_MODEL_ID),
    ("textures and patterns", TEXTURES_AND_PATTERNS_MODEL_ID),
]

# Constants needed for test imports (not used by deprecated models)
TRANSLATION_TEST_DATA = {
    "EN": "I don't feel like studying tonight but I must study",
}
ENGLISH_AUDIO_URL = "https://samples.clarifai.com/english_audio_sample.mp3"

# Empty lists for compatibility with existing test infrastructure
DETECTION_MODEL_TITLE_AND_IDS = []
MULTIMODAL_MODEL_TITLE_AND_IDS = []
TEXT_MODEL_TITLE_IDS_TUPLE = []
TEXT_LLM_MODEL_TITLE_IDS_TUPLE = []
TEXT_FB_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE = []
TEXT_HELSINKI_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE = []
AUDIO_MODEL_TITLE_IDS_TUPLE = []
