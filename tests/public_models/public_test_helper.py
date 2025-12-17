from tests.common import GENERAL_MODEL_ID, MODERATION_MODEL_ID

# Enterprise models to keep
APPAREL_MODEL_ID = "e0be3b9d6a454f0493ac3a30784001ff"
COLOR_MODEL_ID = "eeed0b6733a644cea07cf4c60f87ebb7"
FOOD_MODEL_ID = "bd367be194cf45149e75f01d59f77ba7"
GENERAL_EMBEDDING_MODEL_ID = "bbb5f41425b8468d9b7a554ff10f8581"
LOGO_MODEL_ID = "c443119bf2ed4da98487520d01a0b1e3"
NSFW_MODEL_ID = "e9576d86d2004ed1a38ba0cf39ecb4b1"
TEXTURES_AND_PATTERNS_MODEL_ID = "fbefb47f9fdb410e8ce14f24f54b47ff"
TRAVEL_MODEL_ID = "eee28c313d69466f836ab83287a54ed9"
WEDDING_MODEL_ID = "c386b7a870114f4a87477c0824499348"
LOGO_V2_MODEL_ID = "006764f775d210080d295e6ea1445f93"

# TODO: Why is this not being used? Is there a reason why this does not have a test?
# DEMOGRAPHICS_MODEL_ID = "c0c0ac362b03416da06ab3fa36fb58e3"

# Note(zeiler): cleaned up this list so that we're only testing a couple models and not scaling
# up/down triton all the time.
MODEL_TITLE_AND_ID_PAIRS = [
    # ("apparel", APPAREL_MODEL_ID),
    ("color", COLOR_MODEL_ID),
    # ("food", FOOD_MODEL_ID),
    ("general embedding", GENERAL_EMBEDDING_MODEL_ID),
    ("general", GENERAL_MODEL_ID),
    # ("logo", LOGO_MODEL_ID),
    ("moderation", MODERATION_MODEL_ID),
    # ("nsfw", NSFW_MODEL_ID),
    # ("textures and patterns", TEXTURES_AND_PATTERNS_MODEL_ID),
    # ("travel", TRAVEL_MODEL_ID),
    # ("wedding", WEDDING_MODEL_ID),
    # ("logo v2", LOGO_V2_MODEL_ID),
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
