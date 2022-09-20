import os
import time
from typing import Tuple

from grpc._channel import _Rendezvous

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai_grpc.grpc.api.status.status_pb2 import Status

DOG_IMAGE_URL = "https://samples.clarifai.com/dog2.jpeg"
TRUCK_IMAGE_URL = "https://s3.amazonaws.com/samples.clarifai.com/red-truck.png"
TRAVEL_IMAGE_URL = "https://samples.clarifai.com/travel.jpg"
NON_EXISTING_IMAGE_URL = "http://example.com/non-existing.jpg"
RED_TRUCK_IMAGE_FILE_PATH = os.path.dirname(__file__) + "/assets/red-truck.png"

BEER_VIDEO_URL = "https://samples.clarifai.com/beer.mp4"
CONAN_GIF_VIDEO_URL = "https://samples.clarifai.com/3o6gb3kkXfLvdKEZs4.gif"
TOY_VIDEO_FILE_PATH = os.path.dirname(__file__) + "/assets/toy.mp4"

ENGLISH_AUDIO_URL = "https://samples.clarifai.com/english_audio_sample.mp3"

TRANSLATION_TEST_DATA = {
    "ROMANCE": "No me apetece nada estudiar esta noche.",
    "EN": "My spanish test is tomorrow morning. I don't feel like studying tonight, but I must study.",
    "SPANISH": "No me apetece nada estudiar esta noche.",
    "GERMAN": "Ich habe heute Abend keine Lust zu lernen",
    "CHINESE": "我今晚不想學習",
    "ARABIC": "لا أشعر بالرغبة في الدراسة الليلة",
    "WELSH": "Dydw i ddim yn teimlo fel astudio heno.",
    "FRENCH": "Je n'ai pas envie d'étudier ce soir.",
    "RUSSIAN": "Я не хочу учиться сегодня вечером",
    "TURKISH": "bu gece ders çalışmak istemiyorum",
    "INDONESIAN": "Saya tidak merasa ingin belajar malam ini",
    "PORTUGESE": "Eu não sinto vontade de estudar esta noite",
    "CZECH": "Dnes večer se mi nechce učit",
    "JAPANESE": "今夜は勉強したくない",
    "DANISH": "Jeg har ikke lyst til at studere i aften",
    "CATALAN": "No tinc ganes d'estudiar aquesta nit",
    "BULGARIAN": "Тази вечер не ми се учи",
    "AFRIKAANS": "Ek is nie lus om vanaand te studeer nie",
    "CROATIAN": "Večeras mi se ne uči, ali moram učiti",
    "FINNISH": "En halua opiskella tänä iltana, mutta minun täytyy opiskella",
    "SWEDISH": "Jag känner inte för att plugga ikväll, men jag måste plugga",
    "NORWEGIAN": "Jeg har ikke lyst til å studere i kveld, men jeg må studere",
    "HINDI": "मेरा आज रात पढ़ने का मन नहीं है, लेकिन मुझे पढ़ना चाहिए",
    "URDU": "مجھے آج رات پڑھنا اچھا نہیں لگتا، لیکن مجھے ضرور پڑھنا چاہیے۔",
    "UKRAINIAN": "Мені не хочеться сьогодні вчитися, але я мушу вчитися",
    "VIETNAMESE": "Tôi không muốn học tối nay, nhưng tôi phải học",
    "POLISH": "Nie chce mi się dzisiaj studiować, ale muszę się uczyć",
    "ITALIAN": "Non ho voglia di studiare stasera, ma devo studiare",
    "KOREAN": "오늘 밤은 공부하기 싫지만 공부는 해야겠어",
    "IRISH": "Ní dóigh liom gur mhaith liom staidéar a dhéanamh anocht, ach caithfidh mé staidéar a dhéanamh",
    "THAI": "คืนนี้ไม่อยากเรียนแต่ต้องเรียน",
    "SWAHILI": "Sijisikii kusoma usiku wa leo, lakini lazima nisome",
    "DUTCH": "Ik heb geen zin om te studeren vanavond, maar ik moet studeren",
    "MALAGASY": "Tsy te hianatra aho anio alina, fa tsy maintsy mianatra",
    "BURMESE": "ဒီညတော့ စာကျက်ရမလို ခံစားရပေမယ့် စာကျက်ရမယ်။",
}

APPAREL_MODEL_ID = "e0be3b9d6a454f0493ac3a30784001ff"
COLOR_MODEL_ID = "eeed0b6733a644cea07cf4c60f87ebb7"
DEMOGRAPHICS_MODEL_ID = "c0c0ac362b03416da06ab3fa36fb58e3"
FACE_MODEL_ID = "e15d0f873e66047e579f90cf82c9882z"
FOOD_MODEL_ID = "bd367be194cf45149e75f01d59f77ba7"
GENERAL_EMBEDDING_MODEL_ID = "bbb5f41425b8468d9b7a554ff10f8581"
GENERAL_MODEL_ID = "aaa03c23b3724a16a56b629203edc62c"
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
EASYOCR_ENGLISH_MODEL_ID = "f1b1005c8feaa8d3f34d35f224092915"
PADDLEOCR_ENG_CHINESE_MODEL_ID = "dc09ac965f64826410fbd8fea603abe6"

# general visual detection models (yolo, detic)
# Data Structure: {MODEL_NAME: [<clarifai-id>, <clarifai-name>]}
OBJECT_DETECTION_MODELS = {
    "YOLOV6_S": ["general-detector-yolov6s-coco", "yolov6s-coco"],
    "YOLOV6_NANO": ["general-detector-yolov6n-coco", "yolov6n-coco"],
    "YOLOV6_TINY": ["general-detector-yolov6tiny-coco", "yolov6tiny-coco"],
    "YOLOV7": ["general-detector-yolov7-coco", "yolov7"],
    "YOLOV7_E6": ["general-image-detector-yolov7-e6-coco", "yolov7-e6"],
    "YOLOV7_W6": ["general-image-detector-yolov7-w6-coco", "yolov7-w6"],
    "YOLOV7_D6": ["general-image-detector-yolov7-d6-coco", "yolov7-d6"],
    "YOLOV7_E6E": ["general-image-detector-yolov7-e6e-coco", "yolov7-e6e"],
    "YOLOV7_X": ["general-image-detector-yolov7-x-coco", "yolov7-x"],
    "BLAZE_FACE_DETECTOR": [
        "general-image-detector-blazeface_ssh-widerface",
        "general-image-detector-blazeface_ssh-widerface",
    ],
    "DETIC_CLIP_R50": [
        "general-image-detector-detic_clipR50Caption-coco",
        "detic-clip-r50-1x_caption-CPU",
    ],
    "DETIC_C2_SWINB_LVIS": [
        "general-image-detector-detic_C2_SwinB_896_lvis",
        "general-image-detector-detic_C2_SwinB_896_lvis",
    ],
    "DETIC_C2_SWINB_COCO": [
        "general-image-detector-detic_C2_SwinB-21K_COCO",
        "general-image-detector-detic_C2_SwinB-21K_COCO",
    ],
    "DETIC_C2_IN_L_SWINB_LVIS": [
        "general-image-detector-detic_C2_IN_L_SwinB_lvis",
        "general-image-detector-detic_C2_IN_L_SwinB_lvis",
    ],
}

TEXT_SUM_MODEL_ID = "distilbart-cnn-12-6"
TEXT_GEN_MODEL_ID = "distilgpt2"
TEXT_SENTIMENT_MODEL_ID = "multilingual-uncased-sentiment"  # bert-based
TEXT_MULTILINGUAL_MODERATION_MODEL_ID = "bdcedc0f8da58c396b7df12f634ef923"
NER_ENGLISH_MODEL_ID = "ner_english_v2"

## LANGUAGE TRANSLATION

# Store these in a dict with model_id as key and a list of the
# clarifai name, and clarifai-id as values
### Dictionary Structure: {MODEL_NAME: [<clarifai-id>, <clarifai-name>]}

TRANSLATION_MODELS = {
    "ROMANCE_EN_MODEL": [
        "text-translation-romance-lang-english",
        "Text Translation: Romance to English",  # prev.TRANSLATE_<model_name>_MODEL_ID
<<<<<<< HEAD
    ],
    "EN_SPANISH_MODEL": [
        "text-translation-english-spanish",
        "Helsinki-NLP/opus-mt-en-es",
    ],
    "GERMAN_EN_MODEL": [
        "text-translation-german-english",
        "Helsinki-NLP/opus-mt-de-en",
    ],
    "CHINESE_EN_MODEL": [
        "text-translation-chinese-english",
        "Helsinki-NLP/opus-mt-zh-en",
    ],
    "ARABIC_EN_MODEL": [
        "text-translation-arabic-english",
        "Helsinki-NLP/opus-mt-ar-en",
    ],
=======
    ],
    "EN_SPANISH_MODEL": ["text-translation-english-spanish", "Helsinki-NLP/opus-mt-en-es"],
    "GERMAN_EN_MODEL": ["text-translation-german-english", "Helsinki-NLP/opus-mt-de-en"],
    "CHINESE_EN_MODEL": ["text-translation-chinese-english", "Helsinki-NLP/opus-mt-zh-en"],
    "ARABIC_EN_MODEL": ["text-translation-arabic-english", "Helsinki-NLP/opus-mt-ar-en"],
>>>>>>> 87b2d77384b0b4e9f73fe4b4d5645ddf3979855c
    "WELSH_EN_MODEL": ["text-translation-welsh-english", "Helsinki-NLP/opus-mt-cy-en"],
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
    "RUSSIAN_EN_MODEL": [
        "translation-russian-to-english-text",
        "translations-russian-to-english-text",
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
    "PORTUGESE_EN_MODEL": [
        "translation-portuguese-to-english-text",
        "translation-portuguese-to-english-text",
    ],
    "EN_PORTUGESE_MODEL": [
        "translation-english-to-portuguese-text",
        "translation-english-to-portuguese-text",
<<<<<<< HEAD
    ],
    "CZECH_EN_MODEL": ["text-translation-czech-english", "Helsinki-NLP/opus-mt-cs-en"],
    "JAPANESE_EN_MODEL": [
        "text-translation-japanese-english",
        "Helsinki-NLP/opus-mt-jap-en",
    ],
    "DANISH_EN_MODEL": [
        "text-translation-danish-english",
        "Helsinki-NLP/opus-mt-da-en",
    ],
    "CATALAN_EN_MODEL": [
        "text-translation-catalan-english",
        "Helsinki-NLP/opus-mt-ca-en",
    ],
    "BULGARIAN_EN_MODEL": [
        "text-translation-bulgarian-english",
        "Helsinki-NLP/opus-mt-bg-en",
    ],
    "AFRIKAANS_EN_MODEL": [
        "text-translation-afrikaans-english",
        "Helsinki-NLP/opus-mt-af-en",
=======
>>>>>>> 87b2d77384b0b4e9f73fe4b4d5645ddf3979855c
    ],
    "CZECH_EN_MODEL": ["text-translation-czech-english", "Helsinki-NLP/opus-mt-cs-en"],
    "JAPANESE_EN_MODEL": ["text-translation-japanese-english", "Helsinki-NLP/opus-mt-jap-en"],
    "DANISH_EN_MODEL": ["text-translation-danish-english", "Helsinki-NLP/opus-mt-da-en"],
    "CATALAN_EN_MODEL": ["text-translation-catalan-english", "Helsinki-NLP/opus-mt-ca-en"],
    "BULGARIAN_EN_MODEL": ["text-translation-bulgarian-english", "Helsinki-NLP/opus-mt-bg-en"],
    "AFRIKAANS_EN_MODEL": ["text-translation-afrikaans-english", "Helsinki-NLP/opus-mt-af-en"],
    "CROATIAN_EN_MODEL": [
        "translation-croatian-to-english-text",
        "translation-croatian-to-english-text",
    ],
    "EN_CROATIAN_MODEL": [
        "translation-english-to-croatian-text",
        "translation-english-to-croatian-text",
    ],
    "FINNISH_EN_MODEL": [
        "translation-finnish-to-english-text",
        "translation-finnish-to-english-text",
    ],
    "EN_FINNISH_MODEL": [
        "translation-english-to-finnish-text",
        "translation-english-to-finnish-text",
    ],
    "SWEDISH_EN_MODEL": [
        "translation-swedish-to-english-text",
        "translation-swedish-to-english-text",
    ],
    "EN_SWEDISH_MODEL": [
        "translation-english-to-swedish-text",
        "translation-english-to-swedish-text",
    ],
    "NORWEGIAN_EN_MODEL": [
        "translation-norwegian-to-english-text",
        "translation-norwegian-to-english-text",
    ],
    "EN_NORWEGIAN_MODEL": [
        "translation-english-to-norwegian-text",
        "translation-english-to-norwegian-text",
<<<<<<< HEAD
    ],
    "HINDI_EN_MODEL": [
        "translation-hindi-to-english-text",
        "translation-hindi-to-english-text",
    ],
    "EN_HINDI_MODEL": [
        "translation-english-to-hindi-text",
        "translation-english-to-hindi-text",
    ],
    "URDU_EN_MODEL": [
        "translation-urdu-to-english-text",
        "translation-urdu-to-english-text",
    ],
    "EN_URDU_MODEL": [
        "translation-english-to-urdu-text",
        "translation-english-to-urdu-text",
=======
>>>>>>> 87b2d77384b0b4e9f73fe4b4d5645ddf3979855c
    ],
    "HINDI_EN_MODEL": ["translation-hindi-to-english-text", "translation-hindi-to-english-text"],
    "EN_HINDI_MODEL": ["translation-english-to-hindi-text", "translation-english-to-hindi-text"],
    "URDU_EN_MODEL": ["translation-urdu-to-english-text", "translation-urdu-to-english-text"],
    "EN_URDU_MODEL": ["translation-english-to-urdu-text", "translation-english-to-urdu-text"],
    "UKRAINIAN_EN_MODEL": [
        "translation-ukrainian-to-english-text",
        "translation-ukrainian-to-english-text",
    ],
    "EN_UKRAINIAN_MODEL": [
        "translation-english-to-ukrainian-text",
        "translation-english-to-ukrainian-text",
    ],
    "VIETNAMESE_EN_MODEL": [
        "translation-vietnamese-to-english-text",
        "translation-vietnamese-to-english-text",
    ],
    "EN_VIETNAMESE_MODEL": [
        "translation-english-to-vietnamese-text",
        "translation-english-to-vietnamese-text",
    ],
    "POLISH_EN_MODEL": [
        "translation-polish-to-english-text",
        "translation-polish-to-english-text",
    ],
    "EN_POLISH_MODEL": [
        "translation-english-to-polish-text",
        "translation-english-to-polish-text",
    ],
    "ITALIAN_EN_MODEL": [
        "translation-italian-to-english-text",
        "translation-italian-to-english-text",
    ],
    "EN_ITALIAN_MODEL": [
        "translation-english-to-italian-text",
        "translation-english-to-italian-text",
    ],
    "KOREAN_EN_MODEL": [
        "translation-korean-to-english-text",
        "translation-korean-to-english-text",
    ],
    "EN_KOREAN_MODEL": [
        "translation-english-to-korean-text",
        "translation-english-to-korean-text",
<<<<<<< HEAD
    ],
    "IRISH_EN_MODEL": [
        "translation-irish-to-english-text",
        "translation-irish-to-english-text",
    ],
    "EN_IRISH_MODEL": [
        "translation-english-to-irish-text",
        "translation-english-to-irish-text",
    ],
    "THAI_EN_MODEL": [
        "translation-thai-to-english-text",
        "translation-thai-to-english-text",
    ],
    "EN_THAI_MODEL": [
        "translation-english-to-thai-text",
        "translation-english-to-thai-text",
=======
>>>>>>> 87b2d77384b0b4e9f73fe4b4d5645ddf3979855c
    ],
    "IRISH_EN_MODEL": ["translation-irish-to-english-text", "translation-irish-to-english-text"],
    "EN_IRISH_MODEL": ["translation-english-to-irish-text", "translation-english-to-irish-text"],
    "THAI_EN_MODEL": ["translation-thai-to-english-text", "translation-thai-to-english-text"],
    "EN_THAI_MODEL": ["translation-english-to-thai-text", "translation-english-to-thai-text"],
    "SWAHILI_EN_MODEL": [
        "translation-swahili-to-english-text",
        "translation-swahili-to-english-text",
    ],
    "EN_SWAHILI_MODEL": [
        "translation-english-to-swahili-text",
        "translation-english-to-swahili-text",
    ],
    "DUTCH_EN_MODEL": [
        "translation-dutch-flemish-to-english-text",
        "translation-dutch-flemish-to-english-text",
    ],
    "EN_DUTCH_MODEL": [
        "translation-english-to-dutch-flemish-text",
        "translation-english-to-dutch-flemish-text",
    ],
    "MALAGASY_EN_MODEL": [
        "translation-malagasy-to-english-text",
        "translation-malagasy-to-english-text",
    ],
    "EN_MALAGASY_MODEL": [
        "translation-english-to-malagasy-text",
        "translation-english-to-malagasy-text",
    ],
    "BURMESE_EN_MODEL": [
        "translation-burmese-to-english-text",
        "translation-burmese-to-english-text",
    ],
    "EN_BURMESE_MODEL": [
        "translation-english-to-burmese-text",
        "translation-english-to-burmese-text",
    ],
}

# ASR
ENGLISH_ASR_MODEL_ID = "asr-wav2vec2-base-960h-english"
GENERAL_ASR_NEMO_JASPER_MODEL_ID = "general-asr-nemo_jasper"


def get_status_message(status: Status):
    message = f"{status.code} {status.description}"
    if status.details:
        return f"{message} {status.details}"
    else:
        return message

def metadata(pat=False):
    if pat:
        return (("authorization", "Key %s" % os.environ.get("CLARIFAI_PAT_KEY")),)
    else:
        return (("authorization", "Key %s" % os.environ.get("CLARIFAI_API_KEY")),)

def both_channels(func):
    """
    A decorator that runs the test first using the gRPC channel and then using the JSON channel.
    :param func: The test function.
    :return: A function wrapper.
    """

    def func_wrapper():
        channel = ClarifaiChannel.get_grpc_channel()
        func(channel)

        channel = ClarifaiChannel.get_json_channel()
        func(channel)
    return func_wrapper

def wait_for_inputs_upload(stub, metadata, input_ids):
    for input_id in input_ids:
        while True:
            get_input_response = stub.GetInput(
                service_pb2.GetInputRequest(input_id=input_id), metadata=metadata
            )
            raise_on_failure(get_input_response)
            if (
                get_input_response.input.status.code
                == status_code_pb2.INPUT_DOWNLOAD_SUCCESS
            ):
                break
            elif get_input_response.input.status.code in (
                status_code_pb2.INPUT_DOWNLOAD_PENDING,
                status_code_pb2.INPUT_DOWNLOAD_IN_PROGRESS,
            ):
                time.sleep(1)
            else:
                error_message = get_status_message(get_input_response.status)
                raise Exception(
                    f"Expected inputs to upload, but got {error_message}. "
                    f"Full response: {get_input_response}"
                )
    # At this point, all inputs have been downloaded successfully.

def wait_for_model_trained(
    stub, metadata, model_id, model_version_id, user_app_id=None
):
    while True:
        response = stub.GetModelVersion(
            service_pb2.GetModelVersionRequest(
                user_app_id=user_app_id, model_id=model_id, version_id=model_version_id
            ),
            metadata=metadata,
        )
        raise_on_failure(response)
        if response.model_version.status.code == status_code_pb2.MODEL_TRAINED:
            break
        elif response.model_version.status.code in (
            status_code_pb2.MODEL_QUEUED_FOR_TRAINING,
            status_code_pb2.MODEL_TRAINING,
        ):
            time.sleep(1)
        else:
            message = get_status_message(response.model_version.status)
            raise Exception(
                f"Expected model to be trained, but got model status: {message}. Full response: {response}"
            )
    # At this point, the model has successfully finished training.

def wait_for_model_evaluated(stub, metadata, model_id, model_version_id):
    while True:
        response = stub.GetModelVersionMetrics(
            service_pb2.GetModelVersionMetricsRequest(
                model_id=model_id, version_id=model_version_id
            ),
            metadata=metadata,
        )
        raise_on_failure(response)
        if (
            response.model_version.metrics.status.code
            == status_code_pb2.MODEL_EVALUATED
        ):
            break
        elif response.model_version.metrics.status.code in (
            status_code_pb2.MODEL_NOT_EVALUATED,
            status_code_pb2.MODEL_QUEUED_FOR_EVALUATION,
            status_code_pb2.MODEL_EVALUATING,
        ):
            time.sleep(1)
        else:
            error_message = get_status_message(response.status)
            raise Exception(
                f"Expected model to evaluate, but got {error_message}. Full response: {response}"
            )
    # At this point, the model has successfully finished evaluation.

def raise_on_failure(response, custom_message=""):
    if response.status.code != status_code_pb2.SUCCESS:
        error_message = get_status_message(response.status)
        if custom_message:
            if not str.isspace(custom_message[-1]):
                custom_message += " "
        raise Exception(
            custom_message
            + f"Received failure response `{error_message}`. Whole response object: {response}"
        )

def post_model_outputs_and_maybe_allow_retries(
    stub: service_pb2_grpc.V2Stub,
    request: service_pb2.PostModelOutputsRequest,
    metadata: Tuple,
):
    return _retry_on_504_on_non_prod(
        lambda: stub.PostModelOutputs(request, metadata=metadata)
    )

def _retry_on_504_on_non_prod(func):
    """
    On non-prod, it's possible that PostModelOutputs will return a temporary 504 response.
    We don't care about those as long as, after a few seconds, the response is a success.
    """
    MAX_ATTEMPTS = 15
    for i in range(1, MAX_ATTEMPTS + 1):
        try:
            response = func()
            if (
                len(response.outputs) > 0
                and response.outputs[0].status.code
                != status_code_pb2.RPC_REQUEST_TIMEOUT
            ):  # will want to retry
                break
        except _Rendezvous as e:
            grpc_base = os.environ.get("CLARIFAI_GRPC_BASE")
            if not grpc_base or grpc_base == "api.clarifai.com":
                raise e

            if (
                "status: 504" not in e._state.details
                and "10020 Failure" not in e._state.details
            ):
                raise e

            if i == MAX_ATTEMPTS:
                raise e

            print(f"Received 504, doing retry #{i}")
            time.sleep(1)
    return response
