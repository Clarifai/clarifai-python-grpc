import os
import time
from typing import Tuple

from grpc._channel import _Rendezvous

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

DOG_IMAGE_URL = "https://samples.clarifai.com/dog2.jpeg"
TRUCK_IMAGE_URL = "https://s3.amazonaws.com/samples.clarifai.com/red-truck.png"
TRAVEL_IMAGE_URL = "https://samples.clarifai.com/travel.jpg"
NON_EXISTING_IMAGE_URL = "http://example.com/non-existing.jpg"
RED_TRUCK_IMAGE_FILE_PATH = os.path.dirname(__file__) + "/assets/red-truck.png"

BEER_VIDEO_URL = "https://samples.clarifai.com/beer.mp4"
CONAN_GIF_VIDEO_URL = "https://samples.clarifai.com/3o6gb3kkXfLvdKEZs4.gif"
TOY_VIDEO_FILE_PATH = os.path.dirname(__file__) + "/assets/toy.mp4"

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
LOGO_V2_MODEL_ID= "006764f775d210080d295e6ea1445f93"
PEOPLE_DETECTION_YOLOV5_MODEL_ID = "23aa4f9c9767a2fd61e63c55a73790ad"
GENERAL_ENGLISH_IMAGE_CAPTION_CLIP_MODEL_ID = "86039c857a206810679f7f72b82fff54"
IMAGE_SUBJECT_SEGMENTATION_MODEL_ID = "6a3dc529acf3f720a629cdc8c6ad41a9"
EASYOCR_ENGLISH_MODEL_ID = "f1b1005c8feaa8d3f34d35f224092915"
PADDLEOCR_MULTILINGUAL_MODEL_ID = "d05c045b95d85241c7d79e1ed3da3f8e"


def metadata():
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
            if get_input_response.input.status.code == status_code_pb2.INPUT_DOWNLOAD_SUCCESS:
                break
            elif get_input_response.input.status.code in (
                status_code_pb2.INPUT_DOWNLOAD_PENDING,
                status_code_pb2.INPUT_DOWNLOAD_IN_PROGRESS,
            ):
                time.sleep(1)
            else:
                error_message = (
                    str(get_input_response.status.code)
                    + " "
                    + get_input_response.status.description
                    + " "
                    + get_input_response.status.details
                )
                raise Exception(
                    f"Expected inputs to upload, but got {error_message}. "
                    f"Full response: {get_input_response}"
                )
    # At this point, all inputs have been downloaded successfully.


def wait_for_model_trained(stub, metadata, model_id, model_version_id):
    while True:
        response = stub.GetModelVersion(
            service_pb2.GetModelVersionRequest(model_id=model_id, version_id=model_version_id),
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
            error_message = (
                str(response.status.code)
                + " "
                + response.status.description
                + " "
                + response.status.details
            )
            raise Exception(
                f"Expected model to train, but got {error_message}. Full response: {response}"
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
        if response.model_version.metrics.status.code == status_code_pb2.MODEL_EVALUATED:
            break
        elif response.model_version.metrics.status.code in (
            status_code_pb2.MODEL_NOT_EVALUATED,
            status_code_pb2.MODEL_QUEUED_FOR_EVALUATION,
            status_code_pb2.MODEL_EVALUATING,
        ):
            time.sleep(1)
        else:
            error_message = (
                str(response.status.code)
                + " "
                + response.status.description
                + " "
                + response.status.details
            )
            raise Exception(
                f"Expected model to evaluate, but got {error_message}. Full response: {response}"
            )
    # At this point, the model has successfully finished evaluation.


def raise_on_failure(response, custom_message=""):
    if response.status.code != status_code_pb2.SUCCESS:
        error_message = (
            str(response.status.code)
            + " "
            + response.status.description
            + " "
            + response.status.details
        )
        if custom_message:
            if not str.isspace(custom_message[-1]):
                custom_message += " "
        raise Exception(
            custom_message
            + f"Received failure response `{error_message}`. Whole response object: {response}"
        )


def post_model_outputs_and_maybe_allow_retries(
    stub: service_pb2_grpc.V2Stub, request: service_pb2.PostModelOutputsRequest, metadata: Tuple
):
    return _retry_on_504_on_non_prod(lambda: stub.PostModelOutputs(request, metadata=metadata))


def _retry_on_504_on_non_prod(func):
    """
    On non-prod, it's possible that PostModelOutputs will return a temporary 504 response.
    We don't care about those as long as, after a few seconds, the response is a success.
    """
    MAX_ATTEMPTS = 4
    for i in range(1, MAX_ATTEMPTS + 1):
        try:
            response = func()
            break
        except _Rendezvous as e:
            grpc_base = os.environ.get("CLARIFAI_GRPC_BASE")
            if not grpc_base or grpc_base == "api.clarifai.com":
                raise e

            if "status: 504" not in e._state.details:
                raise e

            if i == MAX_ATTEMPTS:
                raise e

            print(f"Received 504, doing retry #{i}")
    return response
