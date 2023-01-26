import os
import time
from typing import Tuple

from grpc._channel import _Rendezvous

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai_grpc.grpc.api.status.status_pb2 import Status

MAX_RETRY_ATTEMPTS = 15

DOG_IMAGE_URL = "https://samples.clarifai.com/dog2.jpeg"
TRUCK_IMAGE_URL = "https://s3.amazonaws.com/samples.clarifai.com/red-truck.png"
TRAVEL_IMAGE_URL = "https://samples.clarifai.com/travel.jpg"
NON_EXISTING_IMAGE_URL = "http://example.com/non-existing.jpg"
RED_TRUCK_IMAGE_FILE_PATH = os.path.dirname(__file__) + "/assets/red-truck.png"

BEER_VIDEO_URL = "https://samples.clarifai.com/beer.mp4"
CONAN_GIF_VIDEO_URL = "https://samples.clarifai.com/3o6gb3kkXfLvdKEZs4.gif"
TOY_VIDEO_FILE_PATH = os.path.dirname(__file__) + "/assets/toy.mp4"

GENERAL_MODEL_ID = "aaa03c23b3724a16a56b629203edc62c"


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
            if get_input_response.input.status.code == status_code_pb2.INPUT_DOWNLOAD_SUCCESS:
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


def wait_for_model_trained(stub, metadata, model_id, model_version_id, user_app_id=None):
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
        if response.model_version.metrics.status.code == status_code_pb2.MODEL_EVALUATED:
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
    return _retry_on_504_on_non_prod(lambda: stub.PostModelOutputs(request, metadata=metadata))


def _retry_on_504_on_non_prod(func):
    """
    On non-prod, it's possible that PostModelOutputs will return a temporary 504 response.
    We don't care about those as long as, after a few seconds, the response is a success.
    """
    for i in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            response = func()
            if (
                len(response.outputs) > 0
                and response.outputs[0].status.code != status_code_pb2.RPC_REQUEST_TIMEOUT
            ):  # will want to retry
                break
        except _Rendezvous as e:
            grpc_base = os.environ.get("CLARIFAI_GRPC_BASE", "api.clarifai.com")
            if grpc_base == "api.clarifai.com":
                raise e

            if "status: 504" not in e._state.details and "10020 Failure" not in e._state.details:
                raise e

            if i == MAX_RETRY_ATTEMPTS:
                raise e

            print(f"Received 504, doing retry #{i}")
            time.sleep(1)
    return response
