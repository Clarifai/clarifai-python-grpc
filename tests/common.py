import os
import time
from datetime import datetime
from typing import List, Tuple

import pytest
from grpc._channel import _Rendezvous

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.channel.http_client import CLIENT_VERSION
from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai_grpc.grpc.api.status.status_pb2 import Status

MAX_PREDICT_ATTEMPTS = 6  # PostModelOutputs unsuccessful predict retry limit
MAX_RETRY_ATTEMPTS = 15  # gRPC exceeded deadlines/timeout retry limit.

DOG_IMAGE_URL = "https://samples.clarifai.com/dog2.jpeg"
TRUCK_IMAGE_URL = "https://s3.amazonaws.com/samples.clarifai.com/red-truck.png"
TRAVEL_IMAGE_URL = "https://samples.clarifai.com/travel.jpg"
NON_EXISTING_IMAGE_URL = "http://example.com/non-existing.jpg"
RED_TRUCK_IMAGE_FILE_PATH = os.path.dirname(__file__) + "/assets/red-truck.png"

BEER_VIDEO_URL = "https://samples.clarifai.com/beer.mp4"
CONAN_GIF_VIDEO_URL = "https://samples.clarifai.com/3o6gb3kkXfLvdKEZs4.gif"
TOY_VIDEO_FILE_PATH = os.path.dirname(__file__) + "/assets/toy.mp4"

ARCHIVE_CLOUD_URL = "s3://samples.clarifai.com/Archive.zip"
CLOUD_URL = "s3://samples.clarifai.com/storage/"

MAIN_APP_ID = "main"
MAIN_APP_USER_ID = "clarifai"
GENERAL_MODEL_ID = "aaa03c23b3724a16a56b629203edc62c"


def get_status_message(status: Status):
    message = f"{status.code} {status.description}"
    if status.details:
        return f"{message} {status.details}"
    else:
        return message


def headers(pat=False):
    if pat:
        return {
            "authorization": "Key %s"
            % os.environ.get("CLARIFAI_PAT_KEY", os.environ.get("CLARIFAI_PAT"))
        }
    else:
        return {
            "authorization": "Key %s"
            % os.environ.get("CLARIFAI_API_KEY", os.environ.get("CLARIFAI_PAT"))
        }


def metadata(pat: bool = False) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    if pat:
        key = os.environ.get("CLARIFAI_PAT_KEY", os.environ.get("CLARIFAI_PAT"))
    else:
        key = os.environ.get("CLARIFAI_API_KEY", os.environ.get("CLARIFAI_PAT"))

    return (
        ("x-clarifai-request-id-prefix", f"python-grpc-{CLIENT_VERSION}"),
        ("authorization", f"Key {key}"),
    )


def get_channel(channel_key):
    if channel_key == "grpc":
        if os.getenv("CLARIFAI_GRPC_INSECURE", "False").lower() in ("true", "1", "t"):
            return ClarifaiChannel.get_insecure_grpc_channel(port=443)
        else:
            return ClarifaiChannel.get_grpc_channel()

    if channel_key == "json":
        return ClarifaiChannel.get_json_channel()

    if channel_key == "aio_grpc":
        if os.getenv("CLARIFAI_GRPC_INSECURE", "False").lower() in ("true", "1", "t"):
            return ClarifaiChannel.get_aio_insecure_grpc_channel(port=443)
        else:
            return ClarifaiChannel.get_aio_grpc_channel()

    raise ValueError(f"Unknown channel {channel_key}")


def grpc_channel():
    return pytest.mark.parametrize('channel_key', ["grpc"])


def both_channels():
    return pytest.mark.parametrize('channel_key', ["grpc", "json"])


def aio_grpc_channel():
    return pytest.mark.parametrize('channel_key', ["aio_grpc"])


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


def cleanup_inputs(stub, input_ids, metadata):
    delete_request = service_pb2.DeleteInputsRequest(ids=input_ids)
    delete_response = stub.DeleteInputs(delete_request, metadata=metadata)
    raise_on_failure(delete_response)
    wait_for_inputs_delete(stub, input_ids, metadata=metadata)


def wait_for_inputs_delete(stub, input_ids, metadata):
    remaining_input_ids = list(input_ids)
    start = datetime.now()
    timeout = 120
    while remaining_input_ids and (datetime.now() - start).total_seconds() < timeout:
        for input_id in remaining_input_ids:
            get_input_response = stub.GetInput(
                service_pb2.GetInputRequest(input_id=input_id), metadata=metadata
            )
            if get_input_response.status.code == status_code_pb2.CONN_DOES_NOT_EXIST:
                remaining_input_ids.remove(input_id)
            else:
                print(f"Waiting for input '{input_id}' to be deleted")
                time.sleep(1)
                break
    if (datetime.now() - start).total_seconds() >= timeout:
        raise Exception(f"Timeout after {timeout} seconds to delete inputs {remaining_input_ids}")


def wait_for_model_trained(
    stub, metadata, model_id, model_version_id, user_app_id=None, retry_on_internal_failure=False
):
    retry_count = 0
    while True:
        response = stub.GetModelVersion(
            service_pb2.GetModelVersionRequest(
                user_app_id=user_app_id, model_id=model_id, version_id=model_version_id
            ),
            metadata=metadata,
        )
        ## Temp fix for EAGLE-4649 until LT-5281 is implemented.
        if (
            retry_on_internal_failure
            and response.status.code == status_code_pb2.INTERNAL_UNCATEGORIZED
            and retry_count < 3
        ):
            print(f"Retrying: {response.status}")
            retry_count += 1
            continue

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


def wait_for_dataset_version_ready(stub, metadata, dataset_id, dataset_version_id):
    while True:
        response = stub.GetDatasetVersion(
            service_pb2.GetDatasetVersionRequest(
                dataset_id=dataset_id,
                dataset_version_id=dataset_version_id,
            ),
            metadata=metadata,
        )
        raise_on_failure(response)
        if response.dataset_version.status.code == status_code_pb2.DATASET_VERSION_READY:
            break
        elif response.dataset_version.status.code in (
            status_code_pb2.DATASET_VERSION_PENDING,
            status_code_pb2.DATASET_VERSION_IN_PROGRESS,
        ):
            time.sleep(1)
        else:
            error_message = get_status_message(response.dataset_version.status)
            raise Exception(
                f"Expected dataset version to be ready, but got {error_message}. Full response: {response}"
            )
    # At this point, the dataset version is ready.


def wait_for_dataset_version_export_success(
    stub, metadata, dataset_id, dataset_version_id, export_info_fields: List[str]
):
    while True:
        response = stub.GetDatasetVersion(
            service_pb2.GetDatasetVersionRequest(
                dataset_id=dataset_id,
                dataset_version_id=dataset_version_id,
            ),
            metadata=metadata,
        )
        raise_on_failure(response)

        for field in export_info_fields:
            if not response.dataset_version.export_info.HasField(field):
                raise Exception(
                    f"Missing expected dataset version export info field '{field}'. Full response: {response}"
                )
            export = getattr(response.dataset_version.export_info, field)
            if export.status.code == status_code_pb2.DATASET_VERSION_EXPORT_SUCCESS:
                continue
            elif export.status.code in (
                status_code_pb2.DATASET_VERSION_EXPORT_PENDING,
                status_code_pb2.DATASET_VERSION_EXPORT_IN_PROGRESS,
            ):
                time.sleep(1)
                break
            else:
                error_message = get_status_message(export.status)
                raise Exception(
                    f"Expected dataset version to export, but got {error_message}. Full response: {response}"
                )
        else:
            break  # break the while True
    # At this point, the dataset version has successfully finished exporting.


def wait_for_extraction_job_completed(stub: service_pb2_grpc.V2Stub, extraction_job_id: str):
    while True:
        response = stub.GetInputsExtractionJob(
            service_pb2.GetInputsExtractionJobRequest(inputs_extraction_job_id=extraction_job_id),
            metadata=metadata(),
        )
        raise_on_failure(response)
        if response.inputs_extraction_job.status.code == status_code_pb2.JOB_COMPLETED:
            return response
        elif response.inputs_extraction_job.status.code in (
            status_code_pb2.JOB_QUEUED,
            status_code_pb2.JOB_RUNNING,
        ):
            time.sleep(1)
        else:
            error_message = get_status_message(response.inputs_extraction_job.status)
            raise Exception(
                f"Expected extraction job to be completed, but got {error_message}. Full response: {response}"
            )


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


async def async_raise_on_failure(response, custom_message=""):
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
    # first make sure we don't run into GRPC timeout issues and that the API can be reached.
    response = _retry_on_504_on_non_prod(stub.PostModelOutputs, request=request, metadata=metadata)
    # retry when status of response is FAILURE
    response = _retry_on_unsuccessful_predicts_on_non_prod(
        stub.PostModelOutputs,
        request=request,
        metadata=metadata,
        response=response,
    )
    return response


def _generate_model_outputs(
    stub: service_pb2_grpc.V2Stub,
    request: service_pb2.PostModelOutputsRequest,
    metadata: Tuple,
):
    is_model_loaded = False
    for i in range(1, MAX_PREDICT_ATTEMPTS + 1):
        response_iterator = stub.GenerateModelOutputs(request, metadata=metadata)
        for response in response_iterator:
            if not is_model_loaded and response.status.code == status_code_pb2.MODEL_LOADING:
                print(f"Model {request.model_id} is still loading...")
                time.sleep(15)
                break
            is_model_loaded = True
            yield response
        if is_model_loaded:
            break


async def async_post_model_outputs_and_maybe_allow_retries(
    stub: service_pb2_grpc.V2Stub,
    request: service_pb2.PostModelOutputsRequest,
    metadata: Tuple,
):
    # first make sure we don't run into GRPC timeout issues and that the API can be reached.
    response = await _async_retry_on_504_on_non_prod(
        stub.PostModelOutputs, request=request, metadata=metadata
    )
    # retry when status of response is FAILURE
    response = await _async_retry_on_unsuccessful_predicts_on_non_prod(
        stub.PostModelOutputs,
        request=request,
        metadata=metadata,
        response=response,
    )
    return response


def _retry_on_unsuccessful_predicts_on_non_prod(stub_call, request, metadata, response):
    for i in range(1, MAX_PREDICT_ATTEMPTS + 1):
        if response.status.code not in [status_code_pb2.MODEL_DEPLOYING, status_code_pb2.FAILURE]:
            return response  # don't retry on non-FAILURE codes
        response = stub_call(request=request, metadata=metadata)
    return response


async def _async_retry_on_unsuccessful_predicts_on_non_prod(
    stub_call, request, metadata, response
):
    for i in range(1, MAX_PREDICT_ATTEMPTS + 1):
        if response.status.code not in [status_code_pb2.MODEL_DEPLOYING, status_code_pb2.FAILURE]:
            return response  # don't retry on non-FAILURE codes
        response = await stub_call(request=request, metadata=metadata)
    return response


def _retry_on_504_on_non_prod(stub_call, request, metadata):
    """
    On non-prod, it's possible that PostModelOutputs will return a temporary 504 response.
    We don't care about those as long as, after a few seconds, the response is a success.
    """
    for i in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            response = stub_call(request=request, metadata=metadata)
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


async def _async_retry_on_504_on_non_prod(stub_call, request, metadata):
    """
    On non-prod, it's possible that PostModelOutputs will return a temporary 504 response.
    We don't care about those as long as, after a few seconds, the response is a success.
    """
    for i in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            response = await stub_call(request=request, metadata=metadata)
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
