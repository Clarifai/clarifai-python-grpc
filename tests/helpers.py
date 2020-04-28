import time

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2


def both_channels(func):
  """
  A decorator that runs the test first using the gRPC channel and then using the JSON channel.
  :param func: The test function.
  :return: A function wrapper.
  """
  def func_wrapper():
    channel = ClarifaiChannel.get_insecure_grpc_channel()
    func(channel)

    channel = ClarifaiChannel.get_json_channel()
    func(channel)
  return func_wrapper


def _wait_for_inputs_upload(stub, metadata, input_ids):
  for input_id in input_ids:
    while True:
      get_input_response = stub.GetInput(
        service_pb2.GetInputRequest(input_id=input_id),
        metadata=metadata
      )
      _raise_on_failure(get_input_response)
      if get_input_response.input.status.code == status_code_pb2.INPUT_DOWNLOAD_SUCCESS:
        break
      elif get_input_response.input.status.code in (status_code_pb2.INPUT_DOWNLOAD_PENDING,
                                                    status_code_pb2.INPUT_DOWNLOAD_IN_PROGRESS):
        time.sleep(1)
      else:
        error_message = (
          str(get_input_response.status.code) + " " +
          get_input_response.status.description + " " +
          get_input_response.status.details
        )
        raise Exception(
          f"Expected inputs to upload, but got {error_message}. Full response: {get_input_response}"
        )
  # At this point, all inputs have been downloaded successfully.


def _wait_for_model_trained(stub, metadata, model_id, model_version_id):
  while True:
    response = stub.GetModelVersion(
      service_pb2.GetModelVersionRequest(model_id=model_id, version_id=model_version_id),
      metadata=metadata
    )
    _raise_on_failure(response)
    if response.model_version.status.code == status_code_pb2.MODEL_TRAINED:
      break
    elif response.model_version.status.code in (status_code_pb2.MODEL_QUEUED_FOR_TRAINING,
                                                status_code_pb2.MODEL_TRAINING):
      time.sleep(1)
    else:
      error_message = (
        str(response.status.code) + " " +
        response.status.description + " " +
        response.status.details
      )
      raise Exception(
        f"Expected model to train, but got {error_message}. Full response: {response}"
      )
  # At this point, the model has successfully finished training.


def _wait_for_model_evaluated(stub, metadata, model_id, model_version_id):
  while True:
    response = stub.GetModelVersionMetrics(
      service_pb2.GetModelVersionMetricsRequest(model_id=model_id, version_id=model_version_id),
      metadata=metadata
    )
    _raise_on_failure(response)
    if response.model_version.metrics.status.code == status_code_pb2.MODEL_EVALUATED:
      break
    elif response.model_version.metrics.status.code in (status_code_pb2.MODEL_QUEUED_FOR_EVALUATION,
                                                        status_code_pb2.MODEL_EVALUATING):
      time.sleep(1)
    else:
      error_message = (
        str(response.status.code) + " " +
        response.status.description + " " +
        response.status.details
      )
      raise Exception(
        f"Expected model to evaluate, but got {error_message}. Full response: {response}"
      )
  # At this point, the model has successfully finished evaluation.


def _raise_on_failure(response):
  if response.status.code != status_code_pb2.SUCCESS:
    error_message = (
      str(response.status.code) + " " +
      response.status.description + " " +
      response.status.details
    )
    raise Exception(
      f"Received failure response `{error_message}`. Whole response object: {response}"
    )
