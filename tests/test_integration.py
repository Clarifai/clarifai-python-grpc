import os
import time

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

DOG_IMAGE_URL = 'https://samples.clarifai.com/dog2.jpeg'
TRUCK_IMAGE_URL = "https://s3.amazonaws.com/samples.clarifai.com/red-truck.png"


def test_post_model_outputs_on_json_channel():
  _assert_post_model_outputs(ClarifaiChannel.get_json_channel())


def test_post_model_outputs_on_grpc_channel():
  _assert_post_model_outputs(ClarifaiChannel.get_insecure_grpc_channel())


def test_post_patch_delete_input_on_json_channel():
  _assert_post_patch_delete_input(ClarifaiChannel.get_json_channel())


def test_post_patch_delete_input_on_grpc_channel():
  _assert_post_patch_delete_input(ClarifaiChannel.get_insecure_grpc_channel())


def _assert_post_model_outputs(channel):
  stub = service_pb2_grpc.V2Stub(channel)
  request = service_pb2.PostModelOutputsRequest(
    model_id='aaa03c23b3724a16a56b629203edc62c',
    inputs=[
      resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL)))
    ])
  metadata = (('authorization', 'Key %s' % os.environ.get('CLARIFAI_API_KEY')),)
  response = stub.PostModelOutputs(request, metadata=metadata)
  assert len(response.outputs[0].data.concepts) > 0


def _assert_post_patch_delete_input(channel):
  stub = service_pb2_grpc.V2Stub(channel)
  metadata = (('authorization', 'Key %s' % os.environ.get('CLARIFAI_API_KEY')),)

  post_request = service_pb2.PostInputsRequest(
    inputs=[
      resources_pb2.Input(
        data=resources_pb2.Data(
          image=resources_pb2.Image(
            url=TRUCK_IMAGE_URL, allow_duplicate_url=True
          ),
          concepts=[resources_pb2.Concept(id='some-concept')]
        )
      )
    ]
  )
  post_response = stub.PostInputs(post_request, metadata=metadata)
  assert status_code_pb2.SUCCESS == post_response.status.code

  input_id = post_response.inputs[0].id

  try:
    while True:
      get_request = service_pb2.GetInputRequest(input_id=input_id)
      get_response = stub.GetInput(get_request, metadata=metadata)
      status_code = get_response.input.status.code
      if status_code == status_code_pb2.INPUT_DOWNLOAD_SUCCESS:
        break
      elif status_code not in (
          status_code_pb2.INPUT_DOWNLOAD_PENDING,
          status_code_pb2.INPUT_DOWNLOAD_IN_PROGRESS
      ):
        raise Exception(
          f'Waiting for input ID {input_id} failed, status code is {status_code}.')
      time.sleep(0.2)

    patch_request = service_pb2.PatchInputsRequest(
      action='overwrite',
      inputs=[
        resources_pb2.Input(
          id=input_id,
          data=resources_pb2.Data(concepts=[resources_pb2.Concept(id='some-new-concept')])
        )
      ]
    )
    patch_response = stub.PatchInputs(patch_request, metadata=metadata)
    assert status_code_pb2.SUCCESS == patch_response.status.code
  finally:
    delete_request = service_pb2.DeleteInputRequest(input_id=input_id)
    delete_response = stub.DeleteInput(delete_request, metadata=metadata)
    assert status_code_pb2.SUCCESS == delete_response.status.code
