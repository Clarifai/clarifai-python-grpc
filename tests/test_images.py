import time

from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from tests.common import both_channels, raise_on_failure, metadata

from tests.common import TRUCK_IMAGE_URL


@both_channels
def test_post_patch_delete_input(channel):
  stub = service_pb2_grpc.V2Stub(channel)

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
  post_response = stub.PostInputs(post_request, metadata=metadata())

  raise_on_failure(post_response)

  input_id = post_response.inputs[0].id

  try:
    while True:
      get_request = service_pb2.GetInputRequest(input_id=input_id)
      get_response = stub.GetInput(get_request, metadata=metadata())
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
    patch_response = stub.PatchInputs(patch_request, metadata=metadata())
    assert status_code_pb2.SUCCESS == patch_response.status.code
  finally:
    delete_request = service_pb2.DeleteInputRequest(input_id=input_id)
    delete_response = stub.DeleteInput(delete_request, metadata=metadata())
    assert status_code_pb2.SUCCESS == delete_response.status.code


