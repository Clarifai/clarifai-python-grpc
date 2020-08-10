from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from tests.common import (TRUCK_IMAGE_URL, both_channels, metadata, raise_on_failure,
                          wait_for_inputs_upload)


@both_channels
def test_post_patch_delete_input(channel):
  stub = service_pb2_grpc.V2Stub(channel)

  post_request = service_pb2.PostInputsRequest(inputs=[
      resources_pb2.Input(data=resources_pb2.Data(
          image=resources_pb2.Image(url=TRUCK_IMAGE_URL, allow_duplicate_url=True),
          concepts=[resources_pb2.Concept(id='some-concept')]))
  ])
  post_response = stub.PostInputs(post_request, metadata=metadata())

  raise_on_failure(post_response)

  input_id = post_response.inputs[0].id

  try:
    wait_for_inputs_upload(stub, metadata(), [input_id])

    patch_request = service_pb2.PatchInputsRequest(
        action='overwrite',
        inputs=[
            resources_pb2.Input(
                id=input_id,
                data=resources_pb2.Data(concepts=[resources_pb2.Concept(id='some-new-concept')]))
        ])
    patch_response = stub.PatchInputs(patch_request, metadata=metadata())
    assert status_code_pb2.SUCCESS == patch_response.status.code
  finally:
    delete_request = service_pb2.DeleteInputRequest(input_id=input_id)
    delete_response = stub.DeleteInput(delete_request, metadata=metadata())
    assert status_code_pb2.SUCCESS == delete_response.status.code
