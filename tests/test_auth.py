from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from tests.common import both_channels


@both_channels
def test_invalid_api_key(channel):
  stub = service_pb2_grpc.V2Stub(channel)
  response = stub.ListModels(
    service_pb2.ListModelsRequest(),
    metadata=(('authorization', 'Key SOME_INVALID_KEY'),))

  assert response.status.code == status_code_pb2.StatusCode.CONN_KEY_INVALID
  assert response.status.description == "Invalid API key or Invalid API key/application pair"
