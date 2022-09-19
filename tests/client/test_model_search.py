from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import both_channels, metadata, raise_on_failure


@both_channels
def test_search_for_model(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    response = stub.PostModelsSearches(
        service_pb2.PostModelsSearchesRequest(
            model_query=resources_pb2.ModelQuery(name="*general*")
        ),
        metadata=metadata(),
    )
    raise_on_failure(response)
    assert len(response.models) > 0
    for m in response.models:
        assert "general" in m.name.lower()
