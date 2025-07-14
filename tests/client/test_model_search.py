from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import both_channels, get_channel, metadata, raise_on_failure


@both_channels()
def test_search_public_models(channel_key):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

    response = stub.PostModelsSearches(
        service_pb2.PostModelsSearchesRequest(
            user_app_id=resources_pb2.UserAppIDSet(
                user_id="clarifai",
                app_id="main",
            ),
            model_query=resources_pb2.ModelQuery(name="*general*"),
        ),
        metadata=metadata(pat=True),
    )
    raise_on_failure(response)
    assert len(response.models) > 0
    for m in response.models:
        assert "general" in m.name.lower()
