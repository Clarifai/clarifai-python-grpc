from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import both_channels, metadata, raise_on_failure


@both_channels
def test_search_public_concepts_in_english(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    post_concepts_searches_response = stub.PostConceptsSearches(
        service_pb2.PostConceptsSearchesRequest(
            concept_query=resources_pb2.ConceptQuery(name="dog*")
        ),
        metadata=metadata(),
    )
    raise_on_failure(post_concepts_searches_response)
    assert len(post_concepts_searches_response.concepts) > 0


@both_channels
def test_search_public_concepts_in_chinese(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    post_concepts_searches_response = stub.PostConceptsSearches(
        service_pb2.PostConceptsSearchesRequest(
            concept_query=resources_pb2.ConceptQuery(name="狗*", language="zh")
        ),
        metadata=metadata(),
    )
    raise_on_failure(post_concepts_searches_response)
    assert len(post_concepts_searches_response.concepts) > 0
