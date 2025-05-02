import uuid

from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from tests.common import both_channels, metadata, raise_on_failure


@both_channels
def test_concept_post_get_patch(channel):
  stub = service_pb2_grpc.V2Stub(channel)

  random_string = uuid.uuid4().hex[:15]
  random_concept_id = "concept-id-" + random_string
  random_concept_name = "concept-name-的な-" + random_string

  post_concepts_response = stub.PostConcepts(
    service_pb2.PostConceptsRequest(
      concepts=[resources_pb2.Concept(id=random_concept_id, name=random_concept_name)]
    ),
    metadata=metadata(),
  )
  raise_on_failure(post_concepts_response)

  get_concepts_response = stub.GetConcept(
    service_pb2.GetConceptRequest(concept_id=random_concept_id), metadata=metadata()
  )
  raise_on_failure(get_concepts_response)
  assert get_concepts_response.concept.id == random_concept_id
  assert get_concepts_response.concept.name == random_concept_name

  duplicated_post_concepts_response = stub.PostConcepts(
    service_pb2.PostConceptsRequest(
      concepts=[
        resources_pb2.Concept(
          id=random_concept_id,
        )
      ]
    ),
    metadata=metadata(),
  )
  assert (
    duplicated_post_concepts_response.status.code
    == status_code_pb2.StatusCode.CONCEPTS_INVALID_REQUEST
  )
  assert duplicated_post_concepts_response.status.description == "Invalid request"
  assert "duplicate" in duplicated_post_concepts_response.status.details.lower()

  post_concepts_searches_response = stub.PostConceptsSearches(
    service_pb2.PostConceptsSearchesRequest(
      concept_query=resources_pb2.ConceptQuery(name=random_concept_name)
    ),
    metadata=metadata(),
  )
  raise_on_failure(post_concepts_searches_response)
  assert random_concept_name in post_concepts_searches_response.concepts[0].name

  patch_concepts_response = stub.PatchConcepts(
    service_pb2.PatchConceptsRequest(
      action="overwrite",
      concepts=[resources_pb2.Concept(id=random_concept_id, name="some new concept name")],
    ),
    metadata=metadata(),
  )
  raise_on_failure(patch_concepts_response)


@both_channels
def test_patching_public_concept_fails(channel):
  stub = service_pb2_grpc.V2Stub(channel)

  patch_concepts_searches_response = stub.PatchConcepts(
    service_pb2.PatchConceptsRequest(
      action="overwrite",
      concepts=[
        resources_pb2.Concept(
          id="ai_98Xb0K3q",  # The ID of a public concept.
          name="this new name won't be applied",
        )
      ],
    ),
    metadata=metadata(),
  )
  assert (
    patch_concepts_searches_response.status.code == status_code_pb2.StatusCode.CONN_DOES_NOT_EXIST
  )
  assert patch_concepts_searches_response.status.description == "Resource does not exist"
