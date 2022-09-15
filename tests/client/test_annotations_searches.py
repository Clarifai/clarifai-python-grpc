from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.resources_pb2 import Search
from tests.common import (
    raise_on_failure,
    wait_for_inputs_upload,
    metadata,
    both_channels,
    TRAVEL_IMAGE_URL,
)


@both_channels
def test_post_annotations_searches(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    palm_search_response = stub.PostConceptsSearches(
        service_pb2.PostConceptsSearchesRequest(
            concept_query=resources_pb2.ConceptQuery(name="palm")
        ),
        metadata=metadata(),
    )
    raise_on_failure(palm_search_response)
    palm_concept_id = palm_search_response.concepts[0].id

    water_search_response = stub.PostConceptsSearches(
        service_pb2.PostConceptsSearchesRequest(
            concept_query=resources_pb2.ConceptQuery(name="water")
        ),
        metadata=metadata(),
    )
    raise_on_failure(water_search_response)
    water_concept_id = water_search_response.concepts[0].id

    with SetupImage(stub) as input_:
        post_palm_annotations_response = stub.PostAnnotations(
            service_pb2.PostAnnotationsRequest(
                annotations=[
                    resources_pb2.Annotation(
                        input_id=input_.id,
                        data=resources_pb2.Data(
                            regions=[
                                resources_pb2.Region(
                                    region_info=resources_pb2.RegionInfo(
                                        bounding_box=resources_pb2.BoundingBox(
                                            top_row=0, left_col=0, bottom_row=0.45, right_col=1
                                        )
                                    ),
                                    data=resources_pb2.Data(
                                        concepts=[
                                            resources_pb2.Concept(id=palm_concept_id, value=1)
                                        ]
                                    ),
                                ),
                            ]
                        ),
                    ),
                ]
            ),
            metadata=metadata(),
        )
        raise_on_failure(post_palm_annotations_response)

        post_water_annotations_response = stub.PostAnnotations(
            service_pb2.PostAnnotationsRequest(
                annotations=[
                    resources_pb2.Annotation(
                        input_id=input_.id,
                        data=resources_pb2.Data(
                            regions=[
                                resources_pb2.Region(
                                    region_info=resources_pb2.RegionInfo(
                                        bounding_box=resources_pb2.BoundingBox(
                                            top_row=0.6, left_col=0, bottom_row=1, right_col=0.98
                                        )
                                    ),
                                    data=resources_pb2.Data(
                                        concepts=[
                                            resources_pb2.Concept(id=water_concept_id, value=1)
                                        ]
                                    ),
                                ),
                            ]
                        ),
                    ),
                ]
            ),
            metadata=metadata(),
        )
        raise_on_failure(post_water_annotations_response)

        post_palm_annotations_searches_response = stub.PostAnnotationsSearches(
            service_pb2.PostAnnotationsSearchesRequest(
                searches=[
                    Search(
                        query=resources_pb2.Query(
                            filters=[
                                resources_pb2.Filter(
                                    annotation=resources_pb2.Annotation(
                                        data=resources_pb2.Data(
                                            concepts=[
                                                resources_pb2.Concept(id=palm_concept_id, value=1)
                                            ]
                                        )
                                    )
                                )
                            ]
                        )
                    )
                ],
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(post_palm_annotations_searches_response)
        assert input_.id in [hit.input.id for hit in post_palm_annotations_searches_response.hits]

        post_water_annotations_searches_response = stub.PostAnnotationsSearches(
            service_pb2.PostAnnotationsSearchesRequest(
                searches=[
                    Search(
                        query=resources_pb2.Query(
                            filters=[
                                resources_pb2.Filter(
                                    annotation=resources_pb2.Annotation(
                                        data=resources_pb2.Data(
                                            concepts=[
                                                resources_pb2.Concept(id=water_concept_id, value=1)
                                            ]
                                        )
                                    )
                                )
                            ]
                        )
                    )
                ],
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(post_water_annotations_searches_response)
        assert input_.id in [hit.input.id for hit in post_water_annotations_searches_response.hits]

        post_palm_and_water_annotations_searches_response = stub.PostAnnotationsSearches(
            service_pb2.PostAnnotationsSearchesRequest(
                searches=[
                    Search(
                        query=resources_pb2.Query(
                            filters=[
                                resources_pb2.Filter(
                                    annotation=resources_pb2.Annotation(
                                        data=resources_pb2.Data(
                                            concepts=[
                                                resources_pb2.Concept(id=palm_concept_id, value=1)
                                            ]
                                        )
                                    ),
                                ),
                                resources_pb2.Filter(
                                    annotation=resources_pb2.Annotation(
                                        data=resources_pb2.Data(
                                            concepts=[
                                                resources_pb2.Concept(id=water_concept_id, value=1)
                                            ]
                                        )
                                    )
                                ),
                            ]
                        )
                    )
                ],
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(post_palm_and_water_annotations_searches_response)
        # No single annotation can have two concepts, so this will return false.
        assert len(post_palm_and_water_annotations_searches_response.hits) == 0


class SetupImage:
    def __init__(self, stub: service_pb2_grpc.V2Stub) -> None:
        self._stub = stub

    def __enter__(self) -> resources_pb2.Input:
        post_response = self._stub.PostInputs(
            service_pb2.PostInputsRequest(
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(
                                url=TRAVEL_IMAGE_URL, allow_duplicate_url=True
                            ),
                        ),
                    )
                ]
            ),
            metadata=metadata(),
        )
        raise_on_failure(post_response)
        self._input = post_response.inputs[0]

        wait_for_inputs_upload(self._stub, metadata(), [self._input.id])

        return self._input

    def __exit__(self, type_, value, traceback) -> None:
        delete_response = self._stub.DeleteInput(
            service_pb2.DeleteInputRequest(input_id=self._input.id), metadata=metadata()
        )
        raise_on_failure(delete_response)
