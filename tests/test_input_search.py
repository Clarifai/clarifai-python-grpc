import time
import urllib.request
import uuid

from google.protobuf import struct_pb2, timestamp_pb2

from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import (
    DOG_IMAGE_URL,
    both_channels,
    metadata,
    raise_on_failure,
    wait_for_inputs_upload,
)


@both_channels
def test_search_by_annotated_concept_id(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        my_concept_id = input_.data.concepts[0].id
        response = stub.PostSearches(
            service_pb2.PostSearchesRequest(
                query=resources_pb2.Query(
                    ands=[
                        resources_pb2.And(
                            input=resources_pb2.Input(
                                data=resources_pb2.Data(
                                    concepts=[resources_pb2.Concept(id=my_concept_id, value=1)]
                                )
                            )
                        )
                    ]
                )
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) == 1
        assert response.hits[0].input.id == input_.id


@both_channels
def test_search_by_annotated_concept_name(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        my_concept_name = input_.data.concepts[0].name
        response = stub.PostSearches(
            service_pb2.PostSearchesRequest(
                query=resources_pb2.Query(
                    ands=[
                        resources_pb2.And(
                            input=resources_pb2.Input(
                                data=resources_pb2.Data(
                                    concepts=[resources_pb2.Concept(name=my_concept_name, value=1)]
                                )
                            )
                        )
                    ]
                )
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) == 1
        assert response.hits[0].input.id == input_.id


@both_channels
def test_search_by_predicted_concept_id(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        response = stub.PostSearches(
            service_pb2.PostSearchesRequest(
                query=resources_pb2.Query(
                    ands=[
                        resources_pb2.And(
                            output=resources_pb2.Output(
                                data=resources_pb2.Data(
                                    # The ID of the "dog" concept in clarifai/main
                                    concepts=[resources_pb2.Concept(id="ai_8S2Vq3cR", value=1)]
                                )
                            )
                        )
                    ]
                ),
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) > 0
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_predicted_concept_name(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        response = stub.PostSearches(
            service_pb2.PostSearchesRequest(
                query=resources_pb2.Query(
                    ands=[
                        resources_pb2.And(
                            output=resources_pb2.Output(
                                data=resources_pb2.Data(
                                    concepts=[resources_pb2.Concept(name="dog", value=1)]
                                )
                            )
                        )
                    ]
                ),
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) > 0
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_predicted_concept_name_in_chinese(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        response = stub.PostSearches(
            service_pb2.PostSearchesRequest(
                query=resources_pb2.Query(
                    ands=[
                        resources_pb2.And(
                            output=resources_pb2.Output(
                                data=resources_pb2.Data(
                                    concepts=[resources_pb2.Concept(name="狗", value=1)]
                                )
                            )
                        )
                    ],
                    language="zh",
                ),
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) > 0
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_predicted_concept_name_in_japanese(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        response = stub.PostSearches(
            service_pb2.PostSearchesRequest(
                query=resources_pb2.Query(
                    ands=[
                        resources_pb2.And(
                            output=resources_pb2.Output(
                                data=resources_pb2.Data(
                                    concepts=[resources_pb2.Concept(name="犬", value=1)]
                                )
                            )
                        )
                    ],
                    language="ja",
                ),
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) > 0
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_image_url(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        response = stub.PostSearches(
            service_pb2.PostSearchesRequest(
                query=resources_pb2.Query(
                    ands=[
                        resources_pb2.And(
                            output=resources_pb2.Output(
                                input=resources_pb2.Input(
                                    data=resources_pb2.Data(
                                        image=resources_pb2.Image(url=DOG_IMAGE_URL)
                                    )
                                )
                            )
                        )
                    ]
                ),
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) > 0
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_image_bytes(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        http_response = urllib.request.urlopen(DOG_IMAGE_URL)
        url_bytes = http_response.read()
        response = stub.PostSearches(
            service_pb2.PostSearchesRequest(
                query=resources_pb2.Query(
                    ands=[
                        resources_pb2.And(
                            output=resources_pb2.Output(
                                data=resources_pb2.Data(
                                    image=resources_pb2.Image(base64=url_bytes)
                                )
                            )
                        )
                    ]
                ),
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) > 0
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_metadata(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        search_metadata = struct_pb2.Struct()
        search_metadata.update({"another-key": {"inner-key": "inner-value"}})
        response = stub.PostSearches(
            service_pb2.PostSearchesRequest(
                query=resources_pb2.Query(
                    ands=[
                        resources_pb2.And(
                            input=resources_pb2.Input(
                                data=resources_pb2.Data(metadata=search_metadata)
                            )
                        )
                    ]
                ),
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) > 0
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_geo_point_and_limit(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        response = stub.PostSearches(
            service_pb2.PostSearchesRequest(
                query=resources_pb2.Query(
                    ands=[
                        resources_pb2.And(
                            input=resources_pb2.Input(
                                data=resources_pb2.Data(
                                    geo=resources_pb2.Geo(
                                        geo_point=resources_pb2.GeoPoint(
                                            longitude=43, latitude=56
                                        ),
                                        geo_limit=resources_pb2.GeoLimit(
                                            value=1000, type="withinKilometers"
                                        ),
                                    )
                                )
                            )
                        )
                    ]
                ),
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) > 0
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_geo_box(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        response = stub.PostSearches(
            service_pb2.PostSearchesRequest(
                query=resources_pb2.Query(
                    ands=[
                        resources_pb2.And(
                            input=resources_pb2.Input(
                                data=resources_pb2.Data(
                                    geo=resources_pb2.Geo(
                                        geo_box=[
                                            resources_pb2.GeoBoxedPoint(
                                                geo_point=resources_pb2.GeoPoint(
                                                    longitude=43, latitude=54
                                                )
                                            ),
                                            resources_pb2.GeoBoxedPoint(
                                                geo_point=resources_pb2.GeoPoint(
                                                    longitude=45, latitude=56
                                                )
                                            ),
                                        ]
                                    )
                                )
                            )
                        )
                    ]
                ),
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) > 0
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_image_url_and_geo_box(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        response = stub.PostSearches(
            service_pb2.PostSearchesRequest(
                query=resources_pb2.Query(
                    ands=[
                        resources_pb2.And(
                            output=resources_pb2.Output(
                                input=resources_pb2.Input(
                                    data=resources_pb2.Data(
                                        image=resources_pb2.Image(url=DOG_IMAGE_URL)
                                    )
                                )
                            )
                        ),
                        resources_pb2.And(
                            input=resources_pb2.Input(
                                data=resources_pb2.Data(
                                    geo=resources_pb2.Geo(
                                        geo_box=[
                                            resources_pb2.GeoBoxedPoint(
                                                geo_point=resources_pb2.GeoPoint(
                                                    longitude=43, latitude=54
                                                )
                                            ),
                                            resources_pb2.GeoBoxedPoint(
                                                geo_point=resources_pb2.GeoPoint(
                                                    longitude=45, latitude=56
                                                )
                                            ),
                                        ]
                                    )
                                )
                            )
                        ),
                    ]
                ),
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) > 0
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_geo_box_and_annotated_name_and_predicted_name(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        my_concept_name = input_.data.concepts[0].name
        response = stub.PostSearches(
            service_pb2.PostSearchesRequest(
                query=resources_pb2.Query(
                    ands=[
                        resources_pb2.And(
                            input=resources_pb2.Input(
                                data=resources_pb2.Data(
                                    geo=resources_pb2.Geo(
                                        geo_box=[
                                            resources_pb2.GeoBoxedPoint(
                                                geo_point=resources_pb2.GeoPoint(
                                                    longitude=43, latitude=54
                                                )
                                            ),
                                            resources_pb2.GeoBoxedPoint(
                                                geo_point=resources_pb2.GeoPoint(
                                                    longitude=45, latitude=56
                                                )
                                            ),
                                        ]
                                    )
                                )
                            )
                        ),
                        resources_pb2.And(
                            input=resources_pb2.Input(
                                data=resources_pb2.Data(
                                    concepts=[resources_pb2.Concept(name=my_concept_name, value=1)]
                                )
                            )
                        ),
                        resources_pb2.And(
                            output=resources_pb2.Output(
                                data=resources_pb2.Data(
                                    concepts=[resources_pb2.Concept(name="dog", value=1)]
                                )
                            )
                        ),
                    ]
                ),
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) > 0
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_save_and_execute_search_by_id(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    search_id = "my-search-id-" + uuid.uuid4().hex

    with SetupImage(stub) as input_:
        my_concept_id = input_.data.concepts[0].id
        # This saves the search under an ID, but does not execute it / return any results.
        save_search_response = stub.PostSearches(
            service_pb2.PostSearchesRequest(
                searches=[
                    resources_pb2.Search(
                        id=search_id,
                        save=True,
                        query=resources_pb2.Query(
                            ands=[
                                resources_pb2.And(
                                    input=resources_pb2.Input(
                                        data=resources_pb2.Data(
                                            concepts=[
                                                resources_pb2.Concept(id=my_concept_id, value=1)
                                            ]
                                        )
                                    )
                                )
                            ]
                        ),
                    )
                ]
            ),
            metadata=metadata(),
        )
        raise_on_failure(save_search_response)

        # Executing the search returns results.
        post_search_by_id_response = stub.PostSearchesByID(
            service_pb2.PostSearchesByIDRequest(id=search_id),
            metadata=metadata(),
        )
        raise_on_failure(post_search_by_id_response)
        assert len(post_search_by_id_response.hits) == 1
        assert post_search_by_id_response.hits[0].input.id == input_.id


@both_channels
def test_save_and_execute_annotations_search_by_id(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    my_search_id = "my-search-id-" + uuid.uuid4().hex
    my_concept_id = "my-annotation-concept-" + uuid.uuid4().hex

    with SetupImage(stub) as input1, SetupImage(stub) as input2:

        list_annotations_response = stub.ListAnnotations(
            service_pb2.ListAnnotationsRequest(input_ids=[input1.id, input2.id]),
            metadata=metadata(),
        )
        raise_on_failure(list_annotations_response)

        input_id_to_annotation_id = {
            an.input_id: an.id for an in list_annotations_response.annotations
        }

        patch_annotations_response = stub.PatchAnnotations(
            service_pb2.PatchAnnotationsRequest(
                action="merge",
                annotations=[
                    resources_pb2.Annotation(
                        id=input_id_to_annotation_id[input1.id],
                        input_id=input1.id,
                        data=resources_pb2.Data(
                            concepts=[resources_pb2.Concept(id=my_concept_id, value=1)]
                        ),
                    ),
                    resources_pb2.Annotation(
                        id=input_id_to_annotation_id[input2.id],
                        input_id=input2.id,
                        data=resources_pb2.Data(
                            concepts=[resources_pb2.Concept(id=my_concept_id, value=1)]
                        ),
                    ),
                ],
            ),
            metadata=metadata(),
        )
        raise_on_failure(patch_annotations_response)

        as_of = timestamp_pb2.Timestamp()
        as_of.FromSeconds(int(time.time() + 5))

        save_search_response = stub.PostSearches(
            service_pb2.PostSearchesRequest(
                searches=[
                    resources_pb2.Search(
                        id=my_search_id,
                        save=True,
                        as_of=as_of,
                        query=resources_pb2.Query(
                            ands=[
                                resources_pb2.And(
                                    input=resources_pb2.Input(
                                        data=resources_pb2.Data(
                                            concepts=[
                                                resources_pb2.Concept(id=my_concept_id, value=1)
                                            ]
                                        )
                                    )
                                )
                            ]
                        ),
                    )
                ]
            ),
            metadata=metadata(),
        )
        raise_on_failure(save_search_response)

        # Executing the search returns results.
        post_search_by_id_response = stub.PostSearchesByID(
            service_pb2.PostSearchesByIDRequest(id=my_search_id),
            metadata=metadata(),
        )
        raise_on_failure(post_search_by_id_response)
        hits = post_search_by_id_response.hits
        assert len(hits) == 2
        assert input1.id in [hit.input.id for hit in hits]
        assert input2.id in [hit.input.id for hit in hits]
        assert all(hit.score == 1 for hit in hits)


class SetupImage:
    def __init__(self, stub: service_pb2_grpc.V2Stub) -> None:
        self._stub = stub

    def __enter__(self) -> resources_pb2.Input:
        my_concept_id = "my-concept-id-" + uuid.uuid4().hex
        my_concept_name = "my concept name " + uuid.uuid4().hex

        image_metadata = struct_pb2.Struct()
        image_metadata.update(
            {"some-key": "some-value", "another-key": {"inner-key": "inner-value"}}
        )

        post_response = self._stub.PostInputs(
            service_pb2.PostInputsRequest(
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(url=DOG_IMAGE_URL, allow_duplicate_url=True),
                            concepts=[
                                resources_pb2.Concept(
                                    id=my_concept_id, name=my_concept_name, value=1
                                )
                            ],
                            metadata=image_metadata,
                            geo=resources_pb2.Geo(
                                geo_point=resources_pb2.GeoPoint(longitude=44, latitude=55)
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
