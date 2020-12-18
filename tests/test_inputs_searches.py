import urllib.request
import uuid

from google.protobuf import struct_pb2

from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.resources_pb2 import (
    Search,
    Query,
    Rank,
    Annotation,
    Data,
    Concept,
    Filter,
    Image,
)
from clarifai_grpc.grpc.api.service_pb2 import PostInputsSearchesRequest
from tests.common import (
    both_channels,
    metadata,
    raise_on_failure,
    DOG_IMAGE_URL,
    wait_for_inputs_upload,
)


@both_channels
def test_search_by_custom_concept_id(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        concept_id = input_.data.concepts[0].id
        response = stub.PostInputsSearches(
            PostInputsSearchesRequest(
                searches=[
                    Search(
                        query=Query(
                            filters=[
                                Filter(
                                    annotation=Annotation(
                                        data=Data(concepts=[Concept(id=concept_id, value=1)])
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
        raise_on_failure(response)
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_custom_concept_name(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        concept_name = input_.data.concepts[0].name
        response = stub.PostInputsSearches(
            PostInputsSearchesRequest(
                searches=[
                    Search(
                        query=Query(
                            filters=[
                                Filter(
                                    annotation=Annotation(
                                        data=Data(concepts=[Concept(name=concept_name, value=1)])
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
        raise_on_failure(response)
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_predicted_concept_id(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        response = stub.PostInputsSearches(
            PostInputsSearchesRequest(
                searches=[
                    Search(
                        query=Query(
                            ranks=[
                                Rank(
                                    annotation=Annotation(
                                        # The ID of the "dog" concept in clarifai/main
                                        data=Data(concepts=[Concept(id="ai_8S2Vq3cR", value=1)])
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
        raise_on_failure(response)
        assert len(response.hits) >= 1
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_predicted_concept_name(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        response = stub.PostInputsSearches(
            PostInputsSearchesRequest(
                searches=[
                    Search(
                        query=Query(
                            ranks=[
                                Rank(
                                    annotation=Annotation(
                                        data=Data(concepts=[Concept(name="dog", value=1)])
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
        raise_on_failure(response)
        assert len(response.hits) >= 1
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_predicted_concept_name_in_chinese(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        response = stub.PostInputsSearches(
            PostInputsSearchesRequest(
                searches=[
                    Search(
                        query=Query(
                            ranks=[
                                Rank(
                                    annotation=Annotation(
                                        data=Data(concepts=[Concept(name="狗", value=1)])
                                    )
                                )
                            ],
                            language="zh",
                        ),
                    )
                ],
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) >= 1
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_image_url(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        response = stub.PostInputsSearches(
            PostInputsSearchesRequest(
                searches=[
                    Search(
                        query=Query(
                            ranks=[
                                Rank(
                                    annotation=Annotation(
                                        data=Data(image=Image(url=DOG_IMAGE_URL))
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
        raise_on_failure(response)
        assert len(response.hits) >= 1
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_image_bytes(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    http_response = urllib.request.urlopen(DOG_IMAGE_URL)
    url_bytes = http_response.read()

    with SetupImage(stub) as input_:
        response = stub.PostInputsSearches(
            PostInputsSearchesRequest(
                searches=[
                    Search(
                        query=Query(
                            ranks=[
                                Rank(
                                    annotation=Annotation(data=Data(image=Image(base64=url_bytes)))
                                )
                            ]
                        )
                    )
                ],
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) >= 1
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_metadata(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    search_metadata = struct_pb2.Struct()
    search_metadata.update({"another-key": {"inner-key": "inner-value"}})

    with SetupImage(stub) as input_:
        response = stub.PostInputsSearches(
            PostInputsSearchesRequest(
                searches=[
                    Search(
                        query=Query(
                            ranks=[
                                Rank(annotation=Annotation(data=Data(metadata=search_metadata)))
                            ]
                        )
                    )
                ],
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) >= 1
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_geo_point_and_limit(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        response = stub.PostInputsSearches(
            PostInputsSearchesRequest(
                searches=[
                    Search(
                        query=Query(
                            ranks=[
                                Rank(
                                    annotation=Annotation(
                                        data=Data(
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
                        )
                    )
                ],
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) >= 1
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_geo_box(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        response = stub.PostInputsSearches(
            PostInputsSearchesRequest(
                searches=[
                    Search(
                        query=Query(
                            ranks=[
                                Rank(
                                    annotation=Annotation(
                                        data=Data(
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
                        )
                    )
                ],
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) >= 1
        assert input_.id in [hit.input.id for hit in response.hits]


@both_channels
def test_search_by_image_url_and_geo_box(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as input_:
        response = stub.PostInputsSearches(
            PostInputsSearchesRequest(
                searches=[
                    Search(
                        query=Query(
                            ranks=[
                                Rank(
                                    annotation=Annotation(
                                        data=Data(image=Image(url=DOG_IMAGE_URL))
                                    )
                                ),
                                Rank(
                                    annotation=Annotation(
                                        data=Data(
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
                        )
                    )
                ],
                pagination=service_pb2.Pagination(page=1, per_page=1000),
            ),
            metadata=metadata(),
        )
        raise_on_failure(response)
        assert len(response.hits) >= 1
        assert input_.id in [hit.input.id for hit in response.hits]


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
