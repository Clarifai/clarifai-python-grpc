import urllib.request
import uuid

from google.protobuf import struct_pb2

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
