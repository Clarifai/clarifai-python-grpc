import urllib.request
import uuid
from typing import Tuple

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
def test_search(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with SetupImage(stub) as (input_id, my_concept_id, my_concept_name):
        #
        # Search by annotated concept ID
        #
        post_searches_response_1 = stub.PostSearches(
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
        raise_on_failure(post_searches_response_1)
        assert len(post_searches_response_1.hits) == 1
        assert post_searches_response_1.hits[0].input.id == input_id

        #
        # Search by annotated concept name
        #
        post_searches_response_2 = stub.PostSearches(
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
        raise_on_failure(post_searches_response_2)
        assert len(post_searches_response_2.hits) == 1
        assert post_searches_response_2.hits[0].input.id == input_id

        #
        # Search by predicted concept ID
        #
        post_searches_response_3 = stub.PostSearches(
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
        raise_on_failure(post_searches_response_3)
        assert len(post_searches_response_3.hits) > 0
        assert input_id in [hit.input.id for hit in post_searches_response_3.hits]

        #
        # Search by predicted concept name
        #
        post_searches_response_4 = stub.PostSearches(
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
        raise_on_failure(post_searches_response_4)
        assert len(post_searches_response_4.hits) > 0
        assert input_id in [hit.input.id for hit in post_searches_response_4.hits]

        #
        # Search by predicted concept name in Chinese
        #
        post_searches_response_5 = stub.PostSearches(
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
        raise_on_failure(post_searches_response_5)
        assert len(post_searches_response_5.hits) > 0
        assert input_id in [hit.input.id for hit in post_searches_response_5.hits]

        #
        # Search by predicted concept name in Japanese
        #
        post_searches_response_6 = stub.PostSearches(
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
        raise_on_failure(post_searches_response_6)
        assert len(post_searches_response_6.hits) > 0
        assert input_id in [hit.input.id for hit in post_searches_response_6.hits]

        #
        # Search by image URL
        #
        post_searches_response_7 = stub.PostSearches(
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
        raise_on_failure(post_searches_response_7)
        assert len(post_searches_response_7.hits) > 0
        assert input_id in [hit.input.id for hit in post_searches_response_7.hits]

        #
        # Search by image bytes
        #
        response = urllib.request.urlopen(DOG_IMAGE_URL)
        url_bytes = response.read()
        post_searches_response_8 = stub.PostSearches(
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
        raise_on_failure(post_searches_response_8)
        assert len(post_searches_response_8.hits) > 0
        assert input_id in [hit.input.id for hit in post_searches_response_8.hits]

        #
        # Search by metadata
        #
        search_metadata = struct_pb2.Struct()
        search_metadata.update({"another-key": {"inner-key": "inner-value"}})
        post_searches_response_9 = stub.PostSearches(
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
        raise_on_failure(post_searches_response_9)
        assert len(post_searches_response_9.hits) > 0
        assert input_id in [hit.input.id for hit in post_searches_response_9.hits]


class SetupImage:
    def __init__(self, stub: service_pb2_grpc.V2Stub) -> None:
        self._stub = stub

    def __enter__(self) -> Tuple[str, str, str]:
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
        self._input_id = post_response.inputs[0].id

        wait_for_inputs_upload(self._stub, metadata(), [self._input_id])

        return self._input_id, my_concept_id, my_concept_name

    def __exit__(self, type_, value, traceback) -> None:
        delete_response = self._stub.DeleteInput(
            service_pb2.DeleteInputRequest(input_id=self._input_id), metadata=metadata()
        )
        raise_on_failure(delete_response)
