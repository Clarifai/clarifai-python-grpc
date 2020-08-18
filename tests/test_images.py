import uuid

from google.protobuf.struct_pb2 import Struct

from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import (
    RED_TRUCK_IMAGE_FILE_PATH,
    TRUCK_IMAGE_URL,
    both_channels,
    metadata,
    raise_on_failure,
    wait_for_inputs_upload,
)


@both_channels
def test_post_list_patch_get_delete_image(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    post_response = stub.PostInputs(
        service_pb2.PostInputsRequest(
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(url=TRUCK_IMAGE_URL, allow_duplicate_url=True),
                        concepts=[resources_pb2.Concept(id="some-concept")],
                    )
                )
            ]
        ),
        metadata=metadata(),
    )
    raise_on_failure(post_response)
    input_id = post_response.inputs[0].id

    try:
        wait_for_inputs_upload(stub, metadata(), [input_id])

        list_response = stub.ListInputs(
            service_pb2.ListInputsRequest(per_page=1), metadata=metadata()
        )
        raise_on_failure(list_response)
        assert len(list_response.inputs) == 1

        # Most likely we don"t have that many inputs, so this should return 0.
        list_response2 = stub.ListInputs(
            service_pb2.ListInputsRequest(per_page=500, page=1000), metadata=metadata()
        )
        raise_on_failure(list_response2)
        assert len(list_response2.inputs) == 0

        patch_response = stub.PatchInputs(
            service_pb2.PatchInputsRequest(
                action="overwrite",
                inputs=[
                    resources_pb2.Input(
                        id=input_id,
                        data=resources_pb2.Data(
                            concepts=[resources_pb2.Concept(id="some-new-concept")]
                        ),
                    )
                ],
            ),
            metadata=metadata(),
        )
        raise_on_failure(patch_response)

        get_response = stub.GetInput(
            service_pb2.GetInputRequest(input_id=input_id), metadata=metadata()
        )
        raise_on_failure(get_response)
        assert get_response.input.data.concepts[0].name == "some-new-concept"
    finally:
        delete_request = service_pb2.DeleteInputRequest(input_id=input_id)
        delete_response = stub.DeleteInput(delete_request, metadata=metadata())
        raise_on_failure(delete_response)


@both_channels
def test_post_delete_batch_images(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    post_response = stub.PostInputs(
        service_pb2.PostInputsRequest(
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(url=TRUCK_IMAGE_URL, allow_duplicate_url=True)
                    )
                ),
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(url=TRUCK_IMAGE_URL, allow_duplicate_url=True)
                    )
                ),
            ]
        ),
        metadata=metadata(),
    )
    raise_on_failure(post_response)
    input_id1 = post_response.inputs[0].id
    input_id2 = post_response.inputs[1].id

    wait_for_inputs_upload(stub, metadata(), [input_id1, input_id2])

    delete_response = stub.DeleteInputs(
        service_pb2.DeleteInputsRequest(ids=[input_id1, input_id2]), metadata=metadata()
    )
    raise_on_failure(delete_response)


@both_channels
def test_post_patch_get_image_with_id_concepts_geo_and_metadata(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    input_id = uuid.uuid4().hex

    input_metadata = Struct()
    input_metadata.update(
        {"key1": 123, "key2": {"inner-key1": "inner-val1", "inner-key2": "inner-val2",}}
    )

    post_response = stub.PostInputs(
        service_pb2.PostInputsRequest(
            inputs=[
                resources_pb2.Input(
                    id=input_id,
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(url=TRUCK_IMAGE_URL, allow_duplicate_url=True),
                        concepts=[
                            resources_pb2.Concept(id="some-positive-concept", value=1),
                            resources_pb2.Concept(id="some-negative-concept", value=0),
                        ],
                        geo=resources_pb2.Geo(
                            geo_point=resources_pb2.GeoPoint(longitude=55.0, latitude=66),
                        ),
                        metadata=input_metadata,
                    ),
                )
            ]
        ),
        metadata=metadata(),
    )
    raise_on_failure(post_response)

    wait_for_inputs_upload(stub, metadata(), [input_id])

    get_response = stub.GetInput(
        service_pb2.GetInputRequest(input_id=input_id), metadata=metadata()
    )
    raise_on_failure(get_response)

    inp = get_response.input
    assert inp.id == input_id
    assert inp.data.concepts[0].id == "some-positive-concept"
    assert inp.data.concepts[0].value == 1.0
    assert inp.data.concepts[1].id == "some-negative-concept"
    assert inp.data.concepts[1].value == 0.0
    assert inp.data.metadata["key1"] == 123
    assert inp.data.metadata["key2"]["inner-key1"] == "inner-val1"
    assert inp.data.metadata["key2"]["inner-key2"] == "inner-val2"
    assert inp.data.geo.geo_point.longitude == 55.0
    assert inp.data.geo.geo_point.latitude == 66.0

    new_metadata = Struct()
    new_metadata.update({"new-key": "new-value"})
    patch_response = stub.PatchInputs(
        service_pb2.PatchInputsRequest(
            action="merge",
            inputs=[
                resources_pb2.Input(
                    id=input_id,
                    data=resources_pb2.Data(
                        concepts=[resources_pb2.Concept(id="another-positive-concept", value=1)],
                        geo=resources_pb2.Geo(
                            geo_point=resources_pb2.GeoPoint(longitude=77.0, latitude=88.0)
                        ),
                        metadata=new_metadata,
                    ),
                )
            ],
        ),
        metadata=metadata(),
    )
    raise_on_failure(patch_response)
    inp = patch_response.inputs[0]
    assert inp.data.concepts[2].id == "another-positive-concept"
    assert inp.data.geo.geo_point.longitude == 77.0
    assert inp.data.geo.geo_point.latitude == 88.0
    assert inp.data.metadata["new-key"] == "new-value"
    assert (
        inp.data.metadata["key1"] == 123
    )  # Since we use the merge action, the old values should remain

    delete_response = stub.DeleteInputs(
        service_pb2.DeleteInputsRequest(ids=[input_id]), metadata=metadata()
    )
    raise_on_failure(delete_response)


@both_channels
def test_image_with_bytes(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with open(RED_TRUCK_IMAGE_FILE_PATH, "rb") as f:
        file_bytes = f.read()

    post_response = stub.PostInputs(
        service_pb2.PostInputsRequest(
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(image=resources_pb2.Image(base64=file_bytes))
                )
            ]
        ),
        metadata=metadata(),
    )
    raise_on_failure(post_response)
    input_id = post_response.inputs[0].id

    wait_for_inputs_upload(stub, metadata(), [input_id])

    delete_response = stub.DeleteInputs(
        service_pb2.DeleteInputsRequest(ids=[input_id]), metadata=metadata()
    )
    raise_on_failure(delete_response)
