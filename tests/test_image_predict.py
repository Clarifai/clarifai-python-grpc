from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from tests.common import (
    DOG_IMAGE_URL,
    GENERAL_MODEL_ID,
    NON_EXISTING_IMAGE_URL,
    RED_TRUCK_IMAGE_FILE_PATH,
    both_channels,
    metadata,
    raise_on_failure,
)


@both_channels
def test_predict_image_url(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    request = service_pb2.PostModelOutputsRequest(
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL))
            )
        ],
    )
    response = stub.PostModelOutputs(request, metadata=metadata())
    raise_on_failure(response)

    assert len(response.outputs[0].data.concepts) > 0


@both_channels
def test_predict_image_url_with_max_concepts(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    request = service_pb2.PostModelOutputsRequest(
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(
                        url=DOG_IMAGE_URL,
                    ),
                ),
            )
        ],
        model=resources_pb2.Model(
            output_info=resources_pb2.OutputInfo(
                output_config=resources_pb2.OutputConfig(max_concepts=3)
            )
        ),
    )
    response = stub.PostModelOutputs(request, metadata=metadata())
    raise_on_failure(response)

    assert len(response.outputs[0].data.concepts) == 3


@both_channels
def test_predict_image_url_with_min_value(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    request = service_pb2.PostModelOutputsRequest(
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(
                        url=DOG_IMAGE_URL,
                    ),
                ),
            )
        ],
        model=resources_pb2.Model(
            output_info=resources_pb2.OutputInfo(
                output_config=resources_pb2.OutputConfig(min_value=0.98)
            )
        ),
    )
    response = stub.PostModelOutputs(request, metadata=metadata())
    raise_on_failure(response)

    assert len(response.outputs[0].data.concepts) > 0
    for c in response.outputs[0].data.concepts:
        assert c.value >= 0.98


@both_channels
def test_predict_image_url_with_selected_concepts(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    request = service_pb2.PostModelOutputsRequest(
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(
                        url=DOG_IMAGE_URL,
                    ),
                ),
            )
        ],
        model=resources_pb2.Model(
            output_info=resources_pb2.OutputInfo(
                output_config=resources_pb2.OutputConfig(
                    select_concepts=[
                        resources_pb2.Concept(name="dog"),
                        resources_pb2.Concept(name="cat"),
                    ]
                )
            )
        ),
    )
    response = stub.PostModelOutputs(request, metadata=metadata())
    raise_on_failure(response)

    concepts = response.outputs[0].data.concepts
    assert len(concepts) == 2
    dog_concept = [c for c in concepts if c.name == "dog"][0]
    cat_concept = [c for c in concepts if c.name == "cat"][0]
    assert dog_concept.value > cat_concept.value


@both_channels
def test_predict_image_bytes(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with open(RED_TRUCK_IMAGE_FILE_PATH, "rb") as f:
        file_bytes = f.read()

    request = service_pb2.PostModelOutputsRequest(
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(image=resources_pb2.Image(base64=file_bytes))
            )
        ],
    )
    response = stub.PostModelOutputs(request, metadata=metadata())

    raise_on_failure(response)

    assert len(response.outputs[0].data.concepts) > 0


@both_channels
def test_failed_predict(channel):
    stub = service_pb2_grpc.V2Stub(channel)
    request = service_pb2.PostModelOutputsRequest(
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(image=resources_pb2.Image(url=NON_EXISTING_IMAGE_URL))
            )
        ],
    )
    response = stub.PostModelOutputs(request, metadata=metadata())

    assert response.status.code == status_code_pb2.FAILURE
    assert response.status.description == "Failure"

    assert response.outputs[0].status.code == status_code_pb2.INPUT_DOWNLOAD_FAILED


@both_channels
def test_mixed_success_predict(channel):
    stub = service_pb2_grpc.V2Stub(channel)
    request = service_pb2.PostModelOutputsRequest(
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL))
            ),
            resources_pb2.Input(
                data=resources_pb2.Data(image=resources_pb2.Image(url=NON_EXISTING_IMAGE_URL))
            ),
        ],
    )
    response = stub.PostModelOutputs(request, metadata=metadata())

    assert response.status.code == status_code_pb2.MIXED_STATUS

    assert response.outputs[0].status.code == status_code_pb2.SUCCESS
    assert response.outputs[1].status.code == status_code_pb2.INPUT_DOWNLOAD_FAILED
