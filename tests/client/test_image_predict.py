from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from tests.common import (
    DOG_IMAGE_URL,
    GENERAL_MODEL_ID,
    MAIN_APP_ID,
    MAIN_APP_USER_ID,
    NON_EXISTING_IMAGE_URL,
    RED_TRUCK_IMAGE_FILE_PATH,
    both_channels,
    get_channel,
    metadata,
    post_model_outputs_and_maybe_allow_retries,
    raise_on_failure,
)


@both_channels()
def test_predict_image_url(channel_key):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL))
            )
        ],
    )
    response = post_model_outputs_and_maybe_allow_retries(
        stub, request, metadata=metadata(pat=True)
    )
    raise_on_failure(response)

    assert len(response.outputs[0].data.concepts) > 0


@both_channels()
def test_predict_image_url_with_max_concepts(channel_key):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
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
    response = post_model_outputs_and_maybe_allow_retries(
        stub, request, metadata=metadata(pat=True)
    )
    raise_on_failure(response)

    assert len(response.outputs[0].data.concepts) == 3


@both_channels()
def test_predict_image_url_with_min_value(channel_key):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
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
    response = post_model_outputs_and_maybe_allow_retries(
        stub, request, metadata=metadata(pat=True)
    )
    raise_on_failure(response)

    assert len(response.outputs[0].data.concepts) > 0
    for c in response.outputs[0].data.concepts:
        assert c.value >= 0.98


@both_channels()
def test_predict_image_url_with_selected_concepts(channel_key):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
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
    response = post_model_outputs_and_maybe_allow_retries(
        stub, request, metadata=metadata(pat=True)
    )
    raise_on_failure(response)

    concepts = response.outputs[0].data.concepts
    assert len(concepts) == 2
    dog_concept = [c for c in concepts if c.name == "dog"][0]
    cat_concept = [c for c in concepts if c.name == "cat"][0]
    assert dog_concept.value > cat_concept.value


@both_channels()
def test_predict_image_bytes(channel_key):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

    with open(RED_TRUCK_IMAGE_FILE_PATH, "rb") as f:
        file_bytes = f.read()

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(image=resources_pb2.Image(base64=file_bytes))
            )
        ],
    )
    response = post_model_outputs_and_maybe_allow_retries(
        stub, request, metadata=metadata(pat=True)
    )

    raise_on_failure(response)

    assert len(response.outputs[0].data.concepts) > 0


@both_channels()
def test_failed_predict(channel_key):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))
    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(image=resources_pb2.Image(url=NON_EXISTING_IMAGE_URL))
            )
        ],
    )
    response = post_model_outputs_and_maybe_allow_retries(
        stub, request, metadata=metadata(pat=True)
    )

    assert response.status.code == status_code_pb2.INPUT_DOWNLOAD_FAILED
    # Disabled as the example.com domain started returning 500 errors instead
    # assert (
    #     response.status.details == "404 Client Error: Not Found for url: "
    #     "http://example.com/non-existing.jpg"
    # )

    assert response.outputs[0].status.code == status_code_pb2.INPUT_DOWNLOAD_FAILED
    # Disabled as the example.com domain started returning 500 errors instead
    # assert (
    #     response.outputs[0].status.details == "404 Client Error: Not Found for url: "
    #     "http://example.com/non-existing.jpg"
    # )


@both_channels()
def test_mixed_success_predict(channel_key):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))
    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
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
    response = post_model_outputs_and_maybe_allow_retries(
        stub, request, metadata=metadata(pat=True)
    )

    assert response.status.code == status_code_pb2.MIXED_STATUS

    assert response.outputs[0].status.code == status_code_pb2.SUCCESS
    assert response.outputs[1].status.code == status_code_pb2.INPUT_DOWNLOAD_FAILED
