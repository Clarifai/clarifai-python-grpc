import pytest
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import (
    BEER_VIDEO_URL,
    DOG_IMAGE_URL,
    GENERAL_MODEL_ID,
    both_channels,
    metadata,
    post_model_outputs_and_maybe_allow_retries,
    raise_on_failure,
)
from tests.public_models.public_test_helper import (
    AUDIO_MODEL_TITLE_IDS_TUPLE,
    DETECTION_MODEL_TITLE_AND_IDS,
    ENGLISH_AUDIO_URL,
    MODEL_TITLE_AND_ID_PAIRS,
    MULTIMODAL_MODEL_TITLE_AND_IDS,
    TEXT_FB_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE,
    TEXT_HELSINKI_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE,
    TEXT_MODEL_TITLE_IDS_TUPLE,
    TRANSLATION_TEST_DATA,
)


@both_channels
def test_audio_predict_on_public_models(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    for title, model_id, app_id, user_id in AUDIO_MODEL_TITLE_IDS_TUPLE:
        request = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
            model_id=model_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(audio=resources_pb2.Audio(url=ENGLISH_AUDIO_URL))
                )
            ],
        )
        response = post_model_outputs_and_maybe_allow_retries(
            stub, request, metadata=metadata(pat=True)
        )
        raise_on_failure(
            response,
            custom_message=f"Audio predict failed for the {title} model (ID: {model_id}).",
        )


@both_channels
def test_text_predict_on_public_models(channel):
    """Test non translation text/nlp models.
    All these models can take the same test text input.
    """
    stub = service_pb2_grpc.V2Stub(channel)

    for title, model_id, app_id, user_id in TEXT_MODEL_TITLE_IDS_TUPLE:
        request = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
            model_id=model_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(raw=TRANSLATION_TEST_DATA["EN"])
                    )
                )
            ],
        )
        response = post_model_outputs_and_maybe_allow_retries(
            stub, request, metadata=metadata(pat=True)
        )
        raise_on_failure(
            response,
            custom_message=f"Text predict failed for the {title} model (ID: {model_id}).",
        )


@pytest.mark.skip(reason="This test is ready, but will be added in time")
@both_channels
def test_text_fb_translation_predict_on_public_models(channel):
    """Test language translation models.
    Each language-english translation has its own text input while
    all en-language translations use the same english text.
    """
    stub = service_pb2_grpc.V2Stub(channel)
    for title, model_id, text, app_id, user_id in TEXT_FB_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE:
        request = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
            model_id=model_id,
            inputs=[
                resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=text)))
            ],
        )
        response = post_model_outputs_and_maybe_allow_retries(
            stub, request, metadata=metadata(pat=True)
        )
        raise_on_failure(
            response,
            custom_message=f"Text predict failed for the {title} model (ID: {model_id}).",
        )


@both_channels
def test_text_helsinki_translation_predict_on_public_models(channel):
    """Test language translation models.
    Each language-english translation has its own text input while
    all en-language translations use the same english text.
    """
    stub = service_pb2_grpc.V2Stub(channel)
    for (
        title,
        model_id,
        text,
        app_id,
        user_id,
    ) in TEXT_HELSINKI_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE:
        request = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
            model_id=model_id,
            inputs=[
                resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=text)))
            ],
        )
        response = post_model_outputs_and_maybe_allow_retries(
            stub, request, metadata=metadata(pat=True)
        )
        raise_on_failure(
            response,
            custom_message=f"Text predict failed for the {title} model (ID: {model_id}).",
        )


@both_channels
def test_image_predict_on_public_models(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    for title, model_id in MODEL_TITLE_AND_ID_PAIRS:
        request = service_pb2.PostModelOutputsRequest(
            model_id=model_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL))
                )
            ],
        )
        retryable_output_codes = [
            status_code_pb2.INTERNAL_UNCATEGORIZED
        ]  # if any of the output statuses matches this, retry prediction
        response = post_model_outputs_and_maybe_allow_retries(
            stub, request, metadata=metadata(), retryable_codes=retryable_output_codes
        )
        raise_on_failure(
            response,
            custom_message=f"Image predict failed for the {title} model (ID: {model_id}).",
        )


@both_channels
def test_image_detection_predict_on_public_models(channel):
    """Test object detection models using clarifai platform user
    and app id access credentials.
    """
    stub = service_pb2_grpc.V2Stub(channel)

    for title, model_id, app_id, user_id in DETECTION_MODEL_TITLE_AND_IDS:
        request = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
            model_id=model_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL))
                )
            ],
        )
        response = post_model_outputs_and_maybe_allow_retries(
            stub, request, metadata=metadata(pat=True)
        )
        raise_on_failure(
            response,
            custom_message=f"Image predict failed for the {title} model (ID: {model_id}).",
        )


@both_channels
def test_video_predict_on_public_models(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    title = "general"
    model_id = GENERAL_MODEL_ID

    request = service_pb2.PostModelOutputsRequest(
        model_id=model_id,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(video=resources_pb2.Video(url=BEER_VIDEO_URL))
            )
        ],
    )
    response = post_model_outputs_and_maybe_allow_retries(stub, request, metadata=metadata())
    raise_on_failure(
        response,
        custom_message=f"Video predict failed for the {title} model (ID: {model_id}).",
    )


@both_channels
def test_multimodal_predict_on_public_models(channel):
    """Test multimodal models.
    Currently supporting only text and image inputs.
    """
    stub = service_pb2_grpc.V2Stub(channel)

    for title, model_id in MULTIMODAL_MODEL_TITLE_AND_IDS:
        request = service_pb2.PostModelOutputsRequest(
            model_id=model_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL))
                ),
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(raw=TRANSLATION_TEST_DATA["EN"])
                    )
                ),
            ],
        )
        response = post_model_outputs_and_maybe_allow_retries(stub, request, metadata=metadata())
        raise_on_failure(
            response,
            custom_message=f"Image predict failed for the {title} model (ID: {model_id}).",
        )
