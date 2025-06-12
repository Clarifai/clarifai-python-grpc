import asyncio
import os

import pytest
from openai import OpenAI

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from tests.common import (
    BEER_VIDEO_URL,
    DOG_IMAGE_URL,
    GENERAL_MODEL_ID,
    MAIN_APP_ID,
    MAIN_APP_USER_ID,
    async_post_model_outputs_and_maybe_allow_retries,
    async_raise_on_failure,
    asyncio_channel,
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

# New constant for the OpenAI compatible endpoint test
API_KEY = os.environ.get("CLARIFAI_PAT_KEY", os.environ.get("CLARIFAI_PAT"))


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


@asyncio_channel
async def test_audio_predict_on_public_models_async(channel):
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
        response = await async_post_model_outputs_and_maybe_allow_retries(
            stub, request, metadata=metadata(pat=True)
        )

        await async_raise_on_failure(
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


@asyncio_channel
async def test_text_predict_on_public_models_async(channel):
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
        response = await async_post_model_outputs_and_maybe_allow_retries(
            stub, request, metadata=metadata(pat=True)
        )
        await async_raise_on_failure(
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
            user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
            model_id=model_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL))
                )
            ],
        )
        response = post_model_outputs_and_maybe_allow_retries(stub, request, metadata(pat=True))
        raise_on_failure(
            response,
            custom_message=f"Image predict failed for the {title} model (ID: {model_id}).",
        )


@asyncio_channel
async def test_image_predict_on_public_models_async(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    for title, model_id in MODEL_TITLE_AND_ID_PAIRS:
        request = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
            model_id=model_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL))
                )
            ],
        )
        response = await async_post_model_outputs_and_maybe_allow_retries(
            stub, request, metadata(pat=True)
        )
        await async_raise_on_failure(
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


@asyncio_channel
async def test_image_detection_predict_on_public_models_async(channel):
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
        response = await async_post_model_outputs_and_maybe_allow_retries(
            stub, request, metadata=metadata(pat=True)
        )
        await async_raise_on_failure(
            response,
            custom_message=f"Image predict failed for the {title} model (ID: {model_id}).",
        )


@both_channels
def test_video_predict_on_public_models(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    title = "general"
    model_id = GENERAL_MODEL_ID

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
        model_id=model_id,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(video=resources_pb2.Video(url=BEER_VIDEO_URL))
            )
        ],
    )
    response = post_model_outputs_and_maybe_allow_retries(
        stub, request, metadata=metadata(pat=True)
    )
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
            user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
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
        response = post_model_outputs_and_maybe_allow_retries(
            stub, request, metadata=metadata(pat=True)
        )
        raise_on_failure(
            response,
            custom_message=f"Image predict failed for the {title} model (ID: {model_id}).",
        )


# Helper functions for the new OpenAI compatible endpoint test
def _call_openai_model(model_id):
    """Attempts to call a model using OpenAI's chat completions and image generation APIs."""
    client = OpenAI(api_key=API_KEY, base_url="https://api.clarifai.com/v2/ext/openai/v1")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who are you?"},
            ],
            max_tokens=50,
        )
        assert hasattr(response, "choices") and len(response.choices) > 0, "No choices in response"
        return response, None
    except Exception as e1:
        try:
            response = client.images.generate(
                model=model_id,
                prompt="A cat and a dog sitting together in a park",
            )
            assert hasattr(response, "data") and len(response.data) > 0, (
                "No image data in response"
            )
            return response, None
        except Exception as e2:
            return None, f"chat.completions error: {e1}; image.generate error: {e2}"


def _list_featured_models(per_page=50):
    """Lists featured models from the Clarifai platform."""
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    auth_metadata = (("authorization", f"Key {API_KEY}"),)
    request = service_pb2.ListModelsRequest(per_page=per_page, featured_only=True)
    response = stub.ListModels(request, metadata=auth_metadata)
    if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"ListModels failed: {response.status.description}")
    return response.models


# New integrated test
def test_openai_compatible_endpoint_on_featured_models():
    """Tests the OpenAI compatible endpoint with featured models."""
    featured_models = _list_featured_models()
    failed_models = []

    for model in featured_models:
        method_signatures = getattr(model.model_version, "method_signatures", [])
        if any(ms.name == "openai_transport" for ms in method_signatures):
            model_identifier = f"{model.user_id}/{model.app_id}/models/{model.id}"
            _, error = _call_openai_model(model_identifier)
            if error:
                failed_models.append({model_identifier: error})
        else:
            print(f"Skipping model {model.id} as it lacks 'openai_transport' method signature.")

    assert not failed_models, f"The following OpenAI compatible models failed: {failed_models}"


# New integrated async test
async def _call_openai_model_async(model_identifier, session):
    """Async helper to call a single model."""
    # Note: The OpenAI library doesn't natively support asyncio for this type of call.
    # To make this truly async, one would typically use an async-compatible HTTP client
    # like `httpx` or `aiohttp`. For simplicity and to match the provided script's
    # library, we run the synchronous `_call_openai_model` in a thread pool executor.
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _call_openai_model, model_identifier)


@pytest.mark.asyncio
async def test_openai_compatible_endpoint_on_featured_models_async():
    """Tests the OpenAI compatible endpoint concurrently with featured models."""
    featured_models = _list_featured_models()
    tasks = []
    model_identifiers = []

    for model in featured_models:
        method_signatures = getattr(model.model_version, "method_signatures", [])
        if any(ms.name == "openai_transport" for ms in method_signatures):
            model_identifier = f"{model.user_id}/{model.app_id}/models/{model.id}"
            model_identifiers.append(model_identifier)
            tasks.append(_call_openai_model_async(model_identifier, None))
        else:
            print(f"Skipping model {model.id} as it lacks 'openai_transport' method signature.")

    results = await asyncio.gather(*tasks)
    failed_models = []
    for i, (_, error) in enumerate(results):
        if error:
            failed_models.append({model_identifiers[i]: error})

    assert not failed_models, f"The following OpenAI compatible models failed: {failed_models}"
