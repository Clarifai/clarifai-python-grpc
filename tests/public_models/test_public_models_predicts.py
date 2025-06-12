import asyncio
import os
import time

import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

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
MAX_RETRY_ATTEMPTS = 3


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
    """
    Attempts to call a model using OpenAI's chat completions and image generation APIs,
    with an integrated retry mechanism and corrected parameters.
    """
    client = OpenAI(api_key=API_KEY, base_url="https://api.clarifai.com/v2/ext/openai/v1")
    last_err_chat = None
    last_err_image = None

    # --- Attempt 1: Chat Completions with Retry and Corrected Parameter ---
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            # CORRECTED: Replaced 'max_tokens' with 'extra_body' to send 'max_completion_tokens'
            # as required by the model's API error message.
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who are you?"},
                ],
                extra_body={'max_completion_tokens': 50}
            )
            if hasattr(response, 'choices') and response.choices:
                return response, None  # Success
            else:
                last_err_chat = ValueError(f"Chat completions returned no choices. Response: {response}")
                break
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            last_err_chat = e
            if attempt == MAX_RETRY_ATTEMPTS - 1:
                break
            print(f"Retrying chat predict for '{model_id}' after error: {e}. Attempt #{attempt + 1}")
            time.sleep(attempt + 1)
        except Exception as e:
            last_err_chat = e
            break

    # --- Attempt 2: Image Generation with Retry ---
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            response = client.images.generate(
                model=model_id,
                prompt="A cat and a dog sitting together in a park",
            )
            if hasattr(response, 'data') and response.data:
                return response, None  # Success
            else:
                last_err_image = ValueError(f"Image generation returned no data. Response: {response}")
                break
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            last_err_image = e
            if attempt == MAX_RETRY_ATTEMPTS - 1:
                break
            print(f"Retrying image predict for '{model_id}' after error: {e}. Attempt #{attempt + 1}")
            time.sleep(attempt + 1)
        except Exception as e:
            last_err_image = e
            break

    return None, f"chat.completions error: {last_err_chat}; image.generate error: {last_err_image}"


def _list_featured_models(per_page=50):
    """Lists featured models from the Clarifai platform."""
    # This function remains unchanged
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    auth_metadata = (("authorization", f"Key {API_KEY}"),)
    request = service_pb2.ListModelsRequest(per_page=per_page, featured_only=True)
    response = stub.ListModels(request, metadata=auth_metadata)
    if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"ListModels failed: {response.status.description}")
    return response.models


# The test functions below remain unchanged as the retry logic
# is now encapsulated within the _call_openai_model helper.


# New integrated test
def test_openai_compatible_endpoint_on_featured_models():
    """Tests the OpenAI compatible endpoint with featured models."""
    if not API_KEY:
        pytest.skip("Skipping test: CLARIFAI_PAT environment variable not set.")

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
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _call_openai_model, model_identifier)


@pytest.mark.asyncio
async def test_openai_compatible_endpoint_on_featured_models_async():
    """Tests the OpenAI compatible endpoint concurrently with featured models."""
    if not API_KEY:
        pytest.skip("Skipping test: CLARIFAI_PAT environment variable not set.")

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
