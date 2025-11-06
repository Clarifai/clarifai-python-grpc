import asyncio
import json
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
from clarifai_grpc.channel.http_client import CLIENT_VERSION
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from tests.common import (
    BEER_VIDEO_URL,
    DOG_IMAGE_URL,
    GENERAL_MODEL_ID,
    MAIN_APP_ID,
    MAIN_APP_USER_ID,
    # _generate_model_outputs,
    aio_grpc_channel,
    async_post_model_outputs_and_maybe_allow_retries,
    async_raise_on_failure,
    both_channels,
    get_channel,
    #  grpc_channel,
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
    TEXT_HELSINKI_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE,
    #  TEXT_LLM_MODEL_TITLE_IDS_TUPLE,
    TEXT_MODEL_TITLE_IDS_TUPLE,
    TRANSLATION_TEST_DATA,
)

MAX_RETRY_ATTEMPTS = 3


@both_channels()
@pytest.mark.parametrize("title, model_id, app_id, user_id ", AUDIO_MODEL_TITLE_IDS_TUPLE)
def test_audio_predict_on_public_models(channel_key, title, model_id, app_id, user_id):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

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


@aio_grpc_channel()
@pytest.mark.parametrize("title, model_id, app_id, user_id ", AUDIO_MODEL_TITLE_IDS_TUPLE)
async def test_audio_predict_on_public_models_async(channel_key, title, model_id, app_id, user_id):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

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


@both_channels()
@pytest.mark.parametrize("title, model_id, app_id, user_id", TEXT_MODEL_TITLE_IDS_TUPLE)
def test_text_predict_on_public_models(channel_key, title, model_id, app_id, user_id):
    """Test non translation text/nlp models.
    All these models can take the same test text input.
    """
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
        model_id=model_id,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(text=resources_pb2.Text(raw=TRANSLATION_TEST_DATA["EN"]))
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


# @grpc_channel()
# @pytest.mark.parametrize("title, model_id, app_id, user_id", TEXT_LLM_MODEL_TITLE_IDS_TUPLE)
# def test_text_predict_on_public_llm_models(channel_key, title, model_id, app_id, user_id):
#     channel = get_channel(channel_key)
#     if channel._target != "api.clarifai.com":
#         pytest.skip(f"Model not available in {channel._target}")

#     stub = service_pb2_grpc.V2Stub(channel)

#     request = service_pb2.PostModelOutputsRequest(
#         user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
#         model_id=model_id,
#         inputs=[
#             resources_pb2.Input(
#                 data=resources_pb2.Data(
#                     parts=[
#                         resources_pb2.Part(
#                             id="prompt",
#                             data=resources_pb2.Data(
#                                 string_value=TRANSLATION_TEST_DATA["EN"],
#                             ),
#                         ),
#                         resources_pb2.Part(
#                             id="max_tokens",
#                             data=resources_pb2.Data(
#                                 int_value=10,
#                             ),
#                         ),
#                         resources_pb2.Part(
#                             id="temperature",
#                             data=resources_pb2.Data(
#                                 float_value=0.7,
#                             ),
#                         ),
#                         resources_pb2.Part(
#                             id="top_p",
#                             data=resources_pb2.Data(
#                                 float_value=0.95,
#                             ),
#                         ),
#                     ]
#                 )
#             )
#         ],
#     )
#     response_iterator = _generate_model_outputs(stub, request, metadata(pat=True))

#     responses_count = 0
#     for response in response_iterator:
#         responses_count += 1
#         raise_on_failure(
#             response,
#             custom_message=f"Text predict failed for the {title} model (ID: {model_id}).",
#         )

#     assert responses_count > 0


@aio_grpc_channel()
@pytest.mark.parametrize("title, model_id, app_id, user_id", TEXT_MODEL_TITLE_IDS_TUPLE)
async def test_text_predict_on_public_models_async(channel_key, title, model_id, app_id, user_id):
    """Test non translation text/nlp models.
    All these models can take the same test text input.
    """
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
        model_id=model_id,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(text=resources_pb2.Text(raw=TRANSLATION_TEST_DATA["EN"]))
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


# @both_channels()
# @pytest.mark.parametrize(
#     "title, model_id, text, app_id, user_id ", TEXT_FB_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE
# )
# def test_text_fb_translation_predict_on_public_models(
#     channel_key, title, model_id, text, app_id, user_id
# ):
#     """Test language translation models.
#     Each language-english translation has its own text input while
#     all en-language translations use the same english text.
#     """
#     pytest.skip("Skipping test: FB models are currently disabled")
#     stub = service_pb2_grpc.V2Stub(get_channel(channel_key))
#     request = service_pb2.PostModelOutputsRequest(
#         user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
#         model_id=model_id,
#         inputs=[resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=text)))],
#     )
#     response = post_model_outputs_and_maybe_allow_retries(
#         stub, request, metadata=metadata(pat=True)
#     )
#     raise_on_failure(
#         response,
#         custom_message=f"Text predict failed for the {title} model (ID: {model_id}).",
#     )


@both_channels()
@pytest.mark.parametrize(
    "title, model_id, text, app_id, user_id", TEXT_HELSINKI_TRANSLATION_MODEL_TITLE_ID_DATA_TUPLE
)
def test_text_helsinki_translation_predict_on_public_models(
    channel_key, title, model_id, text, app_id, user_id
):
    """Test language translation models.
    Each language-english translation has its own text input while
    all en-language translations use the same english text.
    """
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))
    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
        model_id=model_id,
        inputs=[resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=text)))],
    )
    response = post_model_outputs_and_maybe_allow_retries(
        stub, request, metadata=metadata(pat=True)
    )
    raise_on_failure(
        response,
        custom_message=f"Text predict failed for the {title} model (ID: {model_id}).",
    )


@both_channels()
@pytest.mark.parametrize("title, model_id", MODEL_TITLE_AND_ID_PAIRS)
def test_image_predict_on_public_models(channel_key, title, model_id):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

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


@aio_grpc_channel()
@pytest.mark.parametrize("title, model_id", MODEL_TITLE_AND_ID_PAIRS)
async def test_image_predict_on_public_models_async(channel_key, title, model_id):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

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


@both_channels()
@pytest.mark.parametrize("title, model_id, app_id, user_id", DETECTION_MODEL_TITLE_AND_IDS)
def test_image_detection_predict_on_public_models(channel_key, title, model_id, app_id, user_id):
    """Test object detection models using clarifai platform user
    and app id access credentials.
    """
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

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


@aio_grpc_channel()
@pytest.mark.parametrize("title, model_id, app_id, user_id", DETECTION_MODEL_TITLE_AND_IDS)
async def test_image_detection_predict_on_public_models_async(
    channel_key, title, model_id, app_id, user_id
):
    """Test object detection models using clarifai platform user
    and app id access credentials.
    """
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

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


@both_channels()
@pytest.mark.parametrize("title, model_id", [("general", GENERAL_MODEL_ID)])
def test_video_predict_on_public_models(channel_key, title, model_id):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

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


@both_channels()
@pytest.mark.parametrize("title, model_id", MULTIMODAL_MODEL_TITLE_AND_IDS)
def test_multimodal_predict_on_public_models(channel_key, title, model_id):
    """Test multimodal models.
    Currently supporting only text and image inputs.
    """
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
        model_id=model_id,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL))
            ),
            resources_pb2.Input(
                data=resources_pb2.Data(text=resources_pb2.Text(raw=TRANSLATION_TEST_DATA["EN"]))
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
    channel = ClarifaiChannel.get_grpc_channel()
    client = OpenAI(
        api_key=os.environ.get('CLARIFAI_PAT_KEY'),
        base_url=f"https://{channel._target}/v2/ext/openai/v1",
        default_headers={"X-Clarifai-Request-Id-Prefix": f"python-openai-{CLIENT_VERSION}"},
    )
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
                extra_body={'max_completion_tokens': 256},
            )
            if hasattr(response, 'choices') and response.choices:
                return response, None  # Success
            else:
                last_err_chat = ValueError(
                    f"Chat completions returned no choices. Response: {response}"
                )
                break
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            last_err_chat = e
            if attempt == MAX_RETRY_ATTEMPTS - 1:
                break
            print(
                f"Retrying chat predict for '{model_id}' after error: {e}. Attempt #{attempt + 1}"
            )
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
                last_err_image = ValueError(
                    f"Image generation returned no data. Response: {response}"
                )
                break
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            last_err_image = e
            if attempt == MAX_RETRY_ATTEMPTS - 1:
                break
            print(
                f"Retrying image predict for '{model_id}' after error: {e}. Attempt #{attempt + 1}"
            )
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
    request = service_pb2.ListModelsRequest(per_page=per_page, featured_only=True)
    response = stub.ListModels(request, metadata=metadata(pat=True))
    if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"ListModels failed: {response.status.description}")
    return response.models


def _list_openai_featured_models():
    if not os.environ.get('CLARIFAI_PAT_KEY'):
        return ["Missing API KEY"]

    open_ai_models = []
    for model in _list_featured_models():
        method_signatures = getattr(model.model_version, "method_signatures", [])
        if any(ms.name == "openai_transport" for ms in method_signatures):
            open_ai_models.append(f"{model.user_id}/{model.app_id}/models/{model.id}")

    return open_ai_models


# The test functions below remain unchanged as the retry logic
# is now encapsulated within the _call_openai_model helper.


# New integrated test
@pytest.mark.parametrize("model_identifier", _list_openai_featured_models())
def test_openai_compatible_endpoint_on_featured_models(model_identifier):
    """Tests the OpenAI compatible endpoint with featured models."""
    if not os.environ.get('CLARIFAI_PAT_KEY'):
        pytest.skip("Skipping test: CLARIFAI_PAT_KEY environment variable not set.")

    if model_identifier.startswith("anthropic"):
        # TODO: Re-enable anthropic tests
        pytest.skip("Anthropic models are currently disabled")

    _, error = _call_openai_model(model_identifier)
    assert not error


# New integrated async test
async def _call_openai_model_async(model_identifier, session):
    """Async helper to call a single model."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _call_openai_model, model_identifier)


@pytest.mark.asyncio
async def test_openai_compatible_endpoint_on_featured_models_async():
    """Tests the OpenAI compatible endpoint concurrently with featured models."""
    if not os.environ.get('CLARIFAI_PAT_KEY'):
        pytest.skip("Skipping test: CLARIFAI_PAT_KEY environment variable not set.")

    tasks = []
    # TODO: Re-enable anthropic tests
    model_identifiers = [
        m for m in _list_openai_featured_models() if not m.startswith("anthropic")
    ]

    for model_identifier in model_identifiers:
        tasks.append(_call_openai_model_async(model_identifier, None))

    results = await asyncio.gather(*tasks)
    failed_models = []
    for i, (_, error) in enumerate(results):
        if error:
            failed_models.append({model_identifiers[i]: error})

    assert not failed_models, f"The following OpenAI compatible models failed: {failed_models}"


# Tool calling test suite

# Hardcoded list of models to test for tool calling support
# These should be models from the featured list that support OpenAI tool calling
TOOL_CALLING_TEST_MODELS = [
    "https://clarifai.com/openai/chat-completion/models/gpt-oss-20b",
    #"https://clarifai.com/qwen/qwenLM/models/Qwen3-30B-A3B-Instruct-2507",
    #"https://clarifai.com/qwen/qwenLM/models/Qwen3-30B-A3B-Thinking-2507",
]

# Parameter combinations to test
TOOL_CALLING_CONFIGS = [
    {"stream": True, "tool_choice": "required", "strict": True},
    {"stream": True, "tool_choice": "required", "strict": False},
    {"stream": True, "tool_choice": "auto", "strict": True},
    {"stream": True, "tool_choice": "auto", "strict": False},
    {"stream": False, "tool_choice": "required", "strict": True},
    {"stream": False, "tool_choice": "required", "strict": False},
    {"stream": False, "tool_choice": "auto", "strict": True},
    {"stream": False, "tool_choice": "auto", "strict": False},
]

# Tool definition for weather query
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "additionalProperties": False,
            "required": ["location", "unit"],
        },
    },
}


def _call_openai_tool_calling(model_url, config):
    """
    Call OpenAI-compatible endpoint with tool calling using the specified configuration.

    Args:
        model_url: Full URL to the Clarifai model
        config: Dictionary with stream, tool_choice, and strict parameters

    Returns:
        tuple: (response, error) where error is None on success
    """
    channel = ClarifaiChannel.get_grpc_channel()
    client = OpenAI(
        api_key=os.environ.get('CLARIFAI_PAT_KEY'),
        base_url=f"https://{channel._target}/v2/ext/openai/v1",
        default_headers={"X-Clarifai-Request-Id-Prefix": f"python-openai-{CLIENT_VERSION}"},
    )

    # Build tool definition with strict parameter
    tool = WEATHER_TOOL.copy()
    tool["function"]["strict"] = config["strict"]

    try:
        # Build request parameters
        request_params = {
            "model": model_url,
            "messages": [
                {"role": "user", "content": "What is the weather like in Boston in fahrenheit?"}
            ],
            "temperature": 1,
            "top_p": 1,
            "max_tokens": 32768,
            "stream": config["stream"],
            "tool_choice": config["tool_choice"],
            "tools": [tool],
        }

        # Add stream_options only if streaming is enabled
        if config["stream"]:
            request_params["stream_options"] = {"include_usage": True}

        response = client.chat.completions.create(**request_params)

        # Handle streaming vs non-streaming responses differently
        if config["stream"]:
            # For streaming, we need to consume the iterator
            chunks = []
            for chunk in response:
                chunks.append(chunk)
            return chunks, None
        else:
            # For non-streaming, return the response directly
            return response, None

    except Exception as e:
        return None, str(e)


def _is_valid_tool_arguments(arguments):
    """Check if arguments string is valid JSON with required fields."""
    try:
        args = json.loads(arguments)
        return isinstance(args, dict) and "location" in args and "unit" in args
    except (json.JSONDecodeError, TypeError):
        return False


def _validate_tool_calling_response(response, config):
    """
    Validate tool calling response with clear assertion messages.

    Validation criteria:
    - Streaming: finish_reason='tool_calls', usage info, exactly one tool call with valid JSON
    - Non-streaming: tool_calls present with valid JSON arguments
    """
    assert response is not None, "Response is None"

    if config["stream"]:
        assert isinstance(response, list) and response, f"Invalid streaming response: {type(response)}"

        # Check finish_reason and usage
        has_finish_reason = any(
            chunk.choices and chunk.choices[0].finish_reason == 'tool_calls' for chunk in response
        )
        has_usage = any(hasattr(chunk, 'usage') and chunk.usage for chunk in response)

        assert has_usage, "Missing usage info in streaming response"

        if config["tool_choice"] == "required":
            assert has_finish_reason, "Missing finish_reason='tool_calls'"

            # Find chunks that contain tool calls
            chunks_with_tool_calls = [
                chunk for chunk in response if chunk.choices and chunk.choices[0].delta.tool_calls
            ]

            # Validate exactly ONE chunk contains tool calls
            assert (
                len(chunks_with_tool_calls) == 1
            ), f"Expected exactly 1 chunk with tool calls, got {len(chunks_with_tool_calls)}"

            # Get the single chunk's tool calls
            tool_calls = chunks_with_tool_calls[0].choices[0].delta.tool_calls

            # Validate exactly one tool call
            assert len(tool_calls) == 1, f"Expected exactly 1 tool call, got {len(tool_calls)}"

            tool_call = tool_calls[0]

            # Validate has function name
            assert (
                tool_call.function and tool_call.function.name
            ), "Tool call missing function or name"

            # Validate has complete valid JSON arguments
            assert (
                tool_call.function.arguments
                and _is_valid_tool_arguments(tool_call.function.arguments)
            ), f"Invalid or missing arguments: {tool_call.function.arguments if tool_call.function else 'N/A'}"

    else:
        # Non-streaming
        assert hasattr(response, 'choices') and response.choices, "Response missing choices"

        message = response.choices[0].message

        if config["tool_choice"] == "required":
            assert hasattr(message, 'tool_calls') and message.tool_calls, "Message missing tool_calls"

            tool_call = message.tool_calls[0]
            assert tool_call.function and tool_call.function.name, "Tool call missing function or name"

            assert _is_valid_tool_arguments(
                tool_call.function.arguments
            ), f"Invalid arguments: {tool_call.function.arguments}"


def _get_tool_calling_models():
    """
    Get the list of models to test for tool calling.
    Filters the hardcoded list against featured models.

    Returns:
        list: List of model URLs that support tool calling
    """
    if not os.environ.get('CLARIFAI_PAT_KEY'):
        return ["Missing API KEY"]

    # For now, just return the hardcoded list
    # TODO: Optionally filter against _list_openai_featured_models() to ensure models exist
    return TOOL_CALLING_TEST_MODELS


# Generate test parameters: cartesian product of models and configs
def _generate_tool_calling_test_params():
    """Generate all combinations of models and configurations for testing."""
    models = _get_tool_calling_models()
    params = []
    for model in models:
        for config in TOOL_CALLING_CONFIGS:
            # Create a readable test ID
            test_id = f"{model.split('/')[-1]}-stream_{config['stream']}-choice_{config['tool_choice']}-strict_{config['strict']}"
            params.append((model, config, test_id))
    return params


@pytest.mark.parametrize(
    "model_url,config,test_id",
    [pytest.param(m, c, tid, id=tid) for m, c, tid in _generate_tool_calling_test_params()],
)
def test_openai_tool_calling_with_parameter_combinations(model_url, config, test_id):
    """
    Test OpenAI-compatible tool calling with various parameter combinations.

    This test verifies that models support tool calling with different combinations of:
    - stream: True/False (streaming vs non-streaming responses)
    - tool_choice: "required"/"auto" (force tool use vs let model decide)
    - strict: True/False (strict schema validation)

    Each combination is tested and validated against expected behavior.
    """
    if not os.environ.get('CLARIFAI_PAT_KEY'):
        pytest.skip("Skipping test: CLARIFAI_PAT_KEY environment variable not set.")

    response, error = _call_openai_tool_calling(model_url, config)

    # Assert no error occurred
    assert not error, f"Tool calling failed for {model_url} with config {config}: {error}"

    # Validate response (raises AssertionError with clear message if validation fails)
    _validate_tool_calling_response(response, config)
