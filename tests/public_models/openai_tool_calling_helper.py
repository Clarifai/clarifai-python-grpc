"""
Helper utilities for constructing and validating OpenAI tool calling requests.

This module provides functions and constants for testing OpenAI-compatible
tool calling functionality with Clarifai models.
"""

import json
import os
import time

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.channel.http_client import CLIENT_VERSION
from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from tests.common import metadata

# Maximum retry attempts for API calls
MAX_RETRY_ATTEMPTS = 3

# Parameter combinations to test
# For now, we only test non-streaming with tool calling
TOOL_CALLING_CONFIGS = [
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


def call_openai_tool_calling(model_url, config):
    """
    Call OpenAI-compatible endpoint with tool calling using the specified configuration.
    Includes retry mechanism for transient errors.

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
        timeout=10 # 10 seconds timeout to avoid hanging
    )

    # Build tool definition with strict parameter
    tool = WEATHER_TOOL.copy()
    tool["function"]["strict"] = config["strict"]

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

    last_error = None

    # Retry loop for transient errors
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
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

        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            last_error = e
            if attempt == MAX_RETRY_ATTEMPTS - 1:
                break
            print(
                f"Retrying tool calling for '{model_url}' after error: {e}. "
                f"Attempt #{attempt + 1}"
            )
            time.sleep(attempt + 1)
        except Exception as e:
            last_error = e
            break

    return None, str(last_error)


def is_valid_tool_arguments(arguments):
    """Check if arguments string is valid JSON with required fields."""
    try:
        args = json.loads(arguments)
        return isinstance(args, dict) and "location" in args and "unit" in args
    except (json.JSONDecodeError, TypeError):
        return False


def validate_tool_calling_response(response, config):
    """
    Validate tool calling response with clear assertion messages.

    Validation criteria:
    - Streaming: finish_reason='tool_calls', usage info, exactly one tool call with valid JSON
    - Non-streaming: tool_calls present with valid JSON arguments
    """
    assert response is not None, "Response is None"

    if config["stream"]:
        assert isinstance(
            response, list
        ) and response, f"Invalid streaming response: {type(response)}"

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
                tool_call.function.arguments and is_valid_tool_arguments(tool_call.function.arguments)
            ), f"Invalid or missing arguments: {tool_call.function.arguments if tool_call.function else 'N/A'}"

    else:
        # Non-streaming
        assert hasattr(response, 'choices') and response.choices, "Response missing choices"

        message = response.choices[0].message

        if config["tool_choice"] == "required":
            assert hasattr(message, 'tool_calls') and message.tool_calls, "Message missing tool_calls"

            tool_call = message.tool_calls[0]
            assert tool_call.function and tool_call.function.name, "Tool call missing function or name"

            assert is_valid_tool_arguments(
                tool_call.function.arguments
            ), f"Invalid arguments: {tool_call.function.arguments}"


def _list_featured_models_with_use_case_filters(per_page=50, use_cases=None):
    """Lists featured models from the Clarifai platform."""
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    request = service_pb2.ListModelsRequest(per_page=per_page, featured_only=True, use_cases=use_cases)
    response = stub.ListModels(request, metadata=metadata(pat=True))
    if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"ListModels failed: {response.status.description}")
    return response.models


def get_tool_calling_models():
    """
    Get the list of models to test for tool calling.
    """
    if not os.environ.get('CLARIFAI_PAT_KEY'):
        return ["Missing API KEY"]

    # Get models with function-calling use case
    models_with_use_case = _list_featured_models_with_use_case_filters(
        per_page=100, use_cases=['function-calling']
    )

    tool_calling_models = []
    for model in models_with_use_case:
        # Also check for openai_transport support
        method_signatures = getattr(model.model_version, "method_signatures", [])
        if any(ms.name == "openai_transport" for ms in method_signatures):
            model_url = f"https://clarifai.com/{model.user_id}/{model.app_id}/models/{model.id}"
            tool_calling_models.append(model_url)

    return tool_calling_models


def generate_tool_calling_test_params():
    """Generate all combinations of models and configurations for testing."""
    models = get_tool_calling_models()
    params = []
    for model in models:
        for config in TOOL_CALLING_CONFIGS:
            # Create a readable test ID
            test_id = f"{model.split('/')[-1]}-stream_{config['stream']}-choice_{config['tool_choice']}-strict_{config['strict']}"
            params.append((model, config, test_id))
    return params
