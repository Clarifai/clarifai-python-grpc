import pytest

from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import (
    BEER_VIDEO_URL,
    CONAN_GIF_VIDEO_URL,
    GENERAL_MODEL_ID,
    MAIN_APP_ID,
    MAIN_APP_USER_ID,
    TOY_VIDEO_FILE_PATH,
    both_channels,
    get_channel,
    metadata,
    post_model_outputs_and_maybe_allow_retries,
    raise_on_failure,
)

# Skip all video tests - video decoding not implemented in V3 runners
pytestmark = pytest.mark.skip(reason="Video decoding not implemented in V3 runners")


@both_channels()
def test_predict_video_url(channel_key):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(video=resources_pb2.Video(url=CONAN_GIF_VIDEO_URL))
            )
        ],
    )
    response = post_model_outputs_and_maybe_allow_retries(
        stub, request, metadata=metadata(pat=True)
    )
    raise_on_failure(response)

    assert len(response.outputs[0].data.frames) > 0
    for frame in response.outputs[0].data.frames:
        assert len(frame.data.concepts) > 0


@both_channels()
def test_predict_video_url_with_min_value(channel_key):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(video=resources_pb2.Video(url=CONAN_GIF_VIDEO_URL))
            )
        ],
        model=resources_pb2.Model(
            output_info=resources_pb2.OutputInfo(
                output_config=resources_pb2.OutputConfig(min_value=0.95)
            )
        ),
    )
    response = post_model_outputs_and_maybe_allow_retries(
        stub, request, metadata=metadata(pat=True)
    )
    raise_on_failure(response)

    assert len(response.outputs[0].data.frames) > 0
    for frame in response.outputs[0].data.frames:
        assert len(frame.data.concepts) > 0
        for concept in frame.data.concepts:
            assert concept.value >= 0.95


@both_channels()
def test_predict_video_url_with_max_concepts(channel_key):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(video=resources_pb2.Video(url=CONAN_GIF_VIDEO_URL))
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

    assert len(response.outputs[0].data.frames) > 0
    for frame in response.outputs[0].data.frames:
        assert len(frame.data.concepts) == 3


@both_channels()
def test_predict_video_url_with_custom_sample_ms(channel_key):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(video=resources_pb2.Video(url=BEER_VIDEO_URL))
            )
        ],
        model=resources_pb2.Model(
            output_info=resources_pb2.OutputInfo(
                output_config=resources_pb2.OutputConfig(sample_ms=2000)
            )
        ),
    )
    response = post_model_outputs_and_maybe_allow_retries(
        stub, request, metadata=metadata(pat=True)
    )
    raise_on_failure(response)

    # The expected time per frame is the middle between the start and the end of the frame
    # (in milliseconds).
    expected_time = 1000

    assert len(response.outputs[0].data.frames) > 0
    for frame in response.outputs[0].data.frames:
        assert frame.frame_info.time == expected_time
        expected_time += 2000


@both_channels()
def test_predict_video_bytes(channel_key):
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))

    with open(TOY_VIDEO_FILE_PATH, "rb") as f:
        file_bytes = f.read()

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(video=resources_pb2.Video(base64=file_bytes))
            )
        ],
    )
    response = post_model_outputs_and_maybe_allow_retries(
        stub, request, metadata=metadata(pat=True)
    )
    raise_on_failure(response)

    assert len(response.outputs[0].data.frames) > 0
    for frame in response.outputs[0].data.frames:
        assert len(frame.data.concepts) > 0
