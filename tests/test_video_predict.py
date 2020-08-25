from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import (
    CONAN_GIF_VIDEO_URL,
    GENERAL_MODEL_ID,
    TOY_VIDEO_FILE_PATH,
    both_channels,
    metadata,
    raise_on_failure,
)


@both_channels
def test_predict_video_url(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    request = service_pb2.PostModelOutputsRequest(
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(video=resources_pb2.Video(url=CONAN_GIF_VIDEO_URL))
            )
        ],
    )
    response = stub.PostModelOutputs(request, metadata=metadata())
    raise_on_failure(response)

    assert len(response.outputs[0].data.frames) > 0
    for frame in response.outputs[0].data.frames:
        assert len(frame.data.concepts) > 0


@both_channels
def test_predict_video_url_with_min_value(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    request = service_pb2.PostModelOutputsRequest(
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
    response = stub.PostModelOutputs(request, metadata=metadata())
    raise_on_failure(response)

    assert len(response.outputs[0].data.frames) > 0
    for frame in response.outputs[0].data.frames:
        assert len(frame.data.concepts) > 0
        for concept in frame.data.concepts:
            assert concept.value >= 0.95


@both_channels
def test_predict_video_url_with_max_concepts(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    request = service_pb2.PostModelOutputsRequest(
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
    response = stub.PostModelOutputs(request, metadata=metadata())
    raise_on_failure(response)

    assert len(response.outputs[0].data.frames) > 0
    for frame in response.outputs[0].data.frames:
        assert len(frame.data.concepts) == 3


@both_channels
def test_predict_video_bytes(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with open(TOY_VIDEO_FILE_PATH, "rb") as f:
        file_bytes = f.read()

    request = service_pb2.PostModelOutputsRequest(
        model_id=GENERAL_MODEL_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(video=resources_pb2.Video(base64=file_bytes))
            )
        ],
    )
    response = stub.PostModelOutputs(request, metadata=metadata())
    raise_on_failure(response)

    assert len(response.outputs[0].data.frames) > 0
    for frame in response.outputs[0].data.frames:
        assert len(frame.data.concepts) > 0
