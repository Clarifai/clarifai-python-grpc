from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import (
    DOG_IMAGE_URL,
    MAIN_APP_ID,
    MAIN_APP_USER_ID,
    RED_TRUCK_IMAGE_FILE_PATH,
    both_channels,
    metadata,
    raise_on_failure,
)


@both_channels
def test_workflow_predict_image_url(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    post_workflows_response = stub.PostWorkflowResults(
        service_pb2.PostWorkflowResultsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
            workflow_id="General",
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL))
                )
            ],
            output_config=resources_pb2.OutputConfig(max_concepts=3),
        ),
        metadata=metadata(pat=True),
    )
    raise_on_failure(post_workflows_response)

    assert len(post_workflows_response.results[0].outputs[0].data.concepts) == 3


@both_channels
def test_workflow_predict_image_bytes(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    with open(RED_TRUCK_IMAGE_FILE_PATH, "rb") as f:
        file_bytes = f.read()

    post_workflows_response = stub.PostWorkflowResults(
        service_pb2.PostWorkflowResultsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID),
            workflow_id="General",
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(image=resources_pb2.Image(base64=file_bytes))
                )
            ],
            output_config=resources_pb2.OutputConfig(max_concepts=3),
        ),
        metadata=metadata(pat=True),
    )
    raise_on_failure(post_workflows_response)

    assert len(post_workflows_response.results[0].outputs[0].data.concepts) == 3
