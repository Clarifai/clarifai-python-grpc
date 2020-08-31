import uuid

from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from tests.common import both_channels, metadata, raise_on_failure


@both_channels
def test_post_patch_get_delete_workflow(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    # Max workflow ID is capped at 32 chars.
    workflow_id = "food-and-general-" + uuid.uuid4().hex[:15]

    post_workflows_response = stub.PostWorkflows(
        service_pb2.PostWorkflowsRequest(
            workflows=[
                resources_pb2.Workflow(
                    id=workflow_id,
                    nodes=[
                        resources_pb2.WorkflowNode(
                            id="the-general-node",
                            # This is the public General model.
                            model=resources_pb2.Model(
                                id="aaa03c23b3724a16a56b629203edc62c",
                                model_version=resources_pb2.ModelVersion(
                                    id="aa7f35c01e0642fda5cf400f543e7c40"
                                ),
                            ),
                        ),
                        resources_pb2.WorkflowNode(
                            id="the-food-node",
                            # This is the public Food model.
                            model=resources_pb2.Model(
                                id="bd367be194cf45149e75f01d59f77ba7",
                                model_version=resources_pb2.ModelVersion(
                                    id="dfebc169854e429086aceb8368662641"
                                ),
                            ),
                        ),
                    ],
                )
            ],
        ),
        metadata=metadata(),
    )
    raise_on_failure(post_workflows_response)

    try:
        # Update the workflow to use an older General model version.
        patch_workflows_response = stub.PatchWorkflows(
            service_pb2.PatchWorkflowsRequest(
                action="overwrite",
                workflows=[
                    resources_pb2.Workflow(
                        id=workflow_id,
                        nodes=[
                            resources_pb2.WorkflowNode(
                                id="the-general-node",
                                # This is the public General model, but the version is from 2016.
                                model=resources_pb2.Model(
                                    id="aaa03c23b3724a16a56b629203edc62c",
                                    model_version=resources_pb2.ModelVersion(
                                        id="aa9ca48295b37401f8af92ad1af0d91d"
                                    ),
                                ),
                            ),
                            resources_pb2.WorkflowNode(
                                id="the-food-node",
                                # This is the public Food model.
                                model=resources_pb2.Model(
                                    id="bd367be194cf45149e75f01d59f77ba7",
                                    model_version=resources_pb2.ModelVersion(
                                        id="dfebc169854e429086aceb8368662641"
                                    ),
                                ),
                            ),
                        ],
                    )
                ],
            ),
            metadata=metadata(),
        )
        raise_on_failure(patch_workflows_response)

        get_workflow_response = stub.GetWorkflow(
            service_pb2.GetWorkflowRequest(workflow_id=workflow_id), metadata=metadata()
        )
        raise_on_failure(get_workflow_response)

        # Make sure that after patching the workflow actually has the older General model version
        # ID.
        assert (
            get_workflow_response.workflow.nodes[0].model.model_version.id
            == "aa9ca48295b37401f8af92ad1af0d91d"
        )
    finally:
        delete_workflow_response = stub.DeleteWorkflow(
            service_pb2.DeleteWorkflowRequest(workflow_id=workflow_id), metadata=metadata()
        )
        raise_on_failure(delete_workflow_response)
