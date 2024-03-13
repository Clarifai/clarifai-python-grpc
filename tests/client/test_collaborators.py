import os

from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import both_channels, raise_on_failure


@both_channels
def test_list_collaborators_with_pat(channel):
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (("authorization", "Key %s" % os.environ.get("CLARIFAI_PAT_KEY")),)

    list_apps_response = stub.ListApps(
        service_pb2.ListAppsRequest(
            user_app_id=resources_pb2.UserAppIDSet(
                user_id="me",
            )
        ),
        metadata=metadata,
        insecure=os.environ.get("CLARIFAI_INSECURE_GRPC", False),
    )
    # We should have at least one app. If this turns out not to be the case and the
    # test fails, we should create it in this test.
    assert list_apps_response.apps
    app_id = list_apps_response.apps[0].id

    list_collaborators_response = stub.ListCollaborators(
        service_pb2.ListCollaboratorsRequest(
            user_app_id=resources_pb2.UserAppIDSet(
                user_id="me",
                app_id=app_id,
            )
        ),
        metadata=metadata,
        insecure=os.environ.get("CLARIFAI_INSECURE_GRPC", False),
    )
    raise_on_failure(list_collaborators_response)
