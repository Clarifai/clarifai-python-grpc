import os

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel

from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import both_channels, raise_on_failure


def test_list_apps():
  # channel = service_pb2_grpc.grpc.insecure_channel("api.clarifai.com")
  channel = ClarifaiChannel.get_insecure_grpc_channel()
  stub = service_pb2_grpc.V2Stub(channel)
  metadata = (("authorization", "Key %s" % os.environ.get("CLARIFAI_PAT_KEY")),)

  list_apps_response = stub.ListApps(
    service_pb2.ListAppsRequest(
      user_app_id=resources_pb2.UserAppIDSet(
        user_id="me",
      )
    ),
    metadata=metadata,
  )
  # We should have at least one app. If this turns out not to be the case and the
  # test fails, we should create it in this test.
  assert list_apps_response.apps