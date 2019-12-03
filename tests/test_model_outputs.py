import os

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2

SAMPLE_URL = 'https://samples.clarifai.com/dog2.jpeg'


def test_predict_on_json_channel():
  channel = ClarifaiChannel.get_json_channel()
  response = _post_model_outputs(channel)
  assert len(response.outputs[0].data.concepts) > 0


def test_predict_on_grpc_channel():
  channel = ClarifaiChannel.get_insecure_grpc_channel()
  response = _post_model_outputs(channel)
  assert len(response.outputs[0].data.concepts) > 0


def _post_model_outputs(channel):
  stub = service_pb2_grpc.V2Stub(channel)
  request = service_pb2.PostModelOutputsRequest(
    model_id='aaa03c23b3724a16a56b629203edc62c',
    inputs=[
      resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=SAMPLE_URL)))
    ])
  metadata = (('authorization', 'Key %s' % os.environ.get('CLARIFAI_API_KEY')),)
  response = stub.PostModelOutputs(request, metadata=metadata)
  return response

