import os

from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2

SAMPLE_URL = 'https://samples.clarifai.com/dog2.jpeg'

API_KEY = os.environ.get('CLARIFAI_API_KEY', 'no_key')


def test_PostModelOutputs():
  # creds = service_pb2_grpc.grpc.ssl_channel_credentials()
  # channel = service_pb2_grpc.grpc.secure_channel('api-grpc.clarifai.com:18081', creds)

  channel = service_pb2_grpc.grpc.insecure_channel('api-grpc.clarifai.com:18080')

  stub = service_pb2_grpc.V2Stub(channel)

  request = service_pb2.PostModelOutputsRequest(
      model_id='aaa03c23b3724a16a56b629203edc62c',
      inputs=[
          resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=SAMPLE_URL)))
      ])

  metadata = (('authorization', 'Key %s' % API_KEY),)

  res = stub.PostModelOutputs(request, metadata=metadata)

  for concept in res.outputs[0].data.concepts:
    print("%12s: %.2f%%" % (concept.name, concept.value * 100))


test_PostModelOutputs()
