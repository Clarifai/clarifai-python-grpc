![Clarifai logo](docs/logo.png)

# Clarifai Python gRPC Client

This is the official Clarifai Python gRPC client.

In addition to the actual gRPC channel, it is also possible to use a JSON channel, which uses 
JSON & HTTP/REST, while using the same request / response handling code.

[![PyPI version](https://pypip.in/v/clarifai-grpc/badge.png)](https://pypi.python.org/pypi/clarifai-grpc)

## Installation

```cmd
pip install clarifai-grpc
```

## Getting started

Construct the channel you want to use:

```python
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel

# Construct one of the channels you want to use
channel = ClarifaiChannel.get_json_channel()
channel = ClarifaiChannel.get_insecure_grpc_channel()

# Note: You can also use a secure (encrypted) ClarifaiChannel.get_grpc_channel() however
# it is currently not possible to use it with the latest gRPC version.
```

Predict concepts in an image:

```python
from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

stub = service_pb2_grpc.V2Stub(channel)

request = service_pb2.PostModelOutputsRequest(
    model_id='aaa03c23b3724a16a56b629203edc62c',
    inputs=[
      resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url='YOUR_IMAGE_URL')))
    ])
metadata = (('authorization', 'Key YOUR_CLARIFAI_API_KEY'),)

response = stub.PostModelOutputs(request, metadata=metadata)

if response.status.code != status_code_pb2.SUCCESS:
  raise Exception("Request failed, status code: " + str(response.status.code))


for concept in response.outputs[0].data.concepts:
    print('%12s: %.2f' % (concept.name, concept.value))
```
