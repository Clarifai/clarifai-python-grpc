![Clarifai logo](docs/logo.png)

# Clarifai Python gRPC Clientt

This is the official Clarifai gRPC Python client for interacting with our powerful recognition
[API](https://docs.clarifai.com).
Clarifai provides a platform for data scientists, developers, researchers and enterprises to master the entire
artificial intelligence lifecycle. Gather valuable business insights from images, video and text using computer vision
and natural language processing.

* Try the Clarifai demo at: https://clarifai.com/demo
* Sign up for a free account at: https://portal.clarifai.com/signup
* Read the documentation at: https://docs.clarifai.com/


[![PyPI version](https://pypip.in/v/clarifai-grpc/badge.png)](https://pypi.python.org/pypi/clarifai-grpc)
[![Build](https://github.com/Clarifai/clarifai-python-grpc/workflows/Run%20tests/badge.svg)](https://github.com/Clarifai/clarifai-python-grpc/actions)

## Installation

```cmd
pip install clarifai-grpc
```

## Versioning

This library doesn't use semantic versioning. The first two version numbers (`X.Y` out of `X.Y.Z`) follow the API (backend) versioning, and
whenever the API gets updated, this library follows it.

The third version number (`Z` out of `X.Y.Z`) is used by this library for any independent releases of library-specific improvements and bug fixes.

## Getting started

Construct the `V2Stub` object using which you'll access all the Clarifai API functionality:

```python
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc

stub = service_pb2_grpc.V2Stub(ClarifaiChannel.get_grpc_channel())
```

> Alternatives to the encrypted gRPC channel (`ClarifaiChannel.get_grpc_channel()`) are:
> - the HTTPS+JSON channel (`ClarifaiChannel.get_json_channel()`), and
> - the unencrypted gRPC channel (`ClarifaiChannel.get_insecure_grpc_channel()`).
>
> We only recommend them in special cases.

Predict concepts in an image:

```python
from clarifai_grpc.grpc.api import service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

# This is how you authenticate.
metadata = (('authorization', f'Key {YOUR_CLARIFAI_API_KEY}'),)

request = service_pb2.PostModelOutputsRequest(
    # This is the model ID of a publicly available General model. You may use any other public or custom model ID.
    model_id='aaa03c23b3724a16a56b629203edc62c',
    inputs=[
      resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url='YOUR_IMAGE_URL')))
    ])
response = stub.PostModelOutputs(request, metadata=metadata)

if response.status.code != status_code_pb2.SUCCESS:
    print("There was an error with your request!")
    print("\tCode: {}".format(response.outputs[0].status.code))
    print("\tDescription: {}".format(response.outputs[0].status.description))
    print("\tDetails: {}".format(response.outputs[0].status.details))
    raise Exception("Request failed, status code: " + str(response.status.code))

for concept in response.outputs[0].data.concepts:
    print('%12s: %.2f' % (concept.name, concept.value))
```

See [the Clarifai API documentation](https://docs.clarifai.com/) for all available functionality.

## Troubleshooting

#### I get the following error when installing the library: `Failed building wheel for grpcio`

Try upgrading **setuptools** to a version `40.7.1` or higher.
```
pip install --upgrade setuptools
```
Source: https://github.com/grpc/grpc/issues/17829
