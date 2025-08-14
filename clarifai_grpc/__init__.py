__version__ = "11.7.3"


import os

# pop off env var set to the old python implementation
if os.environ.get('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', None) == 'python':
    os.environ.pop('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION')

try:
    from google.protobuf.internal import api_implementation

    if (
        api_implementation.Type() == 'python'
        and not os.environ.get('CLARIFAI_SKIP_PROTOBUF_CHECK', 'false') == 'true'
    ):
        raise Exception(
            "We do not recommend running this library with the Python implementation of Protocol Buffers. Please check your installation to use the cpp or upb implementation. We recommend setting the environment variable PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb. You can skip this check by setting CLARIFAI_SKIP_PROTOBUF_CHECK=true"
        )
except ImportError:
    pass
