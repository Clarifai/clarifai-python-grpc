__version__ = "11.10.4"

import os

# pop off env var set to the old python implementation
unset = False
if os.environ.get('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', None) == 'python':
    unset = True
    os.environ.pop('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION')
# Don't use the clarifai logger since it is in the SDK package and depends on protobuf.
import logging

logger = logging.getLogger(__name__)

if unset:
    logger.warning(
        "Unsetting PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python env var while importing clarifai package. It's best to unset this env var in your environemnt for faster performance."
    )

try:
    from google.protobuf.internal import api_implementation

    if api_implementation.Type() == 'python':
        logger.warning(
            "The python version of google protobuf is being used. We recommend that you upgrade the protobuf package >=4.21.0 to use the upd version which is much faster."
        )
    #     and not os.environ.get('CLARIFAI_SKIP_PROTOBUF_CHECK', 'false') == 'true'
    # ):
    #     raise Exception(
    #         "We do not recommend running this library with the Python implementation of Protocol Buffers. Please check your installation to use the cpp or upb implementation. We recommend setting the environment variable PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb. You can skip this check by setting CLARIFAI_SKIP_PROTOBUF_CHECK=true"
    #     )
except ImportError:
    pass
