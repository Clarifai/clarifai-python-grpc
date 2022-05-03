import json
import os

import requests

from clarifai_grpc.channel.grpc_json_channel import GRPCJSONChannel
from clarifai_grpc.grpc.api import service_pb2_grpc

RETRIES = 2  # if connections fail retry a couple times.
CONNECTIONS = 20  # number of connections to maintain in pool.

wrap_response_deserializer = None

grpc_json_config = json.dumps(
    {
        "methodConfig": [
            {
                "name": [{"service": "clarifai.api.V2"}],
                "retryPolicy": {
                    "maxAttempts": 5,
                    "initialBackoff": "1s",
                    "maxBackoff": "5s",
                    "backoffMultiplier": 1.5,
                    "retryableStatusCodes": ["UNAVAILABLE"],
                },
            }
        ]
    }
)


def _response_deserializer_for_json(response_deserializer):
    return response_deserializer


def _response_deserializer_for_grpc(response_deserializer):
    return response_deserializer.FromString


class ClarifaiChannel:
    @classmethod
    def get_json_channel(
        cls, base_url=os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com")
    ):
        global wrap_response_deserializer
        wrap_response_deserializer = _response_deserializer_for_json

        session = cls._make_requests_session()

        return GRPCJSONChannel(session=session, base_url=base_url)

    @staticmethod
    def _make_requests_session():
        http_adapter = requests.adapters.HTTPAdapter(
            max_retries=RETRIES, pool_connections=CONNECTIONS, pool_maxsize=CONNECTIONS
        )

        session = requests.Session()
        session.mount("http://", http_adapter)
        session.mount("https://", http_adapter)
        return session

    @staticmethod
    def get_grpc_channel(base=None):
        global wrap_response_deserializer
        wrap_response_deserializer = _response_deserializer_for_grpc

        if not base:
            base = os.environ.get("CLARIFAI_GRPC_BASE", "api.clarifai.com")

        return service_pb2_grpc.grpc.secure_channel(
            base,
            service_pb2_grpc.grpc.ssl_channel_credentials(),
            options=[("grpc.service_config", grpc_json_config)],
        )

    @staticmethod
    def get_insecure_grpc_channel(base=None, port=18080):
        global wrap_response_deserializer
        wrap_response_deserializer = _response_deserializer_for_grpc

        if not base:
            base = os.environ.get("CLARIFAI_GRPC_BASE", None)

        if not base:
            raise ValueError("Please set 'base' via arguments or env variable CLARIFAI_GRPC_BASE")

        channel_address = "{}:{}".format(base, port)

        return service_pb2_grpc.grpc.insecure_channel(
            channel_address, options=[("grpc.service_config", grpc_json_config)]
        )
