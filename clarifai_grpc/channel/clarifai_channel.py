import json
import os

import grpc

from clarifai_grpc.channel.grpc_json_channel import GRPCJSONChannel

RETRIES = 2  # if connections fail retry a couple times.
CONNECTIONS = 20  # number of connections to maintain in pool.
MAX_MESSAGE_LENGTH = 128 * 1024 * 1024  # 128MB

wrap_response_deserializer = None

grpc_json_config = json.dumps(
    {
        "methodConfig": [
            {
                "name": [{"service": "clarifai.api.V2"}],
                "retryPolicy": {
                    "maxAttempts": 5,
                    "initialBackoff": "0.01s",
                    "maxBackoff": "5s",
                    "backoffMultiplier": 2,
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
        import requests  # noqa

        http_adapter = requests.adapters.HTTPAdapter(
            max_retries=RETRIES, pool_connections=CONNECTIONS, pool_maxsize=CONNECTIONS
        )

        session = requests.Session()
        session.mount("http://", http_adapter)
        session.mount("https://", http_adapter)
        return session

    @staticmethod
    def get_grpc_channel(base=None, root_certificates_path=None):
        global wrap_response_deserializer
        wrap_response_deserializer = _response_deserializer_for_grpc

        if not base:
            base = os.environ.get("CLARIFAI_GRPC_BASE", "api.clarifai.com")
        if base.startswith("http:") or base.startswith("https:"):
            raise ValueError(
                "For secure channels the 'base' passed via arguments or env variable CLARIFAI_GRPC_BASE should not start with http:// or https:// but be a direct api endpoint like 'api.clarifai.com'"
            )

        if root_certificates_path:
            with open(root_certificates_path, "rb") as f:
                root_certificates = f.read()
            credentials = grpc.ssl_channel_credentials(root_certificates)
        else:
            credentials = grpc.ssl_channel_credentials()

        return grpc.secure_channel(
            base,
            credentials,
            options=[
                ("grpc.service_config", grpc_json_config),
                ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
            ],
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

        return grpc.insecure_channel(
            channel_address,
            options=[
                ("grpc.service_config", grpc_json_config),
                ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
            ],
        )

    @staticmethod
    def get_aio_grpc_channel(base=None, root_certificates_path=None):
        global wrap_response_deserializer
        wrap_response_deserializer = _response_deserializer_for_grpc

        if not base:
            base = os.environ.get("CLARIFAI_GRPC_BASE", "api.clarifai.com")
        if base.startswith("http:") or base.startswith("https:"):
            raise ValueError(
                "For secure channels the 'base' passed via arguments or env variable CLARIFAI_GRPC_BASE should not start with http:// or https:// but be a direct api endpoint like 'api.clarifai.com'"
            )

        if root_certificates_path:
            with open(root_certificates_path, "rb") as f:
                root_certificates = f.read()
            credentials = grpc.ssl_channel_credentials(root_certificates)
        else:
            credentials = grpc.ssl_channel_credentials()

        return grpc.aio.secure_channel(
            base,
            credentials,
            options=[
                ("grpc.service_config", grpc_json_config),
                ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
            ],
        )

    @staticmethod
    def get_aio_insecure_grpc_channel(base=None, port=18080):
        global wrap_response_deserializer
        wrap_response_deserializer = _response_deserializer_for_grpc

        if not base:
            base = os.environ.get("CLARIFAI_GRPC_BASE", None)

        if not base:
            raise ValueError("Please set 'base' via arguments or env variable CLARIFAI_GRPC_BASE")

        channel_address = "{}:{}".format(base, port)

        return grpc.aio.insecure_channel(
            channel_address,
            options=[
                ("grpc.service_config", grpc_json_config),
                ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
            ],
        )
