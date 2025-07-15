import copy
import json
import logging
import os
import typing  # noqa

from clarifai_grpc import __version__
from clarifai_grpc.channel.errors import ApiError

CLIENT_VERSION = __version__
OS_VER = os.sys.platform
PYTHON_VERSION = ".".join(
    map(
        str,
        [
            os.sys.version_info.major,
            os.sys.version_info.minor,
            os.sys.version_info.micro,
        ],
    )
)

logger = logging.getLogger("clarifai")


class HttpClient:
    def __init__(self, session, auth_string):  # type: (requests.Session, str) -> None
        """
        :param session: The requests session object.
        :param auth_string: Either Clarifai's API key or Personal Access Token.
        """
        self._auth_string = auth_string
        self._session = session

    def execute_request(self, method, params, url):
        # type: (str, typing.Optional[dict], str) -> dict
        headers = {
            "Content-Type": "application/json",
            "X-Clarifai-gRPC-Client": "python:%s" % CLIENT_VERSION,
            "Python-Client": "%s:%s" % (OS_VER, PYTHON_VERSION),
            "X-Clarifai-Request-ID-Prefix": f"python-json-{CLIENT_VERSION}",
            "Authorization": "Key %s" % self._auth_string,
        }
        logger.debug("=" * 100)
        succinct_payload = self._mangle_base64_values(params)
        logger.debug(
            "%s %s\nHEADERS:\n%s\nPAYLOAD:\n%s",
            method,
            url,
            json.dumps(headers, indent=2),
            json.dumps(succinct_payload, indent=2),
        )
        # Avoid import at the top so we don't depend on requests in requirements.txt
        import requests  # noqa

        try:
            if method == "GET":
                res = self._session.get(
                    url, params=self._encode_get_params(params), headers=headers
                )
            elif method == "POST":
                res = self._session.post(url, data=json.dumps(params), headers=headers)
            elif method == "DELETE":
                res = self._session.delete(url, data=json.dumps(params), headers=headers)
            elif method == "PATCH":
                res = self._session.patch(url, data=json.dumps(params), headers=headers)
            elif method == "PUT":
                res = self._session.put(url, data=json.dumps(params), headers=headers)
            else:
                raise Exception("Unsupported request type: '%s'" % method)
        except requests.RequestException as e:
            raise ApiError(url, params, method, e.response)
        try:
            response_json = json.loads(res.content.decode("utf-8"))
        except ValueError:
            logger.exception("Could not get valid JSON from server response.")
            logger.debug("\nRESULT:\n%s", json.dumps(res.text, indent=2))
            error = ApiError(url, params, method, res)
            raise error
        else:
            logger.debug("\nRESULT:\n%s", json.dumps(response_json, indent=2))
        return response_json

    def _mangle_base64_values(self, params):  # type: (dict) -> dict
        """Mangle (shorten) the base64 values because they are too long for output."""
        inputs = (params or {}).get("inputs")
        query = (params or {}).get("query")
        if inputs and len(inputs) > 0:
            return self._mangle_base64_values_in_inputs(params)
        if query and query.get("ands"):
            return self._mangle_base64_values_in_query(params)
        return params

    def _mangle_base64_values_in_inputs(self, params):  # type: (dict) -> dict
        params_copy = copy.deepcopy(params)
        for input in params_copy["inputs"]:
            if "data" not in input:
                continue
            data = input["data"]
            image = data.get("image")
            if image and image.get("base64"):
                image["base64"] = self._shortened_base64_value(image["base64"])

            video = data.get("video")
            if video and video.get("base64"):
                video["base64"] = self._shortened_base64_value(video["base64"])
        return params_copy

    def _mangle_base64_values_in_query(self, params):  # type: (dict) -> dict
        params_copy = copy.deepcopy(params)
        queries = params_copy["query"]["ands"]
        for query in queries:
            image = query.get("output", {}).get("input", {}).get("data", {}).get("image", {})
            base64_val = image.get("base64")
            if base64_val:
                image["base64"] = self._shortened_base64_value(base64_val)
        return params_copy

    def _shortened_base64_value(self, original_base64):  # type: (str) -> str
        # Shorten the value if larger than what we shorten to (10 + 6 + 10).
        if len(original_base64) > 36:
            return original_base64[:10] + "......" + original_base64[-10:]
        else:
            return original_base64

    def _encode_get_params(self, params):
        """
        Encodes message params into format for use in GET args
        """
        encoded_params = {}
        for k, v in params.items():
            if isinstance(v, str):
                encoded_params[k] = v
            elif isinstance(v, bytes):
                encoded_params[k] = v.decode("utf-8")
            elif isinstance(v, (int, float, bool)):
                encoded_params[k] = str(v)
            elif isinstance(v, dict):
                for subk, subv in self._encode_get_params(v).items():
                    encoded_params[k + "." + subk] = subv
            elif isinstance(v, list):
                if v:
                    encoded_params[k] = v
            else:
                raise TypeError("Cannot convert type for get params: %s" % type(v))
        return encoded_params
