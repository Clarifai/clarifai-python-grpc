import hashlib
import os
import requests

from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from tests.common import (
    TRUCK_IMAGE_URL,
    TRAVEL_IMAGE_URL,
    BEER_VIDEO_URL,
    both_channels,
    logger,
    raise_on_failure,
    wait_for_inputs_upload,
)


req_session = requests.Session()

DUMMY_KEY = "0" * 32
PAT = os.environ.get("CLARIFAI_PAT_KEY_SECURE_HOSTING")
API_KEY = os.environ.get("CLARIFAI_API_KEY_SECURE_HOSTING")
SESSION_TOKEN = os.environ.get("CLARIFAI_SESSION_TOKEN_SECURE_HOSTING")

PAT_CLIENT_AUTH = (("authorization", "Key %s" % PAT),)
API_CLIENT_AUTH = (("authorization", "Key %s" % API_KEY),)

HTTP_AUTH_HEADERS = {
    "session_token_header": {"X-Clarifai-Session-Token": SESSION_TOKEN},
    "api_key_header": {API_CLIENT_AUTH[0][0]: API_CLIENT_AUTH[0][1]},
    "pat_header": {PAT_CLIENT_AUTH[0][0]: PAT_CLIENT_AUTH[0][1]},
}

BAD_HTTP_AUTH_HEADERS = {
    "session_token_header": {"X-Clarifai-Session-Token": DUMMY_KEY},
    "api_key_header": {API_CLIENT_AUTH[0][0]: "Key %s" % DUMMY_KEY},
    "pat_header": {PAT_CLIENT_AUTH[0][0]: "Key %s" % DUMMY_KEY},
}

HTTP_COOKIE_HEADERS = {
    "session_token_cookie": {"x_clarifai_session_token": SESSION_TOKEN},
    "api_key_cookie": {"x-clarifai-api-key": API_KEY},
    "pat_cookie": {"x-clarifai-api-key": PAT},
}

BAD_HTTP_COOKIE_HEADERS = {
    "session_token_cookie": {"x_clarifai_session_token": DUMMY_KEY},
    "api_key_cookie": {"x-clarifai-api-key": DUMMY_KEY},
    "pat_cookie": {"x-clarifai-api-key": DUMMY_KEY},
}


def get_secure_hosting_url():
    default_secure_data_hosting_url = "https://data.clarifai.com"
    env_subdomain = os.environ.get("CLARIFAI_GRPC_BASE", "api.clarifai.com").split(".")[0]
    if env_subdomain == "api-dev":
        default_secure_data_hosting_url = "https://data-dev.clarifai.com"
    elif env_subdomain == "api-staging":
        default_secure_data_hosting_url = "https://data-staging.clarifai.com"
    url = os.environ.get("CLARIFAI_SECURE_HOSTING_URL", default_secure_data_hosting_url)
    return url


def get_bytes_hash_from_url(url):
    r = req_session.get(url, stream=True)
    return hashlib.md5(r.raw.data).hexdigest()


def get_rehost_sizes(input_type):
    if input_type == "image":
        sizes = ["orig", "small", "large", "tiny"]
    elif input_type == "video":
        sizes = ["orig", "thumbnail"]
    return sizes


def build_rehost_url_from_api_input(api_input, size, input_type):
    if input_type == "image":
        return "/".join(
            [
                api_input.data.image.hosted.prefix,
                size,
                api_input.data.image.hosted.suffix,
            ]
        )
    elif input_type == "video":
        return "/".join(
            [
                api_input.data.video.hosted.prefix,
                size,
                api_input.data.video.hosted.suffix,
            ]
        )


def verify_url_with_all_auths(expected_input_url, verify_func=None):
    for header_type, header in HTTP_AUTH_HEADERS.items():
        r = req_session.get(expected_input_url, stream=True, headers=header)
        assert (
            r.status_code == 200
        ), f"Non-200 response obtained for URL {expected_input_url}; header type: {header_type}"
        assert len(
            r.raw.data
        ), f"No data was fetched from URL {expected_input_url}; header type: {header_type}"
        if verify_func:
            assert verify_func(
                r.raw.data
            ), f"Data fetched for header type {header_type}, didn't match expected hash"
    for cookie_type, cookie in HTTP_COOKIE_HEADERS.items():
        r = req_session.get(expected_input_url, stream=True, cookies=cookie)
        assert (
            r.status_code == 200
        ), f"Non-200 response obtained for URL {expected_input_url}; cookie type: {cookie_type}"
        assert len(
            r.raw.data
        ), f"No data was fetched from URL {expected_input_url}; cookie type: {cookie_type}"
        if verify_func:
            assert verify_func(
                r.raw.data
            ), f"Data fetched for cookie type {cookie_type}, didn't match expected hash"


def verify_url_with_bad_auth(expected_input_url):
    for header_type, header in BAD_HTTP_AUTH_HEADERS.items():
        r = req_session.get(expected_input_url, stream=True, headers=header)
        assert (
            r.status_code == 401
        ), f"Expected Code: 401, Actual: {r.status_code} header type: {header_type}"
    for cookie_type, cookie in BAD_HTTP_COOKIE_HEADERS.items():
        r = req_session.get(expected_input_url, stream=True, cookies=cookie)
        assert (
            r.status_code == 401
        ), f"Expected Code: 401, Actual: {r.status_code} cookie type: {cookie_type}"
    # No header/cookie results should result in BAD REQUEST (400)
    r = req_session.get(expected_input_url, stream=True)
    assert (
        r.status_code == 400
    ), f"Expected Code: 400, Actual: {r.status_code} cookie type: {cookie_type}"


@both_channels
def test_adding_inputs(channel):
    logger.info(f"Secure Hosting URL for tests: '{get_secure_data_hosting_url()}'")
    stub = service_pb2_grpc.V2Stub(channel)

    input_img1 = "truck-img"
    input_img2 = "travel-img"
    input_vid1 = "beer-vid"

    bytes_hash_by_id = {
        input_img1: get_bytes_hash_from_url(TRUCK_IMAGE_URL),
        input_img2: get_bytes_hash_from_url(TRAVEL_IMAGE_URL),
        input_vid1: get_bytes_hash_from_url(BEER_VIDEO_URL),
    }
    media_type_by_id = {
        input_img1: "image",
        input_img2: "image",
        input_vid1: "video",
    }

    try:
        post_image_response = stub.PostInputs(
            service_pb2.PostInputsRequest(
                inputs=[
                    resources_pb2.Input(
                        id=input_img1,
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(
                                url=TRUCK_IMAGE_URL, allow_duplicate_url=True
                            ),
                        ),
                    ),
                    resources_pb2.Input(
                        id=input_img2,
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(
                                url=TRAVEL_IMAGE_URL, allow_duplicate_url=True
                            ),
                        ),
                    ),
                ]
            ),
            metadata=API_CLIENT_AUTH,
        )
        raise_on_failure(post_image_response)

        post_video_response = stub.PostInputs(  # videos must be uploaded individually
            service_pb2.PostInputsRequest(
                inputs=[
                    resources_pb2.Input(
                        id=input_vid1,
                        data=resources_pb2.Data(
                            video=resources_pb2.Video(
                                url=BEER_VIDEO_URL, allow_duplicate_url=True
                            ),
                        ),
                    )
                ]
            ),
            metadata=API_CLIENT_AUTH,
        )
        raise_on_failure(post_video_response)

        wait_for_inputs_upload(stub, API_CLIENT_AUTH, bytes_hash_by_id.keys())

        # list input
        list_inputs_response = stub.ListInputs(
            service_pb2.ListInputsRequest(per_page=10), metadata=API_CLIENT_AUTH
        )
        raise_on_failure(list_inputs_response)
        assert len(list_inputs_response.inputs) == len(bytes_hash_by_id.keys())
        # verify get and list inputs are fetchable and have the correct data
        for api_input in list_inputs_response.inputs:
            cfid = api_input.id
            get_input_response = stub.GetInput(
                service_pb2.GetInputRequest(input_id=cfid), metadata=API_CLIENT_AUTH
            )
            raise_on_failure(get_input_response)

            input_type = media_type_by_id[cfid]
            sizes = get_rehost_sizes(input_type)
            for s in sizes:
                input_url_from_list = build_rehost_url_from_api_input(api_input, s, input_type)
                input_url_from_get = build_rehost_url_from_api_input(
                    get_input_response.input, s, input_type
                )
                assert (
                    input_url_from_list == input_url_from_get
                ), "URL from Get and List calls didn't match."
                assert (
                    get_secure_hosting_url() in input_url_from_get
                ), f"'{input_url_from_get}' doesn't contain expected SDH server host URL: '{get_secure_hosting_url()}'"
                fn = (
                    (lambda byts: hashlib.md5(byts).hexdigest() == bytes_hash_by_id[cfid])
                    if s == "orig"
                    else None
                )
                verify_url_with_all_auths(input_url_from_get, verify_func=fn)
                verify_url_with_bad_auth(input_url_from_get)  # these should fail
    finally:
        # delete inputs
        for cfid in bytes_hash_by_id.keys():
            delete_request = service_pb2.DeleteInputRequest(input_id=cfid)
            delete_response = stub.DeleteInput(delete_request, metadata=API_CLIENT_AUTH)
            raise_on_failure(delete_response)
