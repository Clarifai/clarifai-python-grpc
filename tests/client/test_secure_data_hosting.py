import hashlib
import os
import requests

from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from tests.common import (
    TRUCK_IMAGE_URL,
    TRAVEL_IMAGE_URL,
    BEER_VIDEO_URL,
    both_channels,
    raise_on_failure,
    wait_for_inputs_upload,
)

req_session = requests.Session()

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

HTTP_COOKIE_HEADERS = {
    "session_token_cookie": {"x_clarifai_session_token": SESSION_TOKEN},
    "api_key_cookie": {"x-clarifai-api-key": API_KEY},
    "pat_cookie": {"x-clarifai-api-key": PAT},
}


def get_secure_hosting_url():
    default_secure_data_hosting_url = "https://data.clarifai.com"
    env_subdomain = os.environ.get("CLARIFAI_GRPC_BASE", "api.clarifai.com").split(".")[0]
    if env_subdomain == "api-dev":
        default_secure_data_hosting_url = "https://data-dev.clarifai.com"
    elif env_subdomain == "api-staging":
        default_secure_data_hosting_url = "https://data-staging.clarifai.com"
    return os.environ.get("CLARIFAI_SECURE_HOSTING_URL", default_secure_data_hosting_url)


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
        return os.path.join(
            api_input.data.image.hosted.prefix,
            size,
            api_input.data.image.hosted.suffix,
        )
    elif input_type == "video":
        return os.path.join(
            api_input.data.video.hosted.prefix,
            size,
            api_input.data.video.hosted.suffix,
        )


def get_expected_input_url(input_id, app_cfid, user_cfid, size, input_type, filename):
    return f"{get_secure_hosting_url()}/{size}/users/{user_cfid}/apps/{app_cfid}/inputs/{input_type}/{filename}"


def verify_url_with_all_auths(expected_input_url):
    for header_type, header in HTTP_AUTH_HEADERS.items():
        r = req_session.get(expected_input_url, stream=True, headers=header)
        assert (
            len(r.raw.data) != 0
        ), f"No data was fetched from URL {expected_input_url}; header type: {header_type}"
    for cookie_type, cookie in HTTP_COOKIE_HEADERS.items():
        r = req_session.get(expected_input_url, stream=True, cookies=cookie)
        assert (
            len(r.raw.data) != 0
        ), f"No data was fetched from URL {expected_input_url}; cookie type: {cookie_type}"


@both_channels
def test_adding_inputs(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    input_img1 = "truck-img"
    input_img2 = "travel-img"
    input_vid1 = "beer-vid"

    bytes_data_hash_by_id = {
        input_img1: get_bytes_hash_from_url(TRUCK_IMAGE_URL),
        input_img2: get_bytes_hash_from_url(TRAVEL_IMAGE_URL),
        input_vid1: get_bytes_hash_from_url(BEER_VIDEO_URL),
    }
    media_type_by_id = {
        input_img1: "image",
        input_img2: "image",
        input_vid1: "video",
    }

    # Get app req to fetch customer-facing user id
    app_cfid = os.environ["CLARIFAI_APP_ID_SECURE_HOSTING"]
    get_app_response = stub.GetApp(
        service_pb2.GetAppRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id="me", app_id=app_cfid)
        ),
        metadata=PAT_CLIENT_AUTH,
    )
    raise_on_failure(get_app_response)
    user_cfid = get_app_response.app.user_id  # needed to build expected secure url

    post_image_response = stub.PostInputs(
        service_pb2.PostInputsRequest(
            inputs=[
                resources_pb2.Input(
                    id=input_img1,
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(url=TRUCK_IMAGE_URL, allow_duplicate_url=True),
                    ),
                ),
                resources_pb2.Input(
                    id=input_img2,
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(url=TRAVEL_IMAGE_URL, allow_duplicate_url=True),
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
                        video=resources_pb2.Video(url=BEER_VIDEO_URL, allow_duplicate_url=True),
                    ),
                )
            ]
        ),
        metadata=API_CLIENT_AUTH,
    )
    raise_on_failure(post_video_response)

    wait_for_inputs_upload(stub, API_CLIENT_AUTH, bytes_data_hash_by_id.keys())

    # list input
    list_inputs_response = stub.ListInputs(
        service_pb2.ListInputsRequest(per_page=10), metadata=API_CLIENT_AUTH
    )
    raise_on_failure(list_inputs_response)
    assert len(list_inputs_response.inputs) == len(bytes_data_hash_by_id.keys())
    # verify get and list inputs are fetchable and have the correct data
    for api_input in list_inputs_response.inputs:
        input_id = api_input.id
        get_input_response = stub.GetInput(
            service_pb2.GetInputRequest(input_id=input_id), metadata=API_CLIENT_AUTH
        )
        raise_on_failure(get_input_response)

        input_type = media_type_by_id[input_id]
        sizes = get_rehost_sizes(input_type)
        for size in sizes:
            expected_input_url = get_expected_input_url(
                input_id, app_cfid, user_cfid, size, input_type, bytes_data_hash_by_id[input_id]
            )
            input_url_from_list = build_rehost_url_from_api_input(api_input, size, input_type)
            assert expected_input_url == input_url_from_list, "URL from List Inputs didnt match"
            input_url_from_get = build_rehost_url_from_api_input(
                get_input_response.input, size, input_type
            )
            assert expected_input_url == input_url_from_get, "URL fron Get Input didnt match"
            verify_url_with_all_auths(expected_input_url)

    # delete inputs
    for inp in bytes_data_hash_by_id.keys():
        delete_request = service_pb2.DeleteInputRequest(input_id=inp)
        delete_response = stub.DeleteInput(delete_request, metadata=API_CLIENT_AUTH)
        raise_on_failure(delete_response)
