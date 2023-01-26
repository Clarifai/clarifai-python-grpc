import hashlib
import os
import requests

from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from tests.common import (
    TRUCK_IMAGE_URL,
    TRAVEL_IMAGE_URL,
    BEER_VIDEO_URL,
    both_channels,
    # metadata,
    # secure_data_hosting_metadata, # make this a dict with all the headers we want to test
    raise_on_failure,
    wait_for_inputs_upload,
)

########
# TODO: make sure we use different authentication methods
def metadata(pat=False):
    if pat:
        return (("authorization", "Key %s" % os.environ.get("CLARIFAI_PAT_KEY_SECURE_HOSTING")),)
    else:
        return (("authorization", "Key %s" % os.environ.get("CLARIFAI_API_KEY_SECURE_HOSTING")),)


def get_bytes_hash(url):
    r = requests.get(url, stream=True)
    return hashlib.md5(r.raw.data).hexdigest()


default_secure_data_hosting_url = "https//data.clarifai.com"
env_subdomain = os.environ.get("CLARIFAI_GRPC_BASE", "api.clarifai.com").split(".")[0]
if env_subdomain == "api-dev":
    default_secure_data_hosting_url = "https//data-dev.clarifai.com"
elif env_subdomain == "api-staging":
    default_secure_data_hosting_url = "https//data-staging.clarifai.com"
secure_data_hosting_url = os.environ.get(
    "CLARIFAI_SECURE_HOSTING_URL", default_secure_data_hosting_url
)
########


@both_channels
def test_adding_inputs(channel):

    input_img1 = "truck-img"
    input_img2 = "travel-img"
    input_vid1 = "beer-vid"

    bytes_data_hashes = {
        input_img1: get_bytes_hash(TRUCK_IMAGE_URL),
        input_img2: get_bytes_hash(TRAVEL_IMAGE_URL),
        input_vid1: get_bytes_hash(BEER_VIDEO_URL),
    }

    inputs = [input_img1, input_img2, input_vid1]

    stub = service_pb2_grpc.V2Stub(channel)

    # Get app req to fetch custom-facing user/app ids
    app_cfid = os.environ["CLARIFAI_APP_ID_SECURE_HOSTING"]
    get_app_response = stub.GetApp(
        service_pb2.GetAppRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id="me", app_id=app_cfid)
        ),
        metadata=metadata(pat=True),
    )
    raise_on_failure(get_app_response)

    # needed to build secure hosting url
    user_cfid = get_app_response.app.user_id  # get user cfid

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
        metadata=metadata(),
    )
    raise_on_failure(post_image_response)

    post_video_response = stub.PostInputs(
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
        metadata=metadata(),
    )
    raise_on_failure(post_video_response)

    try:
        wait_for_inputs_upload(stub, metadata(), inputs)

        for inp in inputs:
            get_input_response = stub.GetInput(
                service_pb2.GetInputRequest(input_id=inp), metadata=metadata()
            )
            raise_on_failure(get_input_response)

            input_type = "image" if "img" in inp else "video"
            if "img" in inp:
                sizes = ["orig", "small", "large", "tiny"]
            elif "vid" in inp:
                sizes = ["orig", "thumbnail"]
            for size in sizes:
                expected_input_url = get_expected_input_url(
                    inp, app_cfid, user_cfid, size, input_type, bytes_data_hashes[inp]
                )
                input_url = os.path.join(
                    get_input_response.input.data.image.hosted.prefix,
                    size,
                    get_input_response.input.data.image.hosted.suffix,
                )
        assert expected_input_url == input_url, "URLs didnt match"

        list_inputs_response = stub.ListInputs(
            service_pb2.ListInputsRequest(per_page=1), metadata=metadata()
        )
        raise_on_failure(list_inputs_response)
        assert len(list_inputs_response.inputs) == len(inputs)
    finally:
        for inp in inputs:
            delete_request = service_pb2.DeleteInputRequest(input_id=inp)
            delete_response = stub.DeleteInput(delete_request, metadata=metadata())
            raise_on_failure(delete_response)


def get_expected_input_urls(input_id, app_cfid, user_cfid, size, input_type, filename):
    expected_input_url = "%s/%s/users/%s/apps/%s/inputs/%s/%s" % (
        secure_data_hosting_url,
        size,
        user_cfid,
        app_cfid,
        input_type,
        filename,
    )
    return expected_input_url

    # TODO: Get password of current test user (and reuse it?) -- done
    # TODO: Create User in dev/staging/prod (delete suffixed emails) -- done
    # TODO: Use admin API to enable secure data hosting -- done
    # TODO: Add github secret for CLARIFAI_USER_EMAIL_SECURE_HOSTING to add to ~/work/clarifai-python-grpc/.github/workflows/run_tests.yml -- done
    # TODO: Update secure data hosting version of CLARIFAI_APP_ID, CLARIFAI_API_KEY, CLARIFAI_PAT_KEY in -- done
    # write tests:
    # POST inputs to the app
    # list inputs for the app
    # get individual inputs
    # fetch inputs with requests library

    # TODO: Add new secret for CLARIFAI_USER_EMAIL_SECURE_HOSTING in ~/work/crons-api-client-tests/.github/workflows/grpc-python-client-test.yml
    #         both run_tests.yml and grpc-python-client-test.yml
