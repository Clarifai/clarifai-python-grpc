#!/usr/bin/env python

import argparse
import json
import os
import sys

try:
    from urllib.error import HTTPError
    from urllib.request import HTTPHandler, Request, build_opener
except ImportError:
    from urllib2 import HTTPError, HTTPHandler, Request, build_opener


EMAIL = os.environ["CLARIFAI_USER_EMAIL"]
PASSWORD = os.environ["CLARIFAI_USER_PASSWORD"]
GENERAL_WORKFLOW_EXTERNAL_ID = "General"


def _assert_response_success(response):
    assert "status" in response, f"Invalid response {response}"
    assert "code" in response["status"], f"Invalid response {response}"
    assert response["status"]["code"] == 10000, f"Invalid response {response}"


def _request(method, url, payload={}, headers={}):
    base_url = os.environ.get("CLARIFAI_GRPC_BASE", "api.clarifai.com")
    base_url_port = os.environ.get("CLARIFAI_GRPC_BASE_PORT", "")
    base_scheme = os.environ.get("CLARIFAI_GRPC_BASE_SCHEME", "https")

    if base_url_port != "":
        base_url = f"{base_url}:{base_url_port}"

    full_url = f"{base_scheme}://{base_url}/v2{url}"

    opener = build_opener(HTTPHandler)

    request = Request(full_url, data=json.dumps(payload).encode())
    for k in headers.keys():
        request.add_header(k, headers[k])
    request.get_method = lambda: method
    try:
        response = opener.open(request).read().decode()
    except HTTPError as e:
        error_body = e.read().decode()
        try:
            error_body = json.dumps(json.loads(error_body), indent=4)
        except:
            pass
        raise Exception(
            "ERROR after a HTTP request to: %s %s" % (method, full_url)
            + ". Response: %d %s:\n%s" % (e.code, e.reason, error_body)
        )
    return json.loads(response)


def create_app(env_name):
    session_token, user_id = _login()

    url = "/users/%s/apps" % user_id
    payload = {
        "apps": [
            {
                "name": "auto-created-in-%s-ci-test-run" % env_name,
                "default_workflow_id": GENERAL_WORKFLOW_EXTERNAL_ID,
            }
        ]
    }

    data = _request(method="POST", url=url, payload=payload, headers=_auth_headers(session_token))
    _assert_response_success(data)

    assert "apps" in data, f"Invalid response {data}"
    assert len(data["apps"]) == 1, f"Invalid response {data}"
    assert "id" in data["apps"][0], f"Invalid response {data}"
    app_id = data["apps"][0]["id"]
    assert app_id, f"Invalid response {data}"

    # This print needs to be present so we can read the value in CI.
    print(app_id)


def create_key(app_id):
    session_token, user_id = _login()

    url = "/users/%s/keys" % user_id
    payload = {
        "keys": [
            {
                "description": "Auto-created in a CI test run",
                "scopes": ["All"],
                "apps": [{"id": app_id, "user_id": user_id}],
            }
        ]
    }
    data = _request(method="POST", url=url, payload=payload, headers=_auth_headers(session_token))
    _assert_response_success(data)

    assert "keys" in data, f"Invalid response {data}"
    assert len(data["keys"]) == 1, f"Invalid response {data}"
    assert "id" in data["keys"][0], f"Invalid response {data}"
    key_id = data["keys"][0]["id"]
    assert key_id, f"Invalid response {data}"

    # This print needs to be present so we can read the value in CI.
    print(key_id)


def create_pat():
    session_token, user_id = _login()
    os.environ["CLARIFAI_USER_ID"] = user_id

    url = "/users/%s/keys" % user_id
    payload = {
        "keys": [
            {
                "description": "Auto-created in a CI test run",
                "scopes": ["All"],
                "type": "personal_access_token",
                "apps": [],
            }
        ]
    }
    data = _request(method="POST", url=url, payload=payload, headers=_auth_headers(session_token))
    _assert_response_success(data)

    assert "keys" in data, f"Invalid response {data}"
    assert len(data["keys"]) == 1, f"Invalid response {data}"
    assert "id" in data["keys"][0], f"Invalid response {data}"
    pat_id = data["keys"][0]["id"]
    assert pat_id, f"Invalid response {data}"

    # This print needs to be present so we can read the value in CI.
    print(pat_id)


def create_session_token():
    session_token, _ = _login()
    assert session_token, "Unable to generate session token"
    print(session_token)


def delete(app_id):
    session_token, user_id = _login()

    # All the related keys will be deleted automatically when the app is deleted
    _delete_app(session_token, user_id, app_id)


def create_sample_workflow(api_key):
    url = "/workflows"
    payload = {
        "workflows": [
            {
                "id": "food-and-general",
                "nodes": [
                    {
                        "id": "food-workflow-node",
                        "model": {
                            "id": "bd367be194cf45149e75f01d59f77ba7",
                            "model_version": {"id": "dfebc169854e429086aceb8368662641"},
                        },
                    },
                    {
                        "id": "general-workflow-node",
                        "model": {
                            "id": "aaa03c23b3724a16a56b629203edc62c",
                            "model_version": {"id": "aa9ca48295b37401f8af92ad1af0d91d"},
                        },
                    },
                ],
            }
        ]
    }
    response = _request(
        method="POST", url=url, payload=payload, headers=_auth_headers_for_api_key_key(api_key)
    )
    _assert_response_success(response)


def _delete_app(session_token, user_id, app_id):
    url = "/users/%s/apps/%s" % (user_id, app_id)
    response = _request(method="DELETE", url=url, headers=_auth_headers(session_token))
    _assert_response_success(response)


def _auth_headers(session_token):
    headers = {"Content-Type": "application/json", "X-Clarifai-Session-Token": session_token}
    return headers


def _auth_headers_for_api_key_key(api_key):
    headers = {"Content-Type": "application/json", "Authorization": "Key " + api_key}
    return headers


def _login():
    url = "/login"
    payload = {"email": EMAIL, "password": PASSWORD}
    data = _request(method="POST", url=url, payload=payload)
    _assert_response_success(data)

    assert "v2_user_id" in data, f"Invalid response {data}"
    user_id = data["v2_user_id"]
    assert user_id, f"Invalid response {data}"

    assert "session_token" in data, f"Invalid response {data}"
    session_token = data["session_token"]
    assert session_token, f"Invalid response {data}"

    return session_token, user_id


def run(arguments):
    if arguments.email:
        global EMAIL
        EMAIL = arguments.email  # override the default testing email
    if arguments.password:
        PASSWORD = arguments.password  # override the default testing password
    # these options are mutually exclusive
    if arguments.env_name:
        create_app(arguments.env_name)
    elif arguments.app_id_for_key:
        create_key(arguments.app_id_for_key)
    elif arguments.create_pat:
        create_pat()
    elif arguments.create_session_token:
        create_session_token()
    elif arguments.app_id_to_delete:
        delete(arguments.app_id_to_delete)
    elif arguments.wf_api_key:
        create_sample_workflow(arguments.wf_api_key)
    else:
        print(
            f"No relevant arguments specified. Run {sys.argv[0]} --help to see available options"
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Applications, Keys, and Workflows for testing."
    )
    parser.add_argument(
        "--user-email",
        dest="email",
        help="The email of the account for which the command will run. (Defaults to ${CLARIFAI_USER_EMAIL})",
    )
    parser.add_argument(
        "--user-password",
        dest="password",
        help="The password of the account for which the command will run. (Defaults to ${CLARIFAI_USER_PASSWORD})",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--create-app", dest="env_name", help="Creates a new application.")
    group.add_argument("--create-key", dest="app_id_for_key", help="Creates a new API key.")
    group.add_argument("--create-pat", action="store_true", help=" Creates a new PAT key.")
    group.add_argument(
        "--create-session-token", action="store_true", help=" Creates a new session token."
    )
    group.add_argument(
        "--delete-app",
        dest="app_id_to_delete",
        help="Deletes an application (API keys that use it are deleted as well).",
    )
    group.add_argument(
        "--create-workflow",
        dest="wf_api_key",
        help="Creates a sample workflow to be used in client tests.",
    )

    args = parser.parse_args()
    run(args)
