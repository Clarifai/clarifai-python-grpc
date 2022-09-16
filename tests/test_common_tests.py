from clarifai_grpc.grpc.api.status.status_pb2 import Status
from tests.common import get_status_message


def test_status_message_with_details():
    status = Status(code=1, description="A", details="B")
    assert get_status_message(status) == "1 A B"


def test_status_message_without_details():
    status = Status(code=1, description="A")
    assert get_status_message(status) == "1 A"
