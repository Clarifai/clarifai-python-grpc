import time
import uuid

from google.protobuf import struct_pb2
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

from tests.common import (
    both_channels,
    metadata,
    post_model_outputs_and_maybe_allow_retries,
    raise_on_failure,
    MAX_RETRY_ATTEMPTS,
    TRUCK_IMAGE_URL,
    DOG_IMAGE_URL,
)

TEST_OPERATOR_CODE = """
import time
import json

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Struct

def main(req):
  outputs = []
  inputs = req.get('inputs', None)
  if not inputs:
    err_status = status_pb2.Status(code=status_code_pb2.INPUT_INVALID_ARGUMENT, description='No Inputs Received')
    err_resp = service_pb2.MultiOutputResponse(status=err_status)
    return MessageToDict(err_resp, preserving_proto_field_name=True)
  for inp in inputs:
    input_pbf = ParseDict(inp, resources_pb2.Input())
    input_id = input_pbf.id
    output = resources_pb2.Output(id=input_id)
    metadata = Struct()
    metadata.update({"processed_at": time.ctime()})
    output.data.metadata.CopyFrom(metadata)
    outputs.append(output)
    time.sleep(1)
  resp = service_pb2.MultiOutputResponse(outputs=outputs,
    status=status_pb2.Status(code=status_code_pb2.SUCCESS))
  return MessageToDict(resp, preserving_proto_field_name=True)
  """


@both_channels
def test_post_predict_delete_custom_code_operator_model(channel):
    """
    Add custom code operator model, model version, run a predict, delete CCO.
    """
    stub = service_pb2_grpc.V2Stub(channel)
    model_id = "coperator_" + uuid.uuid4().hex[:20]

    try:
        model = resources_pb2.Model(
            id=model_id,
            model_type_id="custom-code-operator",
        )
        req = service_pb2.PostModelsRequest(model=model)
        raise_on_failure(stub.PostModels(req, metadata=metadata()))

        output_info_params = struct_pb2.Struct()
        output_info_params.update({"operator_code": TEST_OPERATOR_CODE})
        output_info = resources_pb2.OutputInfo(params=output_info_params)
        req = service_pb2.PostModelVersionsRequest(
            model_id=model_id,
            model_versions=[resources_pb2.ModelVersion(output_info=output_info)],
        )
        raise_on_failure(stub.PostModelVersions(req, metadata=metadata()))

        for i in range(0, MAX_RETRY_ATTEMPTS):
            resp = stub.GetModel(
                service_pb2.GetModelRequest(
                    model_id=model_id,
                ),
                metadata=metadata(),
            )
            latest_status_code = resp.model.model_version.status.code
            if latest_status_code == status_code_pb2.MODEL_TRAINED:
                break
            time.sleep(2)

        assert latest_status_code == status_code_pb2.MODEL_TRAINED

        inputs = [
            resources_pb2.Input(
                data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL)),
            ),
            resources_pb2.Input(
                data=resources_pb2.Data(image=resources_pb2.Image(url=TRUCK_IMAGE_URL)),
            ),
        ]

        req = service_pb2.PostModelOutputsRequest(model_id=model_id, inputs=inputs)
        response = post_model_outputs_and_maybe_allow_retries(
            stub=stub, request=req, metadata=metadata()
        )

        raise_on_failure(response)
        assert len(response.outputs) == 2
        assert len(response.outputs[0].data.metadata) == 1
        assert len(response.outputs[1].data.metadata) == 1
        assert (
            response.outputs[0].data.metadata["processed_at"]
            < response.outputs[1].data.metadata["processed_at"]
        )

    finally:
        raise_on_failure(
            stub.DeleteModel(
                service_pb2.DeleteModelRequest(model_id=model_id), metadata=metadata()
            )
        )
