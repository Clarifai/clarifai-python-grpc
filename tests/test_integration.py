import time
import uuid

import pytest

from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from tests.helpers import (both_channels, raise_on_failure, wait_for_inputs_upload,
                           wait_for_model_trained, wait_for_model_evaluated, metadata, GENERAL_MODEL_ID,
                           DOG_IMAGE_URL, NON_EXISTING_IMAGE_URL)


@both_channels
def test_post_model_outputs(channel):
  stub = service_pb2_grpc.V2Stub(channel)
  request = service_pb2.PostModelOutputsRequest(
    model_id=GENERAL_MODEL_ID,
    inputs=[
      resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL)))
    ])
  response = stub.PostModelOutputs(request, metadata=metadata())

  raise_on_failure(response)

  assert len(response.outputs[0].data.concepts) > 0


@both_channels
def test_failed_post_model_outputs(channel):
  stub = service_pb2_grpc.V2Stub(channel)
  request = service_pb2.PostModelOutputsRequest(
    model_id=GENERAL_MODEL_ID,
    inputs=[
      resources_pb2.Input(
        data=resources_pb2.Data(image=resources_pb2.Image(url=NON_EXISTING_IMAGE_URL))
      )
    ])
  response = stub.PostModelOutputs(request, metadata=metadata())

  assert response.status.code == status_code_pb2.FAILURE
  assert response.status.description == "Failure"

  assert response.outputs[0].status.code == status_code_pb2.INPUT_DOWNLOAD_FAILED


@both_channels
def test_mixed_success_post_model_outputs(channel):
  stub = service_pb2_grpc.V2Stub(channel)
  request = service_pb2.PostModelOutputsRequest(
    model_id=GENERAL_MODEL_ID,
    inputs=[
      resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL))),
      resources_pb2.Input(
        data=resources_pb2.Data(image=resources_pb2.Image(url=NON_EXISTING_IMAGE_URL))
      )
    ])
  response = stub.PostModelOutputs(request, metadata=metadata())

  assert response.status.code == status_code_pb2.MIXED_STATUS

  assert response.outputs[0].status.code == status_code_pb2.SUCCESS
  assert response.outputs[1].status.code == status_code_pb2.INPUT_DOWNLOAD_FAILED


@both_channels
def test_list_models_with_pagination(channel):
  stub = service_pb2_grpc.V2Stub(channel)

  response = stub.ListModels(service_pb2.ListModelsRequest(per_page=2), metadata=metadata())
  raise_on_failure(response)
  assert len(response.models) == 2

  # We shouldn 't have 1000*500 number of models, so the result should be empty.
  response = stub.ListModels(
    service_pb2.ListModelsRequest(page=1000, per_page=500),
    metadata=metadata()
  )
  raise_on_failure(response)
  assert len(response.models) == 0


@pytest.mark.skip(reason="On Github Actions there's 'Model training had no data' error for some reason")
@both_channels
def test_model_creation_training_and_evaluation(channel):
  model_id = str(uuid.uuid4())

  stub = service_pb2_grpc.V2Stub(channel)

  raise_on_failure(
    stub.PostModels(
      service_pb2.PostModelsRequest(
        models=[
          resources_pb2.Model(
            id=model_id,
            output_info=resources_pb2.OutputInfo(
              data=resources_pb2.Data(
                concepts=[
                  resources_pb2.Concept(id="dog"),
                  resources_pb2.Concept(id="toddler"),
                ]
              )
            )
          )
        ]
      ),
      metadata=metadata())
  )

  post_inputs_response = stub.PostInputs(
    service_pb2.PostInputsRequest(
      inputs=[
        resources_pb2.Input(
          data=resources_pb2.Data(
            image=resources_pb2.Image(
              url="https://samples.clarifai.com/dog2.jpeg",
              allow_duplicate_url=True
            ),
            concepts=[resources_pb2.Concept(id="dog")],
          )
        ),
        resources_pb2.Input(
          data=resources_pb2.Data(
            image=resources_pb2.Image(
              url="https://samples.clarifai.com/toddler-flowers.jpeg",
              allow_duplicate_url=True
            ),
            concepts=[resources_pb2.Concept(id="toddler")],
          )
        ),
      ]
    ),
    metadata=metadata()
  )
  raise_on_failure(post_inputs_response)

  input_ids = [i.id for i in post_inputs_response.inputs]
  wait_for_inputs_upload(stub, metadata, input_ids)

  response = stub.PostModelVersions(
    service_pb2.PostModelVersionsRequest(model_id=model_id),
    metadata=metadata()
  )
  raise_on_failure(response)

  model_version_id = response.model.model_version.id
  wait_for_model_trained(stub, metadata, model_id, model_version_id)

  raise_on_failure(stub.PostModelVersionMetrics(
    service_pb2.PostModelVersionMetricsRequest(
      model_id=model_id,
      version_id=model_version_id,
    ),
    metadata=metadata()
  ))

  wait_for_model_evaluated(stub, metadata, model_id, model_version_id)

  response = stub.GetModelVersionMetrics(
    service_pb2.GetModelVersionMetricsRequest(
      model_id=model_id,
      version_id=model_version_id,
      fields=resources_pb2.FieldsValue(
        confusion_matrix=True,
        cooccurrence_matrix=True,
        label_counts=True,
        binary_metrics=True,
        test_set=True,
      )
    ),
    metadata=metadata()
  )
  raise_on_failure(response)

  raise_on_failure(
    stub.DeleteModel(service_pb2.DeleteModelRequest(model_id=model_id), metadata=metadata())
  )

  raise_on_failure(
    stub.DeleteInputs(service_pb2.DeleteInputsRequest(ids=input_ids), metadata=metadata())
  )
