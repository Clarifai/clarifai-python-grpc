import time
import uuid

import pytest

from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from tests.helpers import (both_channels, _raise_on_failure, _wait_for_inputs_upload,
                           _wait_for_model_trained, _wait_for_model_evaluated, _metadata)


DOG_IMAGE_URL = 'https://samples.clarifai.com/dog2.jpeg'
TRUCK_IMAGE_URL = "https://s3.amazonaws.com/samples.clarifai.com/red-truck.png"
NON_EXISTING_IMAGE_URL = "http://example.com/non-existing.jpg"

GENERAL_MODEL_ID = 'aaa03c23b3724a16a56b629203edc62c'


@both_channels
def test_post_model_outputs(channel):
  stub = service_pb2_grpc.V2Stub(channel)
  request = service_pb2.PostModelOutputsRequest(
    model_id=GENERAL_MODEL_ID,
    inputs=[
      resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL)))
    ])
  response = stub.PostModelOutputs(request, metadata=_metadata())

  _raise_on_failure(response)

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
  response = stub.PostModelOutputs(request, metadata=_metadata())

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
  response = stub.PostModelOutputs(request, metadata=_metadata())

  assert response.status.code == status_code_pb2.MIXED_STATUS

  assert response.outputs[0].status.code == status_code_pb2.SUCCESS
  assert response.outputs[1].status.code == status_code_pb2.INPUT_DOWNLOAD_FAILED


@both_channels
def test_post_patch_delete_input(channel):
  stub = service_pb2_grpc.V2Stub(channel)

  post_request = service_pb2.PostInputsRequest(
    inputs=[
      resources_pb2.Input(
        data=resources_pb2.Data(
          image=resources_pb2.Image(
            url=TRUCK_IMAGE_URL, allow_duplicate_url=True
          ),
          concepts=[resources_pb2.Concept(id='some-concept')]
        )
      )
    ]
  )
  post_response = stub.PostInputs(post_request, metadata=_metadata())

  _raise_on_failure(post_response)

  input_id = post_response.inputs[0].id

  try:
    while True:
      get_request = service_pb2.GetInputRequest(input_id=input_id)
      get_response = stub.GetInput(get_request, metadata=_metadata())
      status_code = get_response.input.status.code
      if status_code == status_code_pb2.INPUT_DOWNLOAD_SUCCESS:
        break
      elif status_code not in (
        status_code_pb2.INPUT_DOWNLOAD_PENDING,
        status_code_pb2.INPUT_DOWNLOAD_IN_PROGRESS
      ):
        raise Exception(
          f'Waiting for input ID {input_id} failed, status code is {status_code}.')
      time.sleep(0.2)

    patch_request = service_pb2.PatchInputsRequest(
      action='overwrite',
      inputs=[
        resources_pb2.Input(
          id=input_id,
          data=resources_pb2.Data(concepts=[resources_pb2.Concept(id='some-new-concept')])
        )
      ]
    )
    patch_response = stub.PatchInputs(patch_request, metadata=_metadata())
    assert status_code_pb2.SUCCESS == patch_response.status.code
  finally:
    delete_request = service_pb2.DeleteInputRequest(input_id=input_id)
    delete_response = stub.DeleteInput(delete_request, metadata=_metadata())
    assert status_code_pb2.SUCCESS == delete_response.status.code


@both_channels
def test_list_models_with_pagination(channel):
  stub = service_pb2_grpc.V2Stub(channel)

  response = stub.ListModels(service_pb2.ListModelsRequest(per_page=2), metadata=_metadata())
  _raise_on_failure(response)
  assert len(response.models) == 2

  # We shouldn 't have 1000*500 number of models, so the result should be empty.
  response = stub.ListModels(
    service_pb2.ListModelsRequest(page=1000, per_page=500),
    metadata=_metadata()
  )
  _raise_on_failure(response)
  assert len(response.models) == 0


@pytest.mark.skip(reason="On Github Actions there's 'Model training had no data' error for some reason")
@both_channels
def test_model_creation_training_and_evaluation(channel):
  model_id = str(uuid.uuid4())

  stub = service_pb2_grpc.V2Stub(channel)

  _raise_on_failure(
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
      metadata=_metadata())
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
    metadata=_metadata()
  )
  _raise_on_failure(post_inputs_response)

  input_ids = [i.id for i in post_inputs_response.inputs]
  _wait_for_inputs_upload(stub, _metadata, input_ids)

  response = stub.PostModelVersions(
    service_pb2.PostModelVersionsRequest(model_id=model_id),
    metadata=_metadata()
  )
  _raise_on_failure(response)

  model_version_id = response.model.model_version.id
  _wait_for_model_trained(stub, _metadata, model_id, model_version_id)

  _raise_on_failure(stub.PostModelVersionMetrics(
    service_pb2.PostModelVersionMetricsRequest(
      model_id=model_id,
      version_id=model_version_id,
    ),
    metadata=_metadata()
  ))

  _wait_for_model_evaluated(stub, _metadata, model_id, model_version_id)

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
    metadata=_metadata()
  )
  _raise_on_failure(response)

  _raise_on_failure(
    stub.DeleteModel(service_pb2.DeleteModelRequest(model_id=model_id), metadata=_metadata())
  )

  _raise_on_failure(
    stub.DeleteInputs(service_pb2.DeleteInputsRequest(ids=input_ids), metadata=_metadata())
  )
