import os
import time
import uuid

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

DOG_IMAGE_URL = 'https://samples.clarifai.com/dog2.jpeg'
TRUCK_IMAGE_URL = "https://s3.amazonaws.com/samples.clarifai.com/red-truck.png"
NON_EXISTING_IMAGE_URL = "http://example.com/non-existing.jpg"

metadata = (('authorization', 'Key %s' % os.environ.get('CLARIFAI_API_KEY')),)


def test_post_model_outputs_on_json_channel():
  _assert_post_model_outputs(ClarifaiChannel.get_json_channel())


def test_post_model_outputs_on_grpc_channel():
  _assert_post_model_outputs(ClarifaiChannel.get_insecure_grpc_channel())


def test_failed_post_model_outputs_on_json_channel():
  _assert_failed_post_model_outputs(ClarifaiChannel.get_json_channel())


def test_failed_post_model_outputs_on_grpc_channel():
  _assert_failed_post_model_outputs(ClarifaiChannel.get_insecure_grpc_channel())


def test_mixed_success_post_model_outputs_on_json_channel():
  _assert_mixed_success_post_model_outputs(ClarifaiChannel.get_json_channel())


def test_mixed_success_post_model_outputs_on_grpc_channel():
  _assert_mixed_success_post_model_outputs(ClarifaiChannel.get_insecure_grpc_channel())


def test_post_patch_delete_input_on_json_channel():
  _assert_post_patch_delete_input(ClarifaiChannel.get_json_channel())


def test_post_patch_delete_input_on_grpc_channel():
  _assert_post_patch_delete_input(ClarifaiChannel.get_insecure_grpc_channel())


def test_list_models_with_pagination_on_json_channel():
  _assert_list_models_with_pagination(ClarifaiChannel.get_json_channel())


def test_list_models_with_pagination_on_grpc_channel():
  _assert_list_models_with_pagination(ClarifaiChannel.get_insecure_grpc_channel())


def test_multiple_requests_on_json_channel():
  _assert_multiple_requests(ClarifaiChannel.get_json_channel())


def test_multiple_requests_on_grpc_channel():
  _assert_multiple_requests(ClarifaiChannel.get_insecure_grpc_channel())


def _assert_post_model_outputs(channel):
  stub = service_pb2_grpc.V2Stub(channel)
  request = service_pb2.PostModelOutputsRequest(
    model_id='aaa03c23b3724a16a56b629203edc62c',
    inputs=[
      resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL)))
    ])
  response = stub.PostModelOutputs(request, metadata=metadata)

  _raise_on_failure(response)

  assert len(response.outputs[0].data.concepts) > 0


def _assert_failed_post_model_outputs(channel):
  stub = service_pb2_grpc.V2Stub(channel)
  request = service_pb2.PostModelOutputsRequest(
    model_id='aaa03c23b3724a16a56b629203edc62c',
    inputs=[
      resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=NON_EXISTING_IMAGE_URL)))
    ])
  response = stub.PostModelOutputs(request, metadata=metadata)

  assert response.status.code == status_code_pb2.FAILURE
  assert response.status.description == "Failure"

  assert response.outputs[0].status.code == status_code_pb2.INPUT_DOWNLOAD_FAILED


def _assert_mixed_success_post_model_outputs(channel):
  stub = service_pb2_grpc.V2Stub(channel)
  request = service_pb2.PostModelOutputsRequest(
    model_id='aaa03c23b3724a16a56b629203edc62c',
    inputs=[
      resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL))),
      resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=NON_EXISTING_IMAGE_URL)))
    ])
  response = stub.PostModelOutputs(request, metadata=metadata)

  assert response.status.code == status_code_pb2.MIXED_STATUS

  assert response.outputs[0].status.code == status_code_pb2.SUCCESS
  assert response.outputs[1].status.code == status_code_pb2.INPUT_DOWNLOAD_FAILED


def _assert_post_patch_delete_input(channel):
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
  post_response = stub.PostInputs(post_request, metadata=metadata)

  _raise_on_failure(post_response)

  input_id = post_response.inputs[0].id

  try:
    while True:
      get_request = service_pb2.GetInputRequest(input_id=input_id)
      get_response = stub.GetInput(get_request, metadata=metadata)
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
    patch_response = stub.PatchInputs(patch_request, metadata=metadata)
    assert status_code_pb2.SUCCESS == patch_response.status.code
  finally:
    delete_request = service_pb2.DeleteInputRequest(input_id=input_id)
    delete_response = stub.DeleteInput(delete_request, metadata=metadata)
    assert status_code_pb2.SUCCESS == delete_response.status.code


def _assert_list_models_with_pagination(channel):
  stub = service_pb2_grpc.V2Stub(channel)

  response = stub.ListModels(service_pb2.ListModelsRequest(per_page=2), metadata=metadata)
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception(response.status.description + " " + response.status.details)
  assert len(response.models) == 2

  # We shouldn 't have 1000*500 number of models, so the result should be empty.
  response = stub.ListModels(
    service_pb2.ListModelsRequest(page=1000, per_page=500),
    metadata=metadata
  )
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception(response.status.description + " " + response.status.details)
  assert len(response.models) == 0


def _assert_multiple_requests(channel):
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
      metadata=metadata)
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
    metadata=metadata
  )
  _raise_on_failure(post_inputs_response)

  input_ids = [i.id for i in post_inputs_response.inputs]
  _wait_for_inputs_upload(stub, metadata, input_ids)

  response = stub.PostModelVersions(
    service_pb2.PostModelVersionsRequest(model_id=model_id),
    metadata=metadata
  )
  _raise_on_failure(response)

  model_version_id = response.model.model_version.id
  _wait_for_model_trained(stub, metadata, model_id, model_version_id)

  _raise_on_failure(stub.PostModelVersionMetrics(
    service_pb2.PostModelVersionMetricsRequest(
      model_id=model_id,
      version_id=model_version_id,
    ),
    metadata=metadata
  ))

  _wait_for_model_evaluated(stub, metadata, model_id, model_version_id)

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
    metadata=metadata
  )
  _raise_on_failure(response)

  _raise_on_failure(
    stub.DeleteModel(service_pb2.DeleteModelRequest(model_id=model_id), metadata=metadata)
  )

  _raise_on_failure(
    stub.DeleteInputs(service_pb2.DeleteInputsRequest(ids=input_ids), metadata=metadata)
  )


def _wait_for_inputs_upload(stub, metadata, input_ids):
  for input_id in input_ids:
    while True:
      get_input_response = stub.GetInput(
        service_pb2.GetInputRequest(input_id=input_id),
        metadata=metadata
      )
      _raise_on_failure(get_input_response)
      if get_input_response.input.status.code == status_code_pb2.INPUT_DOWNLOAD_SUCCESS:
        break
      elif get_input_response.input.status.code in (status_code_pb2.INPUT_DOWNLOAD_PENDING,
                                                    status_code_pb2.INPUT_DOWNLOAD_IN_PROGRESS):
        time.sleep(1)
        continue
      else:
        raise Exception(
          get_input_response.status.description + " " + get_input_response.status.details
        )
  # At this point, all inputs have been downloaded successfully.


def _wait_for_model_trained(stub, metadata, model_id, model_version_id):
  while True:
    response = stub.GetModelVersion(
      service_pb2.GetModelVersionRequest(model_id=model_id, version_id=model_version_id),
      metadata=metadata
    )
    _raise_on_failure(response)
    if response.model_version.status.code == status_code_pb2.MODEL_TRAINED:
      break
    elif response.model_version.status.code in (status_code_pb2.MODEL_QUEUED_FOR_TRAINING,
                                                status_code_pb2.MODEL_TRAINING):
      time.sleep(1)
      continue
    else:
      raise Exception(
        response.status.description + " " + response.status.details
      )
  # At this point, the model has successfully finished training.


def _wait_for_model_evaluated(stub, metadata, model_id, model_version_id):
  while True:
    response = stub.GetModelVersionMetrics(
      service_pb2.GetModelVersionMetricsRequest(model_id=model_id, version_id=model_version_id),
      metadata=metadata
    )
    _raise_on_failure(response)
    if response.model_version.metrics.status.code == status_code_pb2.MODEL_EVALUATED:
      break
    elif response.model_version.metrics.status.code in (status_code_pb2.MODEL_QUEUED_FOR_EVALUATION,
                                                        status_code_pb2.MODEL_EVALUATING):
      time.sleep(1)
      continue
    else:
      raise Exception(
        response.status.description + " " + response.status.details
      )
  # At this point, the model has successfully finished evaluation.


def _raise_on_failure(response):
  if response.status.code != status_code_pb2.SUCCESS:
    print("Received failure response:")
    print(response)
    raise Exception(
      str(response.status.code) + " " + response.status.description + " " + response.status.details
    )
