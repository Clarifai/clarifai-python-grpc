import uuid

import pytest

from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import (both_channels, metadata, raise_on_failure, wait_for_inputs_upload,
                          wait_for_model_evaluated, wait_for_model_trained)


@both_channels
def test_list_models_with_pagination(channel):
  stub = service_pb2_grpc.V2Stub(channel)

  response = stub.ListModels(service_pb2.ListModelsRequest(per_page=2), metadata=metadata())
  raise_on_failure(response)
  assert len(response.models) == 2

  # We shouldn 't have 1000*500 number of models, so the result should be empty.
  response = stub.ListModels(service_pb2.ListModelsRequest(page=1000, per_page=500),
                             metadata=metadata())
  raise_on_failure(response)
  assert len(response.models) == 0


@pytest.mark.skip(
    reason="On Github Actions there's 'Model training had no data' error for some reason")
@both_channels
def test_model_creation_training_and_evaluation(channel):
  model_id = str(uuid.uuid4())

  stub = service_pb2_grpc.V2Stub(channel)

  raise_on_failure(
      stub.PostModels(service_pb2.PostModelsRequest(models=[
          resources_pb2.Model(id=model_id,
                              output_info=resources_pb2.OutputInfo(data=resources_pb2.Data(
                                  concepts=[
                                      resources_pb2.Concept(id="dog"),
                                      resources_pb2.Concept(id="toddler"),
                                  ])))
      ]),
                      metadata=metadata()))

  post_inputs_response = stub.PostInputs(service_pb2.PostInputsRequest(inputs=[
      resources_pb2.Input(data=resources_pb2.Data(
          image=resources_pb2.Image(url="https://samples.clarifai.com/dog2.jpeg",
                                    allow_duplicate_url=True),
          concepts=[resources_pb2.Concept(id="dog")],
      )),
      resources_pb2.Input(data=resources_pb2.Data(
          image=resources_pb2.Image(url="https://samples.clarifai.com/toddler-flowers.jpeg",
                                    allow_duplicate_url=True),
          concepts=[resources_pb2.Concept(id="toddler")],
      )),
  ]),
                                         metadata=metadata())
  raise_on_failure(post_inputs_response)

  input_ids = [i.id for i in post_inputs_response.inputs]
  wait_for_inputs_upload(stub, metadata, input_ids)

  response = stub.PostModelVersions(service_pb2.PostModelVersionsRequest(model_id=model_id),
                                    metadata=metadata())
  raise_on_failure(response)

  model_version_id = response.model.model_version.id
  wait_for_model_trained(stub, metadata, model_id, model_version_id)

  raise_on_failure(
      stub.PostModelVersionMetrics(service_pb2.PostModelVersionMetricsRequest(
          model_id=model_id,
          version_id=model_version_id,
      ),
                                   metadata=metadata()))

  wait_for_model_evaluated(stub, metadata, model_id, model_version_id)

  response = stub.GetModelVersionMetrics(service_pb2.GetModelVersionMetricsRequest(
      model_id=model_id,
      version_id=model_version_id,
      fields=resources_pb2.FieldsValue(
          confusion_matrix=True,
          cooccurrence_matrix=True,
          label_counts=True,
          binary_metrics=True,
          test_set=True,
      )),
                                         metadata=metadata())
  raise_on_failure(response)

  raise_on_failure(
      stub.DeleteModel(service_pb2.DeleteModelRequest(model_id=model_id), metadata=metadata()))

  raise_on_failure(
      stub.DeleteInputs(service_pb2.DeleteInputsRequest(ids=input_ids), metadata=metadata()))
