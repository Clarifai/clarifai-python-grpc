import uuid

from google.protobuf import struct_pb2

from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import (both_channels, metadata, raise_on_failure, wait_for_model_evaluated,
                          wait_for_model_trained)


@both_channels
def test_list_all_models(channel):
  stub = service_pb2_grpc.V2Stub(channel)

  list_response = stub.ListModels(service_pb2.ListModelsRequest(), metadata=metadata())
  raise_on_failure(list_response)
  assert len(list_response.models) > 0


@both_channels
def test_post_patch_get_train_evaluate_delete_model(channel):
  stub = service_pb2_grpc.V2Stub(channel)

  model_id = u'我的新模型-' + uuid.uuid4().hex

  post_response = stub.PostModels(
    service_pb2.PostModelsRequest(
      models=[
        resources_pb2.Model(
          id=model_id,
          output_info=resources_pb2.OutputInfo(
            data=resources_pb2.Data(
              concepts=[resources_pb2.Concept(id="some-initial-concept")],
            ),
          )
        )
      ]
    ),
    metadata=metadata()
  )
  raise_on_failure(post_response)

  try:
    patch_response = stub.PatchModels(
      service_pb2.PatchModelsRequest(
        action="overwrite",
        models=[
          resources_pb2.Model(
            id=model_id,
            name="some new name",
            output_info=resources_pb2.OutputInfo(
              data=resources_pb2.Data(
                concepts=[resources_pb2.Concept(id="some-new-concept", value=1)]
              ),
            )
          )
        ]
      ),
      metadata=metadata()
    )
    raise_on_failure(patch_response)

    get_response = stub.GetModelOutputInfo(
      service_pb2.GetModelRequest(model_id=model_id),
      metadata=metadata()
    )
    raise_on_failure(get_response)
    assert get_response.model.id == model_id
    assert get_response.model.name == "some new name"
    assert len(get_response.model.output_info.data.concepts) == 1
    assert get_response.model.output_info.data.concepts[0].id == "some-new-concept"

    post_model_versions_response = stub.PostModelVersions(
      service_pb2.PostModelVersionsRequest(model_id=model_id),
      metadata=metadata()
    )
    raise_on_failure(post_model_versions_response)
    model_version_id = post_model_versions_response.model.model_version.id
    wait_for_model_trained(stub, metadata(), model_id, model_version_id)

    post_model_version_metrics_response = stub.PostModelVersionMetrics(
      service_pb2.PostModelVersionMetricsRequest(model_id=model_id, version_id=model_version_id),
      metadata=metadata()
    )
    raise_on_failure(post_model_version_metrics_response)
    wait_for_model_evaluated(stub, metadata(), model_id, model_version_id)
  finally:
    delete_response = stub.DeleteModel(
      service_pb2.DeleteModelRequest(model_id=model_id),
      metadata=metadata()
    )
    raise_on_failure(delete_response)


@both_channels
def test_post_model_with_hyper_params(channel):
  stub = service_pb2_grpc.V2Stub(channel)

  model_id = uuid.uuid4().hex

  hyper_params = struct_pb2.Struct()
  hyper_params.update(
    {
      'MAX_NITEMS': 1000000,
      'MIN_NITEMS': 1000,
      'N_EPOCHS': 5,
      'custom_training_cfg': 'custom_training_1layer',
      'custom_training_cfg_args': {}
    }
  )
  post_response = stub.PostModels(
    service_pb2.PostModelsRequest(
      models=[
        resources_pb2.Model(
          id=model_id,
          output_info=resources_pb2.OutputInfo(
            data=resources_pb2.Data(
              concepts=[resources_pb2.Concept(id="some-initial-concept")],
            ),
            output_config=resources_pb2.OutputConfig(
              hyper_params=hyper_params
            )
          )
        )
      ]
    ),
    metadata=metadata()
  )
  raise_on_failure(post_response)
  assert (
      post_response.model.output_info.output_config.hyper_params["custom_training_cfg"] ==
      "custom_training_1layer"
  )

  delete_response = stub.DeleteModel(
    service_pb2.DeleteModelRequest(model_id=model_id),
    metadata=metadata()
  )
  raise_on_failure(delete_response)
