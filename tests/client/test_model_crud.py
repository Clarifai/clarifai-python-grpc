import uuid

import pytest
from google.protobuf import struct_pb2

from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import (
    both_channels,
    metadata,
    raise_on_failure,
    wait_for_model_evaluated,
    wait_for_model_trained,
    wait_for_inputs_upload,
    TRUCK_IMAGE_URL,
    DOG_IMAGE_URL,
)


@both_channels
def test_list_all_models(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    list_response = stub.ListModels(service_pb2.ListModelsRequest(), metadata=metadata())
    raise_on_failure(list_response)
    assert len(list_response.models) > 0


@both_channels
def test_list_models_with_pagination(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    response = stub.ListModels(service_pb2.ListModelsRequest(per_page=2), metadata=metadata())
    raise_on_failure(response)
    assert len(response.models) == 2

    # We shouldn't have 1000*500 number of models, so the result should be empty.
    response = stub.ListModels(
        service_pb2.ListModelsRequest(page=1000, per_page=500), metadata=metadata()
    )
    raise_on_failure(response)
    assert len(response.models) == 0


@both_channels
def test_post_patch_get_train_evaluate_delete_model(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    # Add some inputs with the concepts that we'll need in the model.
    post_inputs_response = stub.PostInputs(
        service_pb2.PostInputsRequest(
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(url=TRUCK_IMAGE_URL, allow_duplicate_url=True),
                        concepts=[resources_pb2.Concept(id="some-initial-concept")],
                    )
                ),
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(url=DOG_IMAGE_URL, allow_duplicate_url=True),
                        concepts=[resources_pb2.Concept(id="some-new-concept")],
                    )
                ),
            ]
        ),
        metadata=metadata(),
    )
    raise_on_failure(post_inputs_response)
    input_id_1 = post_inputs_response.inputs[0].id
    input_id_2 = post_inputs_response.inputs[1].id
    wait_for_inputs_upload(stub, metadata(), [input_id_1, input_id_2])

    model_id = "model-id-" + uuid.uuid4().hex[:15]

    post_response = stub.PostModels(
        service_pb2.PostModelsRequest(
            models=[
                resources_pb2.Model(
                    id=model_id,
                    output_info=resources_pb2.OutputInfo(
                        data=resources_pb2.Data(
                            concepts=[resources_pb2.Concept(id="some-initial-concept")],
                        ),
                    ),
                )
            ]
        ),
        metadata=metadata(),
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
                        ),
                    )
                ],
            ),
            metadata=metadata(),
        )
        raise_on_failure(patch_response)

        get_response = stub.GetModelOutputInfo(
            service_pb2.GetModelRequest(model_id=model_id), metadata=metadata()
        )
        raise_on_failure(get_response)
        assert get_response.model.id == model_id
        assert get_response.model.name == "some new name"
        assert len(get_response.model.output_info.data.concepts) == 1
        assert get_response.model.output_info.data.concepts[0].id == "some-new-concept"

        post_model_versions_response = stub.PostModelVersions(
            service_pb2.PostModelVersionsRequest(model_id=model_id), metadata=metadata()
        )
        raise_on_failure(post_model_versions_response)
        model_version_id = post_model_versions_response.model.model_version.id
        wait_for_model_trained(stub, metadata(), model_id, model_version_id)

        post_model_version_metrics_response = stub.PostModelVersionMetrics(
            service_pb2.PostModelVersionMetricsRequest(
                model_id=model_id, version_id=model_version_id
            ),
            metadata=metadata(),
        )
        raise_on_failure(post_model_version_metrics_response)
        wait_for_model_evaluated(stub, metadata(), model_id, model_version_id)
    finally:
        delete_response = stub.DeleteModel(
            service_pb2.DeleteModelRequest(model_id=model_id), metadata=metadata()
        )
        raise_on_failure(delete_response)

        delete_inputs_response = stub.DeleteInputs(
            service_pb2.DeleteInputsRequest(ids=[input_id_1, input_id_2]), metadata=metadata()
        )
        raise_on_failure(delete_inputs_response)


@both_channels
def test_post_model_with_hyper_params(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    model_id = uuid.uuid4().hex[:30]

    hyper_params = struct_pb2.Struct()
    hyper_params.update(
        {
            "MAX_NITEMS": 1000000,
            "MIN_NITEMS": 1000,
            "N_EPOCHS": 5,
            "custom_training_cfg": "custom_training_1layer",
            "custom_training_cfg_args": {},
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
                        output_config=resources_pb2.OutputConfig(hyper_params=hyper_params),
                    ),
                )
            ]
        ),
        metadata=metadata(),
    )
    raise_on_failure(post_response)
    assert (
        post_response.model.output_info.output_config.hyper_params["custom_training_cfg"]
        == "custom_training_1layer"
    )

    delete_response = stub.DeleteModel(
        service_pb2.DeleteModelRequest(model_id=model_id), metadata=metadata()
    )
    raise_on_failure(delete_response)


@pytest.mark.skip(
    reason="On Github Actions there's 'Model training had no data' error for some reason"
)
@both_channels
def test_model_creation_training_and_evaluation(channel):
    model_id = str(uuid.uuid4()[:30])

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
                        ),
                    )
                ]
            ),
            metadata=metadata(),
        )
    )

    post_inputs_response = stub.PostInputs(
        service_pb2.PostInputsRequest(
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            url="https://samples.clarifai.com/dog2.jpeg",
                            allow_duplicate_url=True,
                        ),
                        concepts=[resources_pb2.Concept(id="dog")],
                    )
                ),
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            url="https://samples.clarifai.com/toddler-flowers.jpeg",
                            allow_duplicate_url=True,
                        ),
                        concepts=[resources_pb2.Concept(id="toddler")],
                    )
                ),
            ]
        ),
        metadata=metadata(),
    )
    raise_on_failure(post_inputs_response)

    input_ids = [i.id for i in post_inputs_response.inputs]
    wait_for_inputs_upload(stub, metadata, input_ids)

    response = stub.PostModelVersions(
        service_pb2.PostModelVersionsRequest(model_id=model_id), metadata=metadata()
    )
    raise_on_failure(response)

    model_version_id = response.model.model_version.id
    wait_for_model_trained(stub, metadata, model_id, model_version_id)

    raise_on_failure(
        stub.PostModelVersionMetrics(
            service_pb2.PostModelVersionMetricsRequest(
                model_id=model_id,
                version_id=model_version_id,
            ),
            metadata=metadata(),
        )
    )

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
            ),
        ),
        metadata=metadata(),
    )
    raise_on_failure(response)

    raise_on_failure(
        stub.DeleteModel(service_pb2.DeleteModelRequest(model_id=model_id), metadata=metadata())
    )

    raise_on_failure(
        stub.DeleteInputs(service_pb2.DeleteInputsRequest(ids=input_ids), metadata=metadata())
    )
