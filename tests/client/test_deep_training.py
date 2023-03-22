import uuid

from google.protobuf import struct_pb2

from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from tests.common import (
    raise_on_failure,
    metadata,
    wait_for_inputs_upload,
    wait_for_model_trained,
    post_model_outputs_and_maybe_allow_retries,
    both_channels,
)

URLS = [
    "https://samples.clarifai.com/metro-north.jpg",
    "https://samples.clarifai.com/gun.jpg",
    "https://samples.clarifai.com/logo.jpg",
    "https://samples.clarifai.com/wedding.jpg",
    "https://samples.clarifai.com/adidas_gun.jpg",
    "https://samples.clarifai.com/facebook.png",
    "https://samples.clarifai.com/dog.tiff",
    "https://samples.clarifai.com/penguin.bmp",
    "https://samples.clarifai.com/logoDarkRGB.png",
    "https://samples.clarifai.com/family.jpg",
    "https://samples.clarifai.com/cherry.webp",
]


def api_key_metadata(api_key: str):
    return (("authorization", "Key %s" % api_key),)


@both_channels
def test_deep_classification_training_with_queries(channel):
    stub = service_pb2_grpc.V2Stub(channel)
    app_id = "my-app-" + uuid.uuid4().hex[:20]
    post_apps_response = stub.PostApps(
        service_pb2.PostAppsRequest(
            user_app_id=resources_pb2.UserAppIDSet(
                user_id="me",
                app_id=app_id,
            ),
            apps=[resources_pb2.App(id=app_id, default_workflow_id="General", user_id="me")],
        ),
        metadata=metadata(pat=True),
    )
    raise_on_failure(post_apps_response)

    post_keys_response = stub.PostKeys(
        service_pb2.PostKeysRequest(
            user_app_id=resources_pb2.UserAppIDSet(
                user_id="me",
                app_id=app_id,
            ),
            keys=[
                resources_pb2.Key(
                    description="All scopes",
                    scopes=["All"],
                    apps=[resources_pb2.App(id=app_id, user_id="me")],
                )
            ],
        ),
        metadata=metadata(pat=True),
    )
    raise_on_failure(post_keys_response)
    api_key = post_keys_response.keys[0].id

    template_name = "classification_cifar10_v1"

    model_id = "my-deep-classif-" + uuid.uuid4().hex[:15]
    model_type = _get_model_type_for_template(stub, api_key, template_name)

    train_info_params = struct_pb2.Struct()
    train_info_params.update(
        {
            "template": template_name,
            "num_epochs": 2,
        }
    )

    post_models_response = stub.PostModels(
        service_pb2.PostModelsRequest(
            models=[
                resources_pb2.Model(
                    id=model_id,
                    model_type_id=model_type.id,
                    train_info=resources_pb2.TrainInfo(params=train_info_params),
                    output_info=resources_pb2.OutputInfo(
                        data=resources_pb2.Data(
                            concepts=[
                                resources_pb2.Concept(id="train-concept"),
                                resources_pb2.Concept(id="test-only-concept"),
                            ]
                        ),
                    ),
                )
            ]
        ),
        metadata=api_key_metadata(api_key),
    )
    raise_on_failure(post_models_response)

    train_and_test = ["train", "test"]
    inputs = []
    annotations = []
    for i, url in enumerate(URLS):
        input_id = str(i)
        inputs.append(
            resources_pb2.Input(
                id=input_id, data=resources_pb2.Data(image=resources_pb2.Image(url=url))
            )
        )

        train_annotation_info = struct_pb2.Struct()
        train_annotation_info.update({"split": train_and_test[i % 2]})
        ann = resources_pb2.Annotation(
            input_id=input_id,
            annotation_info=train_annotation_info,
            data=resources_pb2.Data(concepts=[resources_pb2.Concept(id="train-concept", value=1)]),
        )
        # Add an extra concept to the test set which show should up in evals, but have a bad score since there is
        # no instance of it in the train set.
        if i % 2 == 1:
            ann.data.concepts.append(resources_pb2.Concept(id="test-only-concept", value=1))
        annotations.append(ann)

    post_inputs_response = stub.PostInputs(
        service_pb2.PostInputsRequest(inputs=inputs),
        metadata=api_key_metadata(api_key),
    )
    raise_on_failure(post_inputs_response)
    wait_for_inputs_upload(stub, api_key_metadata(api_key), [str(i) for i in range(len(URLS))])

    post_annotations_response = stub.PostAnnotations(
        service_pb2.PostAnnotationsRequest(annotations=annotations),
        metadata=api_key_metadata(api_key),
    )
    raise_on_failure(post_annotations_response)

    train_annotation_info = struct_pb2.Struct()
    train_annotation_info.update({"split": "train"})
    train_query = resources_pb2.Query(
        ands=[
            resources_pb2.And(
                annotation=resources_pb2.Annotation(annotation_info=train_annotation_info)
            ),
        ]
    )

    test_annotation_info = struct_pb2.Struct()
    test_annotation_info.update({"split": "train"})
    test_query = resources_pb2.Query(
        ands=[
            resources_pb2.And(
                negate=True,
                annotation=resources_pb2.Annotation(annotation_info=test_annotation_info),
            ),
        ]
    )

    post_model_versions_response = stub.PostModelVersions(
        service_pb2.PostModelVersionsRequest(
            model_id=model_id,
            train_search=resources_pb2.Search(query=train_query),
            test_search=resources_pb2.Search(query=test_query),
        ),
        metadata=api_key_metadata(api_key),
    )
    raise_on_failure(post_model_versions_response)
    model_version_id = post_model_versions_response.model.model_version.id

    wait_for_model_trained(stub, api_key_metadata(api_key), model_id, model_version_id)

    post_model_outputs_request = service_pb2.PostModelOutputsRequest(
        model_id=model_id,
        version_id=model_version_id,
        inputs=[
            resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=URLS[0])))
        ],
    )

    post_model_outputs_response = post_model_outputs_and_maybe_allow_retries(
        stub, post_model_outputs_request, metadata=api_key_metadata(api_key)
    )
    raise_on_failure(post_model_outputs_response)

    concepts = post_model_outputs_response.outputs[0].data.concepts
    assert len(concepts) == 2
    assert concepts[0].id == "train-concept"
    assert concepts[1].id == "test-only-concept"
    assert concepts[1].value <= 0.0001

    delete_app_response = stub.DeleteApp(
        service_pb2.DeleteAppRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id="me", app_id=app_id)
        ),
        metadata=metadata(pat=True),
    )
    raise_on_failure(delete_app_response)


def _get_model_type_for_template(
    stub: service_pb2_grpc.V2Stub, api_key: str, template_name: str
) -> resources_pb2.ModelType:
    list_model_types_response = stub.ListModelTypes(
        service_pb2.ListModelTypesRequest(page=1, per_page=1000),
        metadata=api_key_metadata(api_key),
    )
    raise_on_failure(list_model_types_response)
    for model_type in list_model_types_response.model_types:
        for model_field_type in model_type.model_type_fields:
            if model_field_type.path == "train_info.params.template":
                for enum_option in model_field_type.model_type_enum_options:
                    if enum_option.id == template_name:
                        return model_type
    raise BaseException(f"Unable to get model type for template {template_name}")
