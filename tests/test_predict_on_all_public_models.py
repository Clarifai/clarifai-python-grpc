from collections import defaultdict

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from tests.common import (
    BEER_VIDEO_URL,
    DOG_IMAGE_URL,
    WEATHER_AUDIO_URL,
    metadata,
    raise_on_failure,
)


def test_predict_on_all_public_models():
    stub = service_pb2_grpc.V2Stub(ClarifaiChannel.get_grpc_channel())

    PREDICT_FUNC_PER_TYPE = {
        "audio-embedder": _predict_audio_url,
        "audio-to-text": _predict_audio_url,
        "centroid-tracker": _predict_video_url,
        "embedding-classifier": _predict_image_url,
        "image-color-recognizer": _predict_image_url,
        "image-crop": _predict_image_url,
        "image-to-text": _predict_image_url,
        "optical-character-recognizer": _predict_image_url,
        "text-classifier": _predict_text,
        "text-embedder": _predict_text,
        "visual-classifier": _predict_image_url,
        "visual-detector": _predict_image_url,
        "visual-embedder": _predict_image_url,
        "visual-detector-embedder": _predict_image_url,
    }

    # Get a list of all the available models (public and private).
    all_models = []
    page = 1
    while True:
        list_models_response = stub.ListModels(
            service_pb2.ListModelsRequest(page=page, per_page=500), metadata=metadata()
        )
        raise_on_failure(list_models_response)

        all_models.extend(list_models_response.models)

        if len(list_models_response.models) < 500:
            break

        page += 1

    models_per_type = defaultdict(list)
    for model in all_models:
        # All public models have 'app_id' set to 'main'. Other models are private.
        if model.app_id == "main":
            models_per_type[model.model_type_id].append(model)

    print("\nPrediction table")
    print("-" * 120)
    print("{:30} | {:40} | {}".format("Model Type ID", "Model Name", "Model ID"))
    print("-" * 120)
    for model_type_id, models in models_per_type.items():
        if model_type_id == "clusterer":
            # It's not straightforward to do PostModelOutputs in a "clusterer" model. Probably it's safe
            # to skip it.
            continue

        predict_func = PREDICT_FUNC_PER_TYPE.get(model_type_id)
        if not predict_func:
            print(f"Warning Model type '{model_type_id}' is not tested!")
            continue

        for model in models:
            print("{:30} | {:40} | {}".format(model.model_type_id, model.name, model.id))
            response = predict_func(stub, model)
            raise_on_failure(
                response,
                f"Prediction failed on model {model.name} ({model.id}) of type {model.model_type_id}.",
            )


def _predict_audio_url(stub: service_pb2_grpc.V2Stub, model: resources_pb2.Model):
    request = service_pb2.PostModelOutputsRequest(
        model_id=model.id,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(audio=resources_pb2.Audio(url=WEATHER_AUDIO_URL))
            )
        ],
    )
    return stub.PostModelOutputs(request, metadata=metadata())


def _predict_image_url(stub: service_pb2_grpc.V2Stub, model: resources_pb2.Model):
    request = service_pb2.PostModelOutputsRequest(
        model_id=model.id,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(image=resources_pb2.Image(url=DOG_IMAGE_URL))
            )
        ],
    )
    return stub.PostModelOutputs(request, metadata=metadata())


def _predict_text(stub: service_pb2_grpc.V2Stub, model: resources_pb2.Model):
    request = service_pb2.PostModelOutputsRequest(
        model_id=model.id,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(text=resources_pb2.Text(raw="Hello, World!"))
            )
        ],
    )
    return stub.PostModelOutputs(request, metadata=metadata())


def _predict_video_url(stub: service_pb2_grpc.V2Stub, model: resources_pb2.Model):
    request = service_pb2.PostModelOutputsRequest(
        model_id=model.id,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(video=resources_pb2.Video(url=BEER_VIDEO_URL))
            )
        ],
    )
    return stub.PostModelOutputs(request, metadata=metadata())
