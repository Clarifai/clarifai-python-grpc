import io
import json
import requests
import uuid
import zipfile

from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from tests.common import (
    DOG_IMAGE_URL,
    TRUCK_IMAGE_URL,
    both_channels,
    metadata,
    raise_on_failure,
    wait_for_inputs_upload,
    wait_for_dataset_version_ready,
    wait_for_dataset_version_export_success,
)


@both_channels
def test_export_dataset_version(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    dataset_id = "export-" + uuid.uuid4().hex[:25]
    post_datasets_response = stub.PostDatasets(
        service_pb2.PostDatasetsRequest(
            datasets=[resources_pb2.Dataset(id=dataset_id)],
        ),
        metadata=metadata(),
    )
    raise_on_failure(post_datasets_response)

    try:
        # Declare variables for the finally-block.
        input_ids = None
        dataset_version_id = None

        post_inputs_response = stub.PostInputs(
            service_pb2.PostInputsRequest(
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(
                                url=DOG_IMAGE_URL,
                                allow_duplicate_url=True,
                            ),
                        ),
                        dataset_ids=[dataset_id],
                    ),
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(
                                url=TRUCK_IMAGE_URL,
                                allow_duplicate_url=True,
                            ),
                        ),
                        dataset_ids=[dataset_id],
                    ),
                ],
            ),
            metadata=metadata(),
        )
        raise_on_failure(post_inputs_response)

        input_ids = [input.id for input in post_inputs_response.inputs]
        wait_for_inputs_upload(stub, metadata(), input_ids)

        post_dataset_versions_response = stub.PostDatasetVersions(
            service_pb2.PostDatasetVersionsRequest(
                dataset_id=dataset_id,
                dataset_versions=[
                    # Reuse dataset external ID for version.
                    resources_pb2.DatasetVersion(id=dataset_id),
                ],
            ),
            metadata=metadata(),
        )
        raise_on_failure(post_dataset_versions_response)

        dataset_version_id = post_dataset_versions_response.dataset_versions[0].id
        wait_for_dataset_version_ready(stub, metadata(), dataset_id, dataset_version_id)

        put_dataset_version_exports_response = stub.PutDatasetVersionExports(
            service_pb2.PutDatasetVersionExportsRequest(
                dataset_id=dataset_id,
                dataset_version_id=dataset_version_id,
                exports=[
                    resources_pb2.DatasetVersionExport(
                        format=resources_pb2.CLARIFAI_DATA_PROTOBUF,
                    ),
                    resources_pb2.DatasetVersionExport(
                        format=resources_pb2.CLARIFAI_DATA_JSON,
                    ),
                ],
            ),
            metadata=metadata(),
        )
        raise_on_failure(put_dataset_version_exports_response)

        wait_for_dataset_version_export_success(stub, metadata(), dataset_id, dataset_version_id)

        get_dataset_version_response = stub.GetDatasetVersion(
            service_pb2.GetDatasetVersionRequest(
                dataset_id=dataset_id,
                dataset_version_id=dataset_version_id,
            ),
            metadata=metadata(),
        )
        raise_on_failure(get_dataset_version_response)

        export_info = get_dataset_version_response.dataset_version.export_info

        def check_protobuf(batch_str):
            input_batch = resources_pb2.InputBatch().FromString(batch_str)
            assert len(input_batch.inputs) == len(input_ids)

        _check_export(export_info.clarifai_data_protobuf, resources_pb2.CLARIFAI_DATA_PROTOBUF,
                      "application/x.clarifai-data+protobuf", check_protobuf)

        def check_json(batch_str):
            input_batch = json.loads(batch_str)
            assert len(input_batch["inputs"]) == len(input_ids)

        _check_export(export_info.clarifai_data_json, resources_pb2.CLARIFAI_DATA_JSON,
                      "application/x.clarifai-data+json", check_json)
    finally:
        if dataset_version_id:
            delete_dataset_versions_response = stub.DeleteDatasetVersions(
                service_pb2.DeleteDatasetVersionsRequest(
                    dataset_id=dataset_id,
                    dataset_version_ids=[dataset_version_id],
                ),
                metadata=metadata(),
            )

        if input_ids:
            delete_inputs_response = stub.DeleteInputs(
                service_pb2.DeleteInputsRequest(ids=input_ids),
                metadata=metadata(),
            )

        delete_datasets_response = stub.DeleteDatasets(
            service_pb2.DeleteDatasetsRequest(dataset_ids=[dataset_id]),
            metadata=metadata(),
        )

        if dataset_version_id:
            raise_on_failure(delete_dataset_versions_response)
        if input_ids:
            raise_on_failure(delete_inputs_response)
        raise_on_failure(delete_datasets_response)


def _check_export(export, expected_format, expected_mimetype, check_fn):
    assert export.format == expected_format
    assert export.size > 0

    get_export_url_response = requests.get(export.url)
    assert get_export_url_response.status_code == 200

    with zipfile.ZipFile(io.BytesIO(get_export_url_response.content)) as zip_file:
        assert zip_file.read("mimetype") == expected_mimetype.encode("ascii")

        namelist = zip_file.namelist()
        namelist.remove("mimetype")
        assert len(namelist) == 1  # All inputs in a single batch.

        check_fn(zip_file.read(namelist[0]))
