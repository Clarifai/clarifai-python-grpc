import io
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

        dataset_version_id = dataset_id  # Reuse dataset external ID for version.
        post_dataset_versions_response = stub.PostDatasetVersions(
            service_pb2.PostDatasetVersionsRequest(
                dataset_id=dataset_id,
                dataset_versions=[resources_pb2.DatasetVersion(id=dataset_version_id)],
            ),
            metadata=metadata(),
        )
        raise_on_failure(post_dataset_versions_response)

        wait_for_dataset_version_ready(stub, metadata(), dataset_id, dataset_version_id)

        put_dataset_version_exports_response = stub.PutDatasetVersionExports(
            service_pb2.PutDatasetVersionExportsRequest(
                dataset_id=dataset_id,
                dataset_version_id=dataset_version_id,
                exports=[
                    resources_pb2.DatasetVersionExport(
                        format=resources_pb2.CLARIFAI_DATA_PROTOBUF,
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

        export = get_dataset_version_response.dataset_version.export_info.clarifai_data_protobuf
        assert export.format == resources_pb2.CLARIFAI_DATA_PROTOBUF
        assert export.size > 0

        get_export_url_response = requests.get(export.url)
        assert get_export_url_response.status_code == 200

        with zipfile.ZipFile(io.BytesIO(get_export_url_response.content)) as zip_file:
            assert zip_file.read("mimetype") == b"application/x.clarifai-data+protobuf"

            namelist = zip_file.namelist()
            namelist.remove("mimetype")
            assert len(namelist) == 1  # All inputs in a single batch.

            input_batch = resources_pb2.InputBatch().FromString(zip_file.read(namelist[0]))
            assert len(input_batch.inputs) == len(input_ids)
    finally:
        delete_datasets_response = stub.DeleteDatasets(
            service_pb2.DeleteDatasetsRequest(dataset_ids=[dataset_id]),
            metadata=metadata(),
        )

        if input_ids:
            delete_inputs_response = stub.DeleteInputs(
                service_pb2.DeleteInputsRequest(ids=input_ids),
                metadata=metadata(),
            )
            raise_on_failure(delete_inputs_response)

        raise_on_failure(delete_datasets_response)
