import os
from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from tests.common import (
    ARCHIVE_CLOUD_URL,
    CLOUD_URL,
    raise_on_failure,
    wait_for_extraction_job_completed,
    both_channels,
    metadata,
)


@both_channels
def test_post_inputs_data_source_single_public_archive(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    post_data_source_response = stub.PostInputsDataSources(
        service_pb2.PostInputsDataSourcesRequest(
            data_sources=[
                resources_pb2.InputsDataSource(
                    url=resources_pb2.DataSourceURL(url=ARCHIVE_CLOUD_URL)
                )
            ],
            app_pat=os.environ.get("CLARIFAI_API_KEY"),
        ),
        metadata=metadata(),
    )
    raise_on_failure(post_data_source_response)

    completed_response = wait_for_extraction_job_completed(
        stub, post_data_source_response.inputs_add_jobs[0].extraction_jobs[0].id
    )

    assert completed_response.inputs_extraction_job.progress.image_inputs_count == 3
    assert completed_response.inputs_extraction_job.progress.video_inputs_count == 1


@both_channels
def test_post_inputs_data_source_public_cloud_directory(channel):
    stub = service_pb2_grpc.V2Stub(channel)

    post_data_source_response = stub.PostInputsDataSources(
        service_pb2.PostInputsDataSourcesRequest(
            data_sources=[
                resources_pb2.InputsDataSource(url=resources_pb2.DataSourceURL(url=CLOUD_URL))
            ],
            app_pat=os.environ.get("CLARIFAI_API_KEY"),
        ),
        metadata=metadata(),
    )
    raise_on_failure(post_data_source_response)

    completed_response = wait_for_extraction_job_completed(
        stub, post_data_source_response.inputs_add_jobs[0].extraction_jobs[0].id
    )

    assert completed_response.inputs_extraction_job.progress.image_inputs_count == 3
    assert completed_response.inputs_extraction_job.progress.video_inputs_count == 1
