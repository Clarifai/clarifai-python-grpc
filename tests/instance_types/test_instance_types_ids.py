from typing import Dict, List

import pytest
import requests

from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from tests.common import get_channel, grpc_channel, metadata


def fetch_csv_from_github(cloud_provider: str, catalog_version: str = "v7") -> str:
    """
    Fetch CSV content directly from GitHub raw URLs instead of cloning the repository.

    Args:
        cloud_provider: Cloud provider name (aws, gcp, azure, etc.)
        catalog_version: Catalog version (default: v7)

    Returns:
        CSV content as string

    Raises:
        requests.RequestException: If the HTTP request fails
    """
    base_url = "https://raw.githubusercontent.com/skypilot-org/skypilot-catalog/master"
    csv_url = f"{base_url}/catalogs/{catalog_version}/{cloud_provider.lower()}/vms.csv"

    print(f"Fetching CSV from: {csv_url}")
    response = requests.get(csv_url, timeout=30)
    response.raise_for_status()

    return response.text


def get_instance_types_simple(csv_content: str) -> List[str]:
    """Parse CSV content and extract instance types from the first column."""
    instance_types = set()

    # Split content into lines and skip header
    lines = csv_content.strip().split('\n')[1:]

    for line in lines:
        if line.strip():
            instance_type = line.split(',')[0]
            if instance_type:
                instance_types.add(instance_type)

    return sorted(list(instance_types))


def fetch_skypilot_instance_types(cloud_providers=None):
    """
    Fetch instance types from skypilot-catalog repository via direct HTTP requests.
    This function:
    1. Makes direct HTTP requests to GitHub raw URLs for vms.csv files
    2. Parses the CSV content to extract instance type IDs
    3. Returns a set of expected instance type IDs
    """
    # If no cloud providers specified, use default ones
    if cloud_providers is None:
        cloud_providers = ['aws', 'gcp', 'azure']

    all_instance_types = set()

    for provider in cloud_providers:
        provider_lower = provider.lower()

        try:
            csv_content = fetch_csv_from_github(provider_lower)
            provider_instance_types = get_instance_types_simple(csv_content)
            all_instance_types.update(provider_instance_types)
            print(f"  Found {len(provider_instance_types)} instance types for {provider_lower}")
        except requests.RequestException as e:
            print(f"Warning: Could not fetch CSV for {provider_lower}: {e}")
        except Exception as e:
            print(f"Warning: Error processing CSV for {provider_lower}: {e}")

    if not all_instance_types:
        raise RuntimeError("No vms.csv files could be fetched for any cloud provider")

    print(
        f"Successfully fetched {len(all_instance_types)} total instance types from skypilot-catalog"
    )
    return all_instance_types


def fetch_skypilot_instance_types_by_provider(cloud_provider_id):
    """
    Fetch instance types for a specific cloud provider from skypilot-catalog.
    Maps Clarifai cloud provider IDs to skypilot-catalog directory names.
    """
    try:
        # Map Clarifai cloud provider IDs to skypilot-catalog directory names
        provider_mapping = {
            'aws': 'aws',
            'gcp': 'gcp',
            'azure': 'azure',
            'local': 'local',  # if local is supported
        }

        provider_lower = provider_mapping.get(cloud_provider_id.lower(), cloud_provider_id.lower())

        csv_content = fetch_csv_from_github(provider_lower)
        instance_types = get_instance_types_simple(csv_content)
        print(f"  Found {len(instance_types)} instance types for {cloud_provider_id}")
        return set(instance_types)

    except requests.RequestException as e:
        print(f"Warning: Could not fetch CSV for {cloud_provider_id}: {e}")
        return set()
    except Exception as e:
        print(f"Warning: Error processing CSV for {cloud_provider_id}: {e}")
        return set()


def get_cloud_providers(stub, metadata_tuple) -> List[resources_pb2.CloudProvider]:
    """Get all available cloud providers."""
    try:
        response = stub.ListCloudProviders(
            service_pb2.ListCloudProvidersRequest(), metadata=metadata_tuple
        )

        if response.status.code != status_code_pb2.StatusCode.SUCCESS:
            pytest.fail(f"Failed to list cloud providers: {response.status.description}")

        return response.cloud_providers
    except Exception as e:
        pytest.fail(f"Error listing cloud providers: {e}")


def get_cloud_regions(
    stub, metadata_tuple, cloud_provider: resources_pb2.CloudProvider
) -> List[str]:
    """Get all regions for a specific cloud provider."""
    try:
        request = service_pb2.ListCloudRegionsRequest(
            cloud_provider=resources_pb2.CloudProvider(id=cloud_provider.id)
        )
        response = stub.ListCloudRegions(request, metadata=metadata_tuple)

        if response.status.code != status_code_pb2.StatusCode.SUCCESS:
            pytest.fail(
                f"Failed to list regions for {cloud_provider.id}: {response.status.description}"
            )

        # API returns regions as a list of region id strings
        return list(response.regions)
    except Exception as e:
        pytest.fail(f"Error listing regions for {cloud_provider.id}: {e}")


def get_instance_types(
    stub, metadata_tuple, cloud_provider: resources_pb2.CloudProvider, region: str
) -> List[str]:
    """Get all instance types for a specific cloud provider and region."""
    try:
        request = service_pb2.ListInstanceTypesRequest(
            cloud_provider=resources_pb2.CloudProvider(id=cloud_provider.id), region=region
        )
        response = stub.ListInstanceTypes(request, metadata=metadata_tuple)

        if response.status.code != status_code_pb2.StatusCode.SUCCESS:
            pytest.fail(
                f"Failed to list instance types for {cloud_provider.id}/{region}: {response.status.description}"
            )

        return [instance_type.id for instance_type in response.instance_types]
    except Exception as e:
        pytest.fail(f"Error listing instance types for {cloud_provider.id}/{region}: {e}")


def collect_all_instance_types(stub, metadata_tuple) -> Dict[str, Dict[str, List[str]]]:
    """
    Collect all instance types across all cloud providers and regions.
    Returns: {cloud_provider_id: {region: [instance_type_ids]}}
    """
    all_instance_types = {}

    # Get all cloud providers
    cloud_providers = get_cloud_providers(stub, metadata_tuple)

    for cloud_provider in cloud_providers:
        all_instance_types[cloud_provider.id] = {}

        # Get all regions for this cloud provider
        regions = get_cloud_regions(stub, metadata_tuple, cloud_provider)

        for region in regions:
            # Get all instance types for this region
            instance_types = get_instance_types(stub, metadata_tuple, cloud_provider, region)
            all_instance_types[cloud_provider.id][region] = instance_types

    return all_instance_types


UNSUPPORTED_SKYCATALOG_PROVIDERS = {"vultr", "oracle", "lambda"}


def is_provider_supported(provider_id: str) -> bool:
    return provider_id.lower() not in UNSUPPORTED_SKYCATALOG_PROVIDERS


@grpc_channel()
def test_instance_types_exist_and_not_deprecated(channel_key):
    """
    Test that all instance types returned by the API exist and are not deprecated.
    This test:
    1. Gets all cloud providers
    2. Gets all regions for each cloud provider
    3. Gets all instance types for each region
    4. Compares with expected instance types from skypilot-catalog (provider-specific)
    5. Raises errors for missing or deprecated instance types
    """
    stub = service_pb2_grpc.V2Stub(get_channel(channel_key))
    metadata_tuple = metadata(pat=True)

    # Collect all instance types from the API
    api_instance_types = collect_all_instance_types(stub, metadata_tuple)

    # Flatten all API instance types for comparison (only supported providers)
    all_api_instance_types = set()
    provider_instance_types = {}

    for cloud_provider_id, regions in api_instance_types.items():
        provider_instance_types[cloud_provider_id] = set()
        for _, instance_types in regions.items():
            provider_instance_types[cloud_provider_id].update(instance_types)
        if is_provider_supported(cloud_provider_id):
            all_api_instance_types.update(provider_instance_types[cloud_provider_id])

    # Get expected instance types for each supported cloud provider
    all_expected_instance_types = set()
    provider_expected_types = {}

    for cloud_provider_id in api_instance_types.keys():
        if not is_provider_supported(cloud_provider_id):
            continue
        expected_types = fetch_skypilot_instance_types_by_provider(cloud_provider_id)
        provider_expected_types[cloud_provider_id] = expected_types
        all_expected_instance_types.update(expected_types)

    # Check for missing instance types (API returns types not in skypilot-catalog)
    missing_in_skypilot = all_api_instance_types - all_expected_instance_types
    if missing_in_skypilot:
        pytest.fail(
            f"Found {len(missing_in_skypilot)} instance types in API that are not in skypilot-catalog: "
            f"{sorted(missing_in_skypilot)}"
        )

    # Log summary for debugging
    print("\nInstance Types Summary:")
    print(f"Total API instance types (supported providers): {len(all_api_instance_types)}")
    print(
        f"Cloud providers checked: {[p for p in api_instance_types.keys() if is_provider_supported(p)]}"
    )

    for cloud_provider_id, regions in api_instance_types.items():
        total_for_provider = sum(len(instance_types) for instance_types in regions.values())
        if is_provider_supported(cloud_provider_id):
            expected_for_provider = len(provider_expected_types.get(cloud_provider_id, set()))
            print(
                f"  {cloud_provider_id}: {total_for_provider} instance types across {len(regions)} regions (expected: {expected_for_provider})"
            )
        else:
            print(f"  {cloud_provider_id}: skipped (unsupported by skypilot-catalog)")

    # Assert that we have a reasonable number of instance types among supported providers
    assert len(all_api_instance_types) > 0, (
        "No instance types found in API for supported providers"
    )
