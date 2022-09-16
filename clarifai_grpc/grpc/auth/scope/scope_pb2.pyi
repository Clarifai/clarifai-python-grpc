"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.descriptor_pb2
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.internal.extension_dict
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _S:
    ValueType = typing.NewType('ValueType', builtins.int)
    V: typing_extensions.TypeAlias = ValueType
class _SEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_S.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    undef: _S.ValueType  # 0
    """introduce undef so that the zero (default/unset) value of the enum is not a real
    permission.  undef is only present for this purpose and should not be used
    to indicate any "real" value.
    """

    All: _S.ValueType  # 1
    Predict: _S.ValueType  # 2
    """Make an rpc to our prediction services."""

    Inputs_Add: _S.ValueType  # 4
    """Write to the inputs table in the DB."""

    Inputs_Get: _S.ValueType  # 5
    """Read from the inputs table in the DB."""

    Inputs_Patch: _S.ValueType  # 7
    """To patch we need read/write.
    Deprecated.
    Optionally needs Concepts_Add.
    """

    Inputs_Delete: _S.ValueType  # 8
    """To delete we need read/write"""

    Outputs_Patch: _S.ValueType  # 9
    """Deprecated."""

    Concepts_Add: _S.ValueType  # 10
    """Write to the concepts DB tables."""

    Concepts_Get: _S.ValueType  # 11
    """Read from the concepts DB tables."""

    Concepts_Patch: _S.ValueType  # 12
    """TODO: No concept searches scope.

    To patch we need read/write.
    Deprecated
    """

    Concepts_Delete: _S.ValueType  # 13
    """To delete we need read/write.
    Note: not implemented.
    """

    Models_Add: _S.ValueType  # 14
    """Write to the models DB tables."""

    Models_Get: _S.ValueType  # 15
    """Read from the models and models versions DB tables."""

    Models_Patch: _S.ValueType  # 16
    """To patch we need read/write.
    Deprecated.
    """

    Models_Delete: _S.ValueType  # 17
    """To delete we need read/write."""

    Models_Train: _S.ValueType  # 26
    """Note: Models_Train is effectively doing POST /models/{models_id}/versions so it's treated that
    way in terms of data access under the hood.
    Write to the model versions DB table.
    """

    Models_Sync: _S.ValueType  # 27
    """Internal only model syncing."""

    Workflows_Add: _S.ValueType  # 18
    """TODO: No model get metrics scope.

    Write to the workflows DB table.
    """

    Workflows_Get: _S.ValueType  # 19
    """Read from the workflows DB table."""

    Workflows_Patch: _S.ValueType  # 20
    """To patch we need read/write.
    Deprecated.
    """

    Workflows_Delete: _S.ValueType  # 21
    """To delete we need read/write."""

    WorkflowMetrics_Get: _S.ValueType  # 96
    WorkflowMetrics_Add: _S.ValueType  # 97
    WorkflowMetrics_Delete: _S.ValueType  # 98
    TSNEVisualizations_Add: _S.ValueType  # 24
    """Write to the visualizations DB table.
    Deprecated
    """

    TSNEVisualizations_Get: _S.ValueType  # 25
    """Read from the visualizations DB table.
    Deprecated
    """

    Annotations_Add: _S.ValueType  # 37
    """Write to the annotations DB table."""

    Annotations_Get: _S.ValueType  # 38
    """Read from the annotations DB table."""

    Annotations_Patch: _S.ValueType  # 39
    """To patch we need read/write.
    Deprecated.
    """

    Annotations_Delete: _S.ValueType  # 40
    """To delete we need read/write."""

    Collectors_Add: _S.ValueType  # 41
    """Write to the collectors DB table."""

    Collectors_Get: _S.ValueType  # 42
    """Read from the collectors DB table."""

    Collectors_Delete: _S.ValueType  # 43
    """To delete we need read/write."""

    Apps_Add: _S.ValueType  # 44
    """Write to the apps DB table."""

    Apps_Get: _S.ValueType  # 45
    """Read from the apps DB table."""

    Apps_Delete: _S.ValueType  # 46
    """To delete we need read/write."""

    Keys_Add: _S.ValueType  # 47
    """Write to the keys DB table."""

    Keys_Get: _S.ValueType  # 48
    """Read from the keys DB table."""

    Keys_Delete: _S.ValueType  # 49
    """To delete we need read/write."""

    Collaborators_Add: _S.ValueType  # 51
    """Write to the app sharing DB table"""

    Collaborators_Get: _S.ValueType  # 50
    """Read from the app sharing DB table"""

    Collaborators_Delete: _S.ValueType  # 52
    """To delete we need read/write"""

    Metrics_Add: _S.ValueType  # 54
    """Write to the metrics table"""

    Metrics_Get: _S.ValueType  # 53
    """Read from metrics table"""

    Metrics_Delete: _S.ValueType  # 63
    """To delete we need read/write"""

    Tasks_Add: _S.ValueType  # 55
    """Write to tasks DB table."""

    Tasks_Get: _S.ValueType  # 56
    """Read from the tasks DB table."""

    Tasks_Delete: _S.ValueType  # 70
    """To delete we need read/write"""

    PasswordPolicies_Add: _S.ValueType  # 57
    """Write to the password_policies DB table"""

    PasswordPolicies_Get: _S.ValueType  # 58
    """Read from the password_policies DB table"""

    PasswordPolicies_Delete: _S.ValueType  # 59
    """To delete password_policies we need read/write"""

    LabelOrders_Get: _S.ValueType  # 67
    """Read from label orders table"""

    LabelOrders_Add: _S.ValueType  # 68
    """Write to label orders table"""

    LabelOrders_Delete: _S.ValueType  # 69
    """To delete label orders we need read/write"""

    UserFeatureConfigs_Get: _S.ValueType  # 71
    """Read from user_feature_configs table"""

    FindDuplicateAnnotationsJobs_Add: _S.ValueType  # 102
    """CRUD on FindDuplicateAnnotationsJobs table"""

    FindDuplicateAnnotationsJobs_Get: _S.ValueType  # 103
    FindDuplicateAnnotationsJobs_Delete: _S.ValueType  # 104
    Datasets_Get: _S.ValueType  # 105
    Datasets_Add: _S.ValueType  # 106
    Datasets_Delete: _S.ValueType  # 107
    Modules_Add: _S.ValueType  # 108
    """Write to the modules DB tables."""

    Modules_Get: _S.ValueType  # 109
    """Read from the modules and modules versions DB tables."""

    Modules_Delete: _S.ValueType  # 110
    """To delete we need read/write."""

    InstalledModuleVersions_Add: _S.ValueType  # 111
    """Write to the InstalledModuleVersions DB tables."""

    InstalledModuleVersions_Get: _S.ValueType  # 112
    """Read from the InstalledModuleVersions and InstalledModuleVersions versions DB tables."""

    InstalledModuleVersions_Delete: _S.ValueType  # 113
    """To delete we need read/write."""

    Search: _S.ValueType  # 3
    """Make an rpc to our search services."""

    SavedSearch_Get: _S.ValueType  # 114
    """To get a saved search."""

    SavedSearch_Add: _S.ValueType  # 115
    """To add a saved search"""

    SavedSearch_Delete: _S.ValueType  # 116
    """To delete a saved search"""

    ModelVersionPublications_Add: _S.ValueType  # 117
    ModelVersionPublications_Delete: _S.ValueType  # 118
    WorkflowPublications_Add: _S.ValueType  # 119
    WorkflowPublications_Delete: _S.ValueType  # 120
    BulkOperation_Add: _S.ValueType  # 121
    """To write bulk operations to the DB"""

    BulkOperation_Get: _S.ValueType  # 122
    """To Read Bulk Operations from the DB"""

    BulkOperation_Delete: _S.ValueType  # 123
    """To Delete Bulk Operations from the DB"""

    HistoricalUsage_Get: _S.ValueType  # 124
    """To read historical usage from usage.dashboard_items table"""

    InputsAddJobs_Add: _S.ValueType  # 125
    """TODO(Hemanth): Expose scope after endpoints implementation
    To write Ingest cloud inputs jobs to the DB
    """

    InputsAddJobs_Get: _S.ValueType  # 126
    """To Read Ingest cloud inputs jobs to the DB
    [(clarfai_exposed) = true];
    """

    Uploads_Get: _S.ValueType  # 128
    """To read uploaded files and archives info from Uploads endpoints"""

    Uploads_Add: _S.ValueType  # 129
    """To upload files or archives through the Uploads endpoints"""

    Uploads_Delete: _S.ValueType  # 130
class S(_S, metaclass=_SEnumTypeWrapper):
    """Next index: 41
    NOTE: When updating the list of "clarifai_exposed" scopes, please also
    update the TestScopes function in main_key_test.go and TestGetExposedScopes function in
    scope_test.go

    The dependencies listed for each scope are simply recommendations so that a user
    cannot make a key that would be useless. Beyond the key creation they are not enforced
    but rather the scopes are enforce when data is accessed.


    There is the following conventions in place, make sure you add them to the scopes for all new
    resource types:

    1. *_Add requires the corresponding _Get.
    2. *_Delete requires the corresponding _Add and _Get.
    3. *_Patch is deprecated and not check anywhere.

    Think of the dependencies in this file at the DB level. If you cannot make a DB call to Get, Add
    or Delete a resource without having access to another resource then you should add it here. That
    should for the most part be the same resource type. In service.proto for the API level you will
    also specify cl_depending_scopes for each API endpoint. Those cover cases where an endpoint
    might need to access more than just that one resource type in order to operate (ie. API handlers
    that make multiple DB calls of various resource types likely have more cl_depending_scopes than
    the ones listed below). For example: PostCollectors to create a collector we make sure that you
    can do model predictions, get concepts, etc. so that you don't have a collector that would be
    useless at the end of that API handler but below you can see that the dependencies of Collector
    scopes are only on other Collector scopes.
    """
    pass

undef: S.ValueType  # 0
"""introduce undef so that the zero (default/unset) value of the enum is not a real
permission.  undef is only present for this purpose and should not be used
to indicate any "real" value.
"""

All: S.ValueType  # 1
Predict: S.ValueType  # 2
"""Make an rpc to our prediction services."""

Inputs_Add: S.ValueType  # 4
"""Write to the inputs table in the DB."""

Inputs_Get: S.ValueType  # 5
"""Read from the inputs table in the DB."""

Inputs_Patch: S.ValueType  # 7
"""To patch we need read/write.
Deprecated.
Optionally needs Concepts_Add.
"""

Inputs_Delete: S.ValueType  # 8
"""To delete we need read/write"""

Outputs_Patch: S.ValueType  # 9
"""Deprecated."""

Concepts_Add: S.ValueType  # 10
"""Write to the concepts DB tables."""

Concepts_Get: S.ValueType  # 11
"""Read from the concepts DB tables."""

Concepts_Patch: S.ValueType  # 12
"""TODO: No concept searches scope.

To patch we need read/write.
Deprecated
"""

Concepts_Delete: S.ValueType  # 13
"""To delete we need read/write.
Note: not implemented.
"""

Models_Add: S.ValueType  # 14
"""Write to the models DB tables."""

Models_Get: S.ValueType  # 15
"""Read from the models and models versions DB tables."""

Models_Patch: S.ValueType  # 16
"""To patch we need read/write.
Deprecated.
"""

Models_Delete: S.ValueType  # 17
"""To delete we need read/write."""

Models_Train: S.ValueType  # 26
"""Note: Models_Train is effectively doing POST /models/{models_id}/versions so it's treated that
way in terms of data access under the hood.
Write to the model versions DB table.
"""

Models_Sync: S.ValueType  # 27
"""Internal only model syncing."""

Workflows_Add: S.ValueType  # 18
"""TODO: No model get metrics scope.

Write to the workflows DB table.
"""

Workflows_Get: S.ValueType  # 19
"""Read from the workflows DB table."""

Workflows_Patch: S.ValueType  # 20
"""To patch we need read/write.
Deprecated.
"""

Workflows_Delete: S.ValueType  # 21
"""To delete we need read/write."""

WorkflowMetrics_Get: S.ValueType  # 96
WorkflowMetrics_Add: S.ValueType  # 97
WorkflowMetrics_Delete: S.ValueType  # 98
TSNEVisualizations_Add: S.ValueType  # 24
"""Write to the visualizations DB table.
Deprecated
"""

TSNEVisualizations_Get: S.ValueType  # 25
"""Read from the visualizations DB table.
Deprecated
"""

Annotations_Add: S.ValueType  # 37
"""Write to the annotations DB table."""

Annotations_Get: S.ValueType  # 38
"""Read from the annotations DB table."""

Annotations_Patch: S.ValueType  # 39
"""To patch we need read/write.
Deprecated.
"""

Annotations_Delete: S.ValueType  # 40
"""To delete we need read/write."""

Collectors_Add: S.ValueType  # 41
"""Write to the collectors DB table."""

Collectors_Get: S.ValueType  # 42
"""Read from the collectors DB table."""

Collectors_Delete: S.ValueType  # 43
"""To delete we need read/write."""

Apps_Add: S.ValueType  # 44
"""Write to the apps DB table."""

Apps_Get: S.ValueType  # 45
"""Read from the apps DB table."""

Apps_Delete: S.ValueType  # 46
"""To delete we need read/write."""

Keys_Add: S.ValueType  # 47
"""Write to the keys DB table."""

Keys_Get: S.ValueType  # 48
"""Read from the keys DB table."""

Keys_Delete: S.ValueType  # 49
"""To delete we need read/write."""

Collaborators_Add: S.ValueType  # 51
"""Write to the app sharing DB table"""

Collaborators_Get: S.ValueType  # 50
"""Read from the app sharing DB table"""

Collaborators_Delete: S.ValueType  # 52
"""To delete we need read/write"""

Metrics_Add: S.ValueType  # 54
"""Write to the metrics table"""

Metrics_Get: S.ValueType  # 53
"""Read from metrics table"""

Metrics_Delete: S.ValueType  # 63
"""To delete we need read/write"""

Tasks_Add: S.ValueType  # 55
"""Write to tasks DB table."""

Tasks_Get: S.ValueType  # 56
"""Read from the tasks DB table."""

Tasks_Delete: S.ValueType  # 70
"""To delete we need read/write"""

PasswordPolicies_Add: S.ValueType  # 57
"""Write to the password_policies DB table"""

PasswordPolicies_Get: S.ValueType  # 58
"""Read from the password_policies DB table"""

PasswordPolicies_Delete: S.ValueType  # 59
"""To delete password_policies we need read/write"""

LabelOrders_Get: S.ValueType  # 67
"""Read from label orders table"""

LabelOrders_Add: S.ValueType  # 68
"""Write to label orders table"""

LabelOrders_Delete: S.ValueType  # 69
"""To delete label orders we need read/write"""

UserFeatureConfigs_Get: S.ValueType  # 71
"""Read from user_feature_configs table"""

FindDuplicateAnnotationsJobs_Add: S.ValueType  # 102
"""CRUD on FindDuplicateAnnotationsJobs table"""

FindDuplicateAnnotationsJobs_Get: S.ValueType  # 103
FindDuplicateAnnotationsJobs_Delete: S.ValueType  # 104
Datasets_Get: S.ValueType  # 105
Datasets_Add: S.ValueType  # 106
Datasets_Delete: S.ValueType  # 107
Modules_Add: S.ValueType  # 108
"""Write to the modules DB tables."""

Modules_Get: S.ValueType  # 109
"""Read from the modules and modules versions DB tables."""

Modules_Delete: S.ValueType  # 110
"""To delete we need read/write."""

InstalledModuleVersions_Add: S.ValueType  # 111
"""Write to the InstalledModuleVersions DB tables."""

InstalledModuleVersions_Get: S.ValueType  # 112
"""Read from the InstalledModuleVersions and InstalledModuleVersions versions DB tables."""

InstalledModuleVersions_Delete: S.ValueType  # 113
"""To delete we need read/write."""

Search: S.ValueType  # 3
"""Make an rpc to our search services."""

SavedSearch_Get: S.ValueType  # 114
"""To get a saved search."""

SavedSearch_Add: S.ValueType  # 115
"""To add a saved search"""

SavedSearch_Delete: S.ValueType  # 116
"""To delete a saved search"""

ModelVersionPublications_Add: S.ValueType  # 117
ModelVersionPublications_Delete: S.ValueType  # 118
WorkflowPublications_Add: S.ValueType  # 119
WorkflowPublications_Delete: S.ValueType  # 120
BulkOperation_Add: S.ValueType  # 121
"""To write bulk operations to the DB"""

BulkOperation_Get: S.ValueType  # 122
"""To Read Bulk Operations from the DB"""

BulkOperation_Delete: S.ValueType  # 123
"""To Delete Bulk Operations from the DB"""

HistoricalUsage_Get: S.ValueType  # 124
"""To read historical usage from usage.dashboard_items table"""

InputsAddJobs_Add: S.ValueType  # 125
"""TODO(Hemanth): Expose scope after endpoints implementation
To write Ingest cloud inputs jobs to the DB
"""

InputsAddJobs_Get: S.ValueType  # 126
"""To Read Ingest cloud inputs jobs to the DB
[(clarfai_exposed) = true];
"""

Uploads_Get: S.ValueType  # 128
"""To read uploaded files and archives info from Uploads endpoints"""

Uploads_Add: S.ValueType  # 129
"""To upload files or archives through the Uploads endpoints"""

Uploads_Delete: S.ValueType  # 130
global___S = S


class ScopeList(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    SCOPES_FIELD_NUMBER: builtins.int
    ENDPOINTS_FIELD_NUMBER: builtins.int
    @property
    def scopes(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[global___S.ValueType]:
        """These are the list of low-level scopes to check from the enum below."""
        pass
    @property
    def endpoints(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """This is a list of fully qualified grpc names to check."""
        pass
    def __init__(self,
        *,
        scopes: typing.Optional[typing.Iterable[global___S.ValueType]] = ...,
        endpoints: typing.Optional[typing.Iterable[typing.Text]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["endpoints",b"endpoints","scopes",b"scopes"]) -> None: ...
global___ScopeList = ScopeList

CLARFAI_EXPOSED_FIELD_NUMBER: builtins.int
CLARIFAI_DEPENDING_SCOPES_FIELD_NUMBER: builtins.int
clarfai_exposed: google.protobuf.internal.extension_dict._ExtensionFieldDescriptor[google.protobuf.descriptor_pb2.EnumValueOptions, builtins.bool]
"""indicates whether the given scope should be returned by the Get /scopes/ call
or any other call that returns list of available perms.
"""

clarifai_depending_scopes: google.protobuf.internal.extension_dict._ExtensionFieldDescriptor[google.protobuf.descriptor_pb2.EnumValueOptions, google.protobuf.internal.containers.RepeatedScalarFieldContainer[global___S.ValueType]]
"""TODO: We have no way of picking extension field numbers within clarifai to guarantee
uniqueness.  Note: 50000-99999 are for organizational use (like this)
"""
