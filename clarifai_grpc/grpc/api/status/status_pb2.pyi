"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""

import builtins
import collections.abc
import sys

import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import proto.clarifai.api.status.status_code_pb2

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class Status(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    CODE_FIELD_NUMBER: builtins.int
    DESCRIPTION_FIELD_NUMBER: builtins.int
    DETAILS_FIELD_NUMBER: builtins.int
    STACK_TRACE_FIELD_NUMBER: builtins.int
    PERCENT_COMPLETED_FIELD_NUMBER: builtins.int
    TIME_REMAINING_FIELD_NUMBER: builtins.int
    REQ_ID_FIELD_NUMBER: builtins.int
    INTERNAL_DETAILS_FIELD_NUMBER: builtins.int
    REDIRECT_INFO_FIELD_NUMBER: builtins.int
    DEVELOPER_NOTES_FIELD_NUMBER: builtins.int
    code: proto.clarifai.api.status.status_code_pb2.StatusCode.ValueType
    """Status code from internal codes."""
    description: builtins.str
    """A short description of the error."""
    details: builtins.str
    """More details of the given error.
    These details may be exposed to non-technical users.
    For technical details, try to use developer_notes field.
    """
    @property
    def stack_trace(
        self,
    ) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """For some environment we may return a stack trace to help debug
        any issues.
        """
    percent_completed: builtins.int
    """specifically for long running jobs"""
    time_remaining: builtins.int
    """if status is pending, how much time is remaining (in seconds)"""
    req_id: builtins.str
    """A request ID may be present, to help monitoring and tracking requests"""
    internal_details: builtins.str
    """Internal Annotation (do not set in production, for internal Clarifai use only)."""
    @property
    def redirect_info(self) -> global___RedirectInfo:
        """Resource location info for redirect, when resource location has been changed."""
    developer_notes: builtins.str
    """Notes for developer.
    These notes are rather technical details for developers how to interpret the status,
    e.g. why an error occurred and how to avoid getting the error.
    """
    def __init__(
        self,
        *,
        code: proto.clarifai.api.status.status_code_pb2.StatusCode.ValueType = ...,
        description: builtins.str = ...,
        details: builtins.str = ...,
        stack_trace: collections.abc.Iterable[builtins.str] | None = ...,
        percent_completed: builtins.int = ...,
        time_remaining: builtins.int = ...,
        req_id: builtins.str = ...,
        internal_details: builtins.str = ...,
        redirect_info: global___RedirectInfo | None = ...,
        developer_notes: builtins.str = ...,
    ) -> None: ...
    def HasField(
        self, field_name: typing_extensions.Literal["redirect_info", b"redirect_info"]
    ) -> builtins.bool: ...
    def ClearField(
        self,
        field_name: typing_extensions.Literal[
            "code",
            b"code",
            "description",
            b"description",
            "details",
            b"details",
            "developer_notes",
            b"developer_notes",
            "internal_details",
            b"internal_details",
            "percent_completed",
            b"percent_completed",
            "redirect_info",
            b"redirect_info",
            "req_id",
            b"req_id",
            "stack_trace",
            b"stack_trace",
            "time_remaining",
            b"time_remaining",
        ],
    ) -> None: ...

global___Status = Status

@typing_extensions.final
class RedirectInfo(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    URL_FIELD_NUMBER: builtins.int
    RESOURCE_TYPE_FIELD_NUMBER: builtins.int
    OLD_RESOURCE_ID_FIELD_NUMBER: builtins.int
    NEW_RESOURCE_ID_FIELD_NUMBER: builtins.int
    url: builtins.str
    """New location for the resource. Used to set response Location header."""
    resource_type: builtins.str
    """Resource type"""
    old_resource_id: builtins.str
    """Old resource id"""
    new_resource_id: builtins.str
    """New resource id"""
    def __init__(
        self,
        *,
        url: builtins.str = ...,
        resource_type: builtins.str = ...,
        old_resource_id: builtins.str = ...,
        new_resource_id: builtins.str = ...,
    ) -> None: ...
    def ClearField(
        self,
        field_name: typing_extensions.Literal[
            "new_resource_id",
            b"new_resource_id",
            "old_resource_id",
            b"old_resource_id",
            "resource_type",
            b"resource_type",
            "url",
            b"url",
        ],
    ) -> None: ...

global___RedirectInfo = RedirectInfo

@typing_extensions.final
class BaseResponse(google.protobuf.message.Message):
    """Base message to return when there is a internal server error that
    is not caught elsewhere.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STATUS_FIELD_NUMBER: builtins.int
    @property
    def status(self) -> global___Status: ...
    def __init__(
        self,
        *,
        status: global___Status | None = ...,
    ) -> None: ...
    def HasField(
        self, field_name: typing_extensions.Literal["status", b"status"]
    ) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["status", b"status"]) -> None: ...

global___BaseResponse = BaseResponse
