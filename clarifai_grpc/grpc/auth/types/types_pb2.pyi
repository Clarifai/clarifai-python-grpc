"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""

import builtins
import sys
import typing

import google.protobuf.descriptor
import google.protobuf.internal.enum_type_wrapper

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _AuthType:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _AuthTypeEnumTypeWrapper(
    google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_AuthType.ValueType], builtins.type
):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    undef: _AuthType.ValueType  # 0
    """introduce undef so that the zero (default/unset) value of the enum is not a real
    permission.  undef is only present for this purpose and should not be used
    to indicate any "real" value.
    """
    NoAuth: _AuthType.ValueType  # 1
    """No authorization need for this endpoint."""
    KeyAuth: _AuthType.ValueType  # 2
    """This authorization requires API keys (both app-spceific keys and personal access tokens).
    The endpoints that use this AuthType may also include a list of
    clarifai.auth.utils.cl_depending_scopes.
    """
    SessionTokenAuth: _AuthType.ValueType  # 3
    """This uses a session token from your web browser. This is reserved for users/account level APIs
    that are only needed in a browser.
    """
    AdminAuth: _AuthType.ValueType  # 4
    """This uses a special token for admin access to the APIs."""
    PATAuth: _AuthType.ValueType  # 5
    """This authorization requires personal access tokens. This is used for endpoints such as
    /users/{user_id}/apps which are not specific. An app-specific API key will not work
    when PATAuth is used.
    """

class AuthType(_AuthType, metaclass=_AuthTypeEnumTypeWrapper):
    """Authorization type for endpoints."""

undef: AuthType.ValueType  # 0
"""introduce undef so that the zero (default/unset) value of the enum is not a real
permission.  undef is only present for this purpose and should not be used
to indicate any "real" value.
"""
NoAuth: AuthType.ValueType  # 1
"""No authorization need for this endpoint."""
KeyAuth: AuthType.ValueType  # 2
"""This authorization requires API keys (both app-spceific keys and personal access tokens).
The endpoints that use this AuthType may also include a list of
clarifai.auth.utils.cl_depending_scopes.
"""
SessionTokenAuth: AuthType.ValueType  # 3
"""This uses a session token from your web browser. This is reserved for users/account level APIs
that are only needed in a browser.
"""
AdminAuth: AuthType.ValueType  # 4
"""This uses a special token for admin access to the APIs."""
PATAuth: AuthType.ValueType  # 5
"""This authorization requires personal access tokens. This is used for endpoints such as
/users/{user_id}/apps which are not specific. An app-specific API key will not work
when PATAuth is used.
"""
global___AuthType = AuthType
