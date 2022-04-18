"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class MatrixUint64(google.protobuf.message.Message):
    """Store matrix of uint64s values.
    It is recommended to store the matrix as a 1D array
    because it produces less bytes during serialization than a 2D array.
    The matrix does store the number of columns, but it does not store the number of rows.
    The number of rows can automatically be calculated as length(data)/n_cols.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    N_COLS_FIELD_NUMBER: builtins.int
    DATA_FIELD_NUMBER: builtins.int
    n_cols: builtins.int
    """Number of columns"""

    @property
    def data(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """Matrix data stored as an array.
        In order to access matrix element at row i & column j, use data[i*n_cols+j].
        """
        pass
    def __init__(self,
        *,
        n_cols: builtins.int = ...,
        data: typing.Optional[typing.Iterable[builtins.int]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["data",b"data","n_cols",b"n_cols"]) -> None: ...
global___MatrixUint64 = MatrixUint64