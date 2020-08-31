import typing  # noqa

from google.protobuf import descriptor
from google.protobuf.json_format import _IsMapEntry, _Printer
from google.protobuf.message import Message  # noqa

from clarifai_grpc.grpc.api.utils import extensions_pb2


def protobuf_to_dict(object_protobuf, use_integers_for_enums=True, ignore_show_empty=False):
    # type: (Message, typing.Optional[bool], typing.Optional[bool]) -> dict

    # printer = _CustomPrinter(
    printer = _CustomPrinter(
        including_default_value_fields=False,
        preserving_proto_field_name=True,
        use_integers_for_enums=use_integers_for_enums,
        ignore_show_empty=ignore_show_empty,
    )
    # pylint: disable=protected-access
    return printer._MessageToJsonObject(object_protobuf)


class _CustomPrinter(_Printer):
    def __init__(
        self,
        including_default_value_fields,
        preserving_proto_field_name,
        use_integers_for_enums,
        ignore_show_empty,
    ):
        super(_CustomPrinter, self).__init__(
            including_default_value_fields,
            preserving_proto_field_name,
            use_integers_for_enums,
        )
        self._ignore_show_empty = ignore_show_empty

    def _RegularMessageToJsonObject(self, message, js):
        """
        Because of the fields with the custom extension `cl_show_if_empty`, we need to adjust the
        original's method's return JSON object and keep these fields.
        """

        js = super(_CustomPrinter, self)._RegularMessageToJsonObject(message, js)

        message_descriptor = message.DESCRIPTOR
        for field in message_descriptor.fields:

            if (
                self._ignore_show_empty
                and not field.GetOptions().Extensions[extensions_pb2.cl_default_float]
            ):
                continue
            if not field.GetOptions().Extensions[extensions_pb2.cl_show_if_empty]:
                continue

            # Singular message fields and oneof fields will not be affected.
            if (
                field.label != descriptor.FieldDescriptor.LABEL_REPEATED
                and field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE
            ) or field.containing_oneof:
                continue
            if self.preserving_proto_field_name:
                name = field.name
            else:
                name = field.json_name
            if name in js:
                # Skip the field which has been serialized already.
                continue
            if _IsMapEntry(field):
                js[name] = {}
            elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
                js[name] = []
            else:
                js[name] = self._FieldToJsonObject(field, field.default_value)

        return js

    def _StructMessageToJsonObject(self, message):
        """
        Converts Struct message according to Proto3 JSON Specification.

        However, by default, empty objects {} get converted to null. We overwrite this behavior so {}
        get converted to {}.
        """

        fields = message.fields
        ret = {}
        for key in fields:
            # When there's a Struct with an empty Struct field, this condition will hold True.
            # Far as I know this is the only case this condition will be true. If not, this condition
            # needs to be amended.
            if fields[key].WhichOneof("kind") is None:
                json_object = {}
            else:
                json_object = self._ValueMessageToJsonObject(fields[key])
            ret[key] = json_object
        return ret
