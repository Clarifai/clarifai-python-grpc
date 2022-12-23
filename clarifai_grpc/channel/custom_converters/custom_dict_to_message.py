from google.protobuf.json_format import _Parser
from google.protobuf.message import Message  # noqa

from clarifai_grpc.grpc.api.utils import extensions_pb2


def dict_to_protobuf(
    protobuf_class,
    js_dict,
    ignore_unknown_fields=False,
    descriptor_pool=None,
    max_recursion_depth=100,
):
    """Parses a JSON dictionary representation into a message.

    Args:
      js_dict: Dict representation of a JSON message.
      message: A protocol buffer message to merge into.
      ignore_unknown_fields: If True, do not raise errors for unknown fields.
      descriptor_pool: A Descriptor Pool for resolving types. If None use the
        default.
      max_recursion_depth: max recursion depth of JSON message to be
        deserialized. JSON messages over this depth will fail to be
        deserialized. Default value is 100.

    Returns:
      The same message passed as argument.
    """
    message = protobuf_class()

    parser = _CustomParser(ignore_unknown_fields, descriptor_pool, max_recursion_depth)

    parser.ConvertMessage(js_dict, message, path="")
    return message


class _CustomParser(_Parser):
    def _ConvertFieldValuePair(self, js, message, path):
        """
        Because of fields with custom extensions such as cl_default_float, we need
        to adjust the original's method's JSON object parameter by setting them explicitly to the
        default value.
        """

        message_descriptor = message.DESCRIPTOR
        for f in message_descriptor.fields:
            default_float = f.GetOptions().Extensions[extensions_pb2.cl_default_float]
            if default_float:
                if f.name not in js:
                    js[f.name] = default_float

        super(_CustomParser, self)._ConvertFieldValuePair(js, message, path)
