from google.protobuf.json_format import _Parser
from google.protobuf.message import Message  # noqa

from clarifai_grpc.grpc.api.utils import extensions_pb2

# Python 3 deprecates getargspec and introduces getfullargspec, which Python 2 doesn't have.
try:
    from inspect import getfullargspec as get_args
except ImportError:
    from inspect import getargspec as get_args


def dict_to_protobuf(protobuf_class, js_dict, ignore_unknown_fields=False):
    # type: (type(Message), dict, bool) -> Message
    message = protobuf_class()

    # Protobuf versions 3.6.* and 3.7.0 require a different number of parameters in the _Parser's
    # constructor. In the case of 3.6.*, we pass only the argument ignore_unknown_fields, but in
    # the case of 3.7.0, we pass in one additional None parameter. To be future proof(ish), pass in
    # None to any subsequent parameter.
    num_of_args = len(get_args(_Parser.__init__).args)
    none_args = [None] * (num_of_args - 2)  # Subtract 2 for self and ignore_unknown_fields.
    parser = _CustomParser(ignore_unknown_fields, *none_args)

    parser.ConvertMessage(js_dict, message)
    return message


class _CustomParser(_Parser):
    def _ConvertFieldValuePair(self, js, message):
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

        super(_CustomParser, self)._ConvertFieldValuePair(js, message)
