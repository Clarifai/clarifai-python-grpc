# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/clarifai/api/utils/extensions.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import descriptor_pb2 as google_dot_protobuf_dot_descriptor__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n)proto/clarifai/api/utils/extensions.proto\x12\x12\x63larifai.api.utils\x1a google/protobuf/descriptor.proto:9\n\x10\x63l_show_if_empty\x12\x1d.google.protobuf.FieldOptions\x18\xd0\x86\x03 \x01(\x08:4\n\x0b\x63l_moretags\x12\x1d.google.protobuf.FieldOptions\x18\xd1\x86\x03 \x01(\t:9\n\x10\x63l_default_float\x12\x1d.google.protobuf.FieldOptions\x18\xda\x86\x03 \x01(\x02\x42\x90\x01\n\x1b\x63om.clarifai.grpc.api.utilsP\x01Zhgithub.com/Clarifai/clarifai-go-grpc/proto/github.com/Clarifai/clarifai-go-grpc/proto/clarifai/api/utils\xa2\x02\x04\x43\x41IPb\x06proto3')


CL_SHOW_IF_EMPTY_FIELD_NUMBER = 50000
cl_show_if_empty = DESCRIPTOR.extensions_by_name['cl_show_if_empty']
CL_MORETAGS_FIELD_NUMBER = 50001
cl_moretags = DESCRIPTOR.extensions_by_name['cl_moretags']
CL_DEFAULT_FLOAT_FIELD_NUMBER = 50010
cl_default_float = DESCRIPTOR.extensions_by_name['cl_default_float']

if _descriptor._USE_C_DESCRIPTORS == False:
  google_dot_protobuf_dot_descriptor__pb2.FieldOptions.RegisterExtension(cl_show_if_empty)
  google_dot_protobuf_dot_descriptor__pb2.FieldOptions.RegisterExtension(cl_moretags)
  google_dot_protobuf_dot_descriptor__pb2.FieldOptions.RegisterExtension(cl_default_float)

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\033com.clarifai.grpc.api.utilsP\001Zhgithub.com/Clarifai/clarifai-go-grpc/proto/github.com/Clarifai/clarifai-go-grpc/proto/clarifai/api/utils\242\002\004CAIP'
# @@protoc_insertion_point(module_scope)
