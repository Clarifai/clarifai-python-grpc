# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/clarifai/auth/scope/scope.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import descriptor_pb2 as google_dot_protobuf_dot_descriptor__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n%proto/clarifai/auth/scope/scope.proto\x12\x13\x63larifai.auth.scope\x1a google/protobuf/descriptor.proto\"F\n\tScopeList\x12&\n\x06scopes\x18\x01 \x03(\x0e\x32\x16.clarifai.auth.scope.S\x12\x11\n\tendpoints\x18\x02 \x03(\t*\xe6\x14\n\x01S\x12\t\n\x05undef\x10\x00\x12\r\n\x03\x41ll\x10\x01\x1a\x04\xf0\x9b\'\x01\x12\x11\n\x07Predict\x10\x02\x1a\x04\xf0\x9b\'\x01\x12\x18\n\nInputs_Add\x10\x04\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'\x05\x12\x14\n\nInputs_Get\x10\x05\x1a\x04\xf0\x9b\'\x01\x12 \n\x0cInputs_Patch\x10\x07\x1a\x0e\x08\x01\xf0\x9b\'\x01\xf8\x9b\'\x04\xf8\x9b\'\x05\x12\x1f\n\rInputs_Delete\x10\x08\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'\x04\xf8\x9b\'\x05\x12\x1d\n\rOutputs_Patch\x10\t\x1a\n\x08\x01\xf8\x9b\'\x05\xf8\x9b\'\x02\x12\x1a\n\x0c\x43oncepts_Add\x10\n\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'\x0b\x12\x16\n\x0c\x43oncepts_Get\x10\x0b\x1a\x04\xf0\x9b\'\x01\x12\"\n\x0e\x43oncepts_Patch\x10\x0c\x1a\x0e\x08\x01\xf0\x9b\'\x01\xf8\x9b\'\n\xf8\x9b\'\x0b\x12\x1d\n\x0f\x43oncepts_Delete\x10\r\x1a\x08\xf8\x9b\'\n\xf8\x9b\'\x0b\x12\x18\n\nModels_Add\x10\x0e\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'\x0f\x12\x14\n\nModels_Get\x10\x0f\x1a\x04\xf0\x9b\'\x01\x12$\n\x0cModels_Patch\x10\x10\x1a\x12\x08\x01\xf0\x9b\'\x01\xf8\x9b\'\x0e\xf8\x9b\'\x0f\xf8\x9b\'\x1a\x12\x1f\n\rModels_Delete\x10\x11\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'\x0e\xf8\x9b\'\x0f\x12\x1a\n\x0cModels_Train\x10\x1a\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'\x0f\x12\x15\n\x0bModels_Sync\x10\x1b\x1a\x04\xf8\x9b\'\x0f\x12\x1b\n\rWorkflows_Add\x10\x12\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'\x13\x12\x17\n\rWorkflows_Get\x10\x13\x1a\x04\xf0\x9b\'\x01\x12#\n\x0fWorkflows_Patch\x10\x14\x1a\x0e\x08\x01\xf0\x9b\'\x01\xf8\x9b\'\x12\xf8\x9b\'\x13\x12\"\n\x10Workflows_Delete\x10\x15\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'\x12\xf8\x9b\'\x13\x12\x1d\n\x13WorkflowMetrics_Get\x10`\x1a\x04\xf0\x9b\'\x01\x12!\n\x13WorkflowMetrics_Add\x10\x61\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'`\x12(\n\x16WorkflowMetrics_Delete\x10\x62\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'a\xf8\x9b\'`\x12\"\n\x16TSNEVisualizations_Add\x10\x18\x1a\x06\x08\x01\xf8\x9b\'\x19\x12\x1e\n\x16TSNEVisualizations_Get\x10\x19\x1a\x02\x08\x01\x12\x1d\n\x0f\x41nnotations_Add\x10%\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'&\x12\x19\n\x0f\x41nnotations_Get\x10&\x1a\x04\xf0\x9b\'\x01\x12%\n\x11\x41nnotations_Patch\x10\'\x1a\x0e\x08\x01\xf0\x9b\'\x01\xf8\x9b\'%\xf8\x9b\'&\x12$\n\x12\x41nnotations_Delete\x10(\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'%\xf8\x9b\'&\x12\x1c\n\x0e\x43ollectors_Add\x10)\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'*\x12\x18\n\x0e\x43ollectors_Get\x10*\x1a\x04\xf0\x9b\'\x01\x12#\n\x11\x43ollectors_Delete\x10+\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\')\xf8\x9b\'*\x12\x16\n\x08\x41pps_Add\x10,\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'-\x12\x12\n\x08\x41pps_Get\x10-\x1a\x04\xf0\x9b\'\x01\x12\x1d\n\x0b\x41pps_Delete\x10.\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\',\xf8\x9b\'-\x12\x16\n\x08Keys_Add\x10/\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'0\x12\x12\n\x08Keys_Get\x10\x30\x1a\x04\xf0\x9b\'\x01\x12\x1d\n\x0bKeys_Delete\x10\x31\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'/\xf8\x9b\'0\x12\x1f\n\x11\x43ollaborators_Add\x10\x33\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'2\x12\x1b\n\x11\x43ollaborators_Get\x10\x32\x1a\x04\xf0\x9b\'\x01\x12&\n\x14\x43ollaborators_Delete\x10\x34\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'3\xf8\x9b\'2\x12\x19\n\x0bMetrics_Add\x10\x36\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'5\x12\x15\n\x0bMetrics_Get\x10\x35\x1a\x04\xf0\x9b\'\x01\x12 \n\x0eMetrics_Delete\x10?\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'6\xf8\x9b\'5\x12\x17\n\tTasks_Add\x10\x37\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'8\x12\x13\n\tTasks_Get\x10\x38\x1a\x04\xf0\x9b\'\x01\x12\x1e\n\x0cTasks_Delete\x10\x46\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'7\xf8\x9b\'8\x12\"\n\x14PasswordPolicies_Add\x10\x39\x1a\x08\xf0\x9b\'\x01\xf8\x9b\':\x12\x1e\n\x14PasswordPolicies_Get\x10:\x1a\x04\xf0\x9b\'\x01\x12)\n\x17PasswordPolicies_Delete\x10;\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'9\xf8\x9b\':\x12\x19\n\x0fLabelOrders_Get\x10\x43\x1a\x04\xf0\x9b\'\x01\x12\x1d\n\x0fLabelOrders_Add\x10\x44\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'C\x12$\n\x12LabelOrders_Delete\x10\x45\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'D\xf8\x9b\'C\x12 \n\x16UserFeatureConfigs_Get\x10G\x1a\x04\xf0\x9b\'\x01\x12.\n FindDuplicateAnnotationsJobs_Add\x10\x66\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'g\x12*\n FindDuplicateAnnotationsJobs_Get\x10g\x1a\x04\xf0\x9b\'\x01\x12\x35\n#FindDuplicateAnnotationsJobs_Delete\x10h\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'f\xf8\x9b\'g\x12\x16\n\x0c\x44\x61tasets_Get\x10i\x1a\x04\xf0\x9b\'\x01\x12\x1a\n\x0c\x44\x61tasets_Add\x10j\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'i\x12!\n\x0f\x44\x61tasets_Delete\x10k\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'i\xf8\x9b\'j\x12\x19\n\x0bModules_Add\x10l\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'m\x12\x15\n\x0bModules_Get\x10m\x1a\x04\xf0\x9b\'\x01\x12 \n\x0eModules_Delete\x10n\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'l\xf8\x9b\'m\x12-\n\x1bInstalledModuleVersions_Add\x10o\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'p\xf8\x9b\'m\x12)\n\x1bInstalledModuleVersions_Get\x10p\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'m\x12\x34\n\x1eInstalledModuleVersions_Delete\x10q\x1a\x10\xf0\x9b\'\x01\xf8\x9b\'o\xf8\x9b\'p\xf8\x9b\'m\x12\x10\n\x06Search\x10\x03\x1a\x04\xf0\x9b\'\x01\x12\x19\n\x0fSavedSearch_Get\x10r\x1a\x04\xf0\x9b\'\x01\x12\x1d\n\x0fSavedSearch_Add\x10s\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'r\x12$\n\x12SavedSearch_Delete\x10t\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'r\xf8\x9b\'s\x12&\n\x1cModelVersionPublications_Add\x10u\x1a\x04\xf0\x9b\'\x01\x12)\n\x1fModelVersionPublications_Delete\x10v\x1a\x04\xf0\x9b\'\x01\x12\"\n\x18WorkflowPublications_Add\x10w\x1a\x04\xf0\x9b\'\x01\x12%\n\x1bWorkflowPublications_Delete\x10x\x1a\x04\xf0\x9b\'\x01\x12\x1f\n\x11\x42ulkOperation_Add\x10y\x1a\x08\xf0\x9b\'\x01\xf8\x9b\'z\x12\x1b\n\x11\x42ulkOperation_Get\x10z\x1a\x04\xf0\x9b\'\x01\x12&\n\x14\x42ulkOperation_Delete\x10{\x1a\x0c\xf0\x9b\'\x01\xf8\x9b\'y\xf8\x9b\'z\x12\x17\n\x13HistoricalUsage_Get\x10|\x12\x16\n\x0bUploads_Get\x10\x80\x01\x1a\x04\xf0\x9b\'\x01\x12\x1b\n\x0bUploads_Add\x10\x81\x01\x1a\t\xf0\x9b\'\x01\xf8\x9b\'\x80\x01\x12#\n\x0eUploads_Delete\x10\x82\x01\x1a\x0e\xf0\x9b\'\x01\xf8\x9b\'\x80\x01\xf8\x9b\'\x81\x01\"\x04\x08\x1e\x10\x1e\"\x04\x08\x1f\x10\x1f\"\x04\x08 \x10 \"\x04\x08!\x10!\"\x04\x08\"\x10\"\"\x04\x08}\x10}\"\x04\x08~\x10~:<\n\x0f\x63larfai_exposed\x12!.google.protobuf.EnumValueOptions\x18\xbe\xf3\x04 \x01(\x08:^\n\x19\x63larifai_depending_scopes\x12!.google.protobuf.EnumValueOptions\x18\xbf\xf3\x04 \x03(\x0e\x32\x16.clarifai.auth.scope.SBg\n\x1c\x63om.clarifai.grpc.auth.scopeP\x01Z>github.com/Clarifai/clarifai-go-grpc/proto/clarifai/auth/scope\xa2\x02\x04\x43\x41IPb\x06proto3')

_S = DESCRIPTOR.enum_types_by_name['S']
S = enum_type_wrapper.EnumTypeWrapper(_S)
undef = 0
All = 1
Predict = 2
Inputs_Add = 4
Inputs_Get = 5
Inputs_Patch = 7
Inputs_Delete = 8
Outputs_Patch = 9
Concepts_Add = 10
Concepts_Get = 11
Concepts_Patch = 12
Concepts_Delete = 13
Models_Add = 14
Models_Get = 15
Models_Patch = 16
Models_Delete = 17
Models_Train = 26
Models_Sync = 27
Workflows_Add = 18
Workflows_Get = 19
Workflows_Patch = 20
Workflows_Delete = 21
WorkflowMetrics_Get = 96
WorkflowMetrics_Add = 97
WorkflowMetrics_Delete = 98
TSNEVisualizations_Add = 24
TSNEVisualizations_Get = 25
Annotations_Add = 37
Annotations_Get = 38
Annotations_Patch = 39
Annotations_Delete = 40
Collectors_Add = 41
Collectors_Get = 42
Collectors_Delete = 43
Apps_Add = 44
Apps_Get = 45
Apps_Delete = 46
Keys_Add = 47
Keys_Get = 48
Keys_Delete = 49
Collaborators_Add = 51
Collaborators_Get = 50
Collaborators_Delete = 52
Metrics_Add = 54
Metrics_Get = 53
Metrics_Delete = 63
Tasks_Add = 55
Tasks_Get = 56
Tasks_Delete = 70
PasswordPolicies_Add = 57
PasswordPolicies_Get = 58
PasswordPolicies_Delete = 59
LabelOrders_Get = 67
LabelOrders_Add = 68
LabelOrders_Delete = 69
UserFeatureConfigs_Get = 71
FindDuplicateAnnotationsJobs_Add = 102
FindDuplicateAnnotationsJobs_Get = 103
FindDuplicateAnnotationsJobs_Delete = 104
Datasets_Get = 105
Datasets_Add = 106
Datasets_Delete = 107
Modules_Add = 108
Modules_Get = 109
Modules_Delete = 110
InstalledModuleVersions_Add = 111
InstalledModuleVersions_Get = 112
InstalledModuleVersions_Delete = 113
Search = 3
SavedSearch_Get = 114
SavedSearch_Add = 115
SavedSearch_Delete = 116
ModelVersionPublications_Add = 117
ModelVersionPublications_Delete = 118
WorkflowPublications_Add = 119
WorkflowPublications_Delete = 120
BulkOperation_Add = 121
BulkOperation_Get = 122
BulkOperation_Delete = 123
HistoricalUsage_Get = 124
Uploads_Get = 128
Uploads_Add = 129
Uploads_Delete = 130

CLARFAI_EXPOSED_FIELD_NUMBER = 80318
clarfai_exposed = DESCRIPTOR.extensions_by_name['clarfai_exposed']
CLARIFAI_DEPENDING_SCOPES_FIELD_NUMBER = 80319
clarifai_depending_scopes = DESCRIPTOR.extensions_by_name['clarifai_depending_scopes']

_SCOPELIST = DESCRIPTOR.message_types_by_name['ScopeList']
ScopeList = _reflection.GeneratedProtocolMessageType('ScopeList', (_message.Message,), {
  'DESCRIPTOR' : _SCOPELIST,
  '__module__' : 'proto.clarifai.auth.scope.scope_pb2'
  # @@protoc_insertion_point(class_scope:clarifai.auth.scope.ScopeList)
  })
_sym_db.RegisterMessage(ScopeList)

if _descriptor._USE_C_DESCRIPTORS == False:
  google_dot_protobuf_dot_descriptor__pb2.EnumValueOptions.RegisterExtension(clarfai_exposed)
  google_dot_protobuf_dot_descriptor__pb2.EnumValueOptions.RegisterExtension(clarifai_depending_scopes)

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\034com.clarifai.grpc.auth.scopeP\001Z>github.com/Clarifai/clarifai-go-grpc/proto/clarifai/auth/scope\242\002\004CAIP'
  _S.values_by_name["All"]._options = None
  _S.values_by_name["All"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Predict"]._options = None
  _S.values_by_name["Predict"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Inputs_Add"]._options = None
  _S.values_by_name["Inputs_Add"]._serialized_options = b'\360\233\'\001\370\233\'\005'
  _S.values_by_name["Inputs_Get"]._options = None
  _S.values_by_name["Inputs_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Inputs_Patch"]._options = None
  _S.values_by_name["Inputs_Patch"]._serialized_options = b'\010\001\360\233\'\001\370\233\'\004\370\233\'\005'
  _S.values_by_name["Inputs_Delete"]._options = None
  _S.values_by_name["Inputs_Delete"]._serialized_options = b'\360\233\'\001\370\233\'\004\370\233\'\005'
  _S.values_by_name["Outputs_Patch"]._options = None
  _S.values_by_name["Outputs_Patch"]._serialized_options = b'\010\001\370\233\'\005\370\233\'\002'
  _S.values_by_name["Concepts_Add"]._options = None
  _S.values_by_name["Concepts_Add"]._serialized_options = b'\360\233\'\001\370\233\'\013'
  _S.values_by_name["Concepts_Get"]._options = None
  _S.values_by_name["Concepts_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Concepts_Patch"]._options = None
  _S.values_by_name["Concepts_Patch"]._serialized_options = b'\010\001\360\233\'\001\370\233\'\n\370\233\'\013'
  _S.values_by_name["Concepts_Delete"]._options = None
  _S.values_by_name["Concepts_Delete"]._serialized_options = b'\370\233\'\n\370\233\'\013'
  _S.values_by_name["Models_Add"]._options = None
  _S.values_by_name["Models_Add"]._serialized_options = b'\360\233\'\001\370\233\'\017'
  _S.values_by_name["Models_Get"]._options = None
  _S.values_by_name["Models_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Models_Patch"]._options = None
  _S.values_by_name["Models_Patch"]._serialized_options = b'\010\001\360\233\'\001\370\233\'\016\370\233\'\017\370\233\'\032'
  _S.values_by_name["Models_Delete"]._options = None
  _S.values_by_name["Models_Delete"]._serialized_options = b'\360\233\'\001\370\233\'\016\370\233\'\017'
  _S.values_by_name["Models_Train"]._options = None
  _S.values_by_name["Models_Train"]._serialized_options = b'\360\233\'\001\370\233\'\017'
  _S.values_by_name["Models_Sync"]._options = None
  _S.values_by_name["Models_Sync"]._serialized_options = b'\370\233\'\017'
  _S.values_by_name["Workflows_Add"]._options = None
  _S.values_by_name["Workflows_Add"]._serialized_options = b'\360\233\'\001\370\233\'\023'
  _S.values_by_name["Workflows_Get"]._options = None
  _S.values_by_name["Workflows_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Workflows_Patch"]._options = None
  _S.values_by_name["Workflows_Patch"]._serialized_options = b'\010\001\360\233\'\001\370\233\'\022\370\233\'\023'
  _S.values_by_name["Workflows_Delete"]._options = None
  _S.values_by_name["Workflows_Delete"]._serialized_options = b'\360\233\'\001\370\233\'\022\370\233\'\023'
  _S.values_by_name["WorkflowMetrics_Get"]._options = None
  _S.values_by_name["WorkflowMetrics_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["WorkflowMetrics_Add"]._options = None
  _S.values_by_name["WorkflowMetrics_Add"]._serialized_options = b'\360\233\'\001\370\233\'`'
  _S.values_by_name["WorkflowMetrics_Delete"]._options = None
  _S.values_by_name["WorkflowMetrics_Delete"]._serialized_options = b'\360\233\'\001\370\233\'a\370\233\'`'
  _S.values_by_name["TSNEVisualizations_Add"]._options = None
  _S.values_by_name["TSNEVisualizations_Add"]._serialized_options = b'\010\001\370\233\'\031'
  _S.values_by_name["TSNEVisualizations_Get"]._options = None
  _S.values_by_name["TSNEVisualizations_Get"]._serialized_options = b'\010\001'
  _S.values_by_name["Annotations_Add"]._options = None
  _S.values_by_name["Annotations_Add"]._serialized_options = b'\360\233\'\001\370\233\'&'
  _S.values_by_name["Annotations_Get"]._options = None
  _S.values_by_name["Annotations_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Annotations_Patch"]._options = None
  _S.values_by_name["Annotations_Patch"]._serialized_options = b'\010\001\360\233\'\001\370\233\'%\370\233\'&'
  _S.values_by_name["Annotations_Delete"]._options = None
  _S.values_by_name["Annotations_Delete"]._serialized_options = b'\360\233\'\001\370\233\'%\370\233\'&'
  _S.values_by_name["Collectors_Add"]._options = None
  _S.values_by_name["Collectors_Add"]._serialized_options = b'\360\233\'\001\370\233\'*'
  _S.values_by_name["Collectors_Get"]._options = None
  _S.values_by_name["Collectors_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Collectors_Delete"]._options = None
  _S.values_by_name["Collectors_Delete"]._serialized_options = b'\360\233\'\001\370\233\')\370\233\'*'
  _S.values_by_name["Apps_Add"]._options = None
  _S.values_by_name["Apps_Add"]._serialized_options = b'\360\233\'\001\370\233\'-'
  _S.values_by_name["Apps_Get"]._options = None
  _S.values_by_name["Apps_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Apps_Delete"]._options = None
  _S.values_by_name["Apps_Delete"]._serialized_options = b'\360\233\'\001\370\233\',\370\233\'-'
  _S.values_by_name["Keys_Add"]._options = None
  _S.values_by_name["Keys_Add"]._serialized_options = b'\360\233\'\001\370\233\'0'
  _S.values_by_name["Keys_Get"]._options = None
  _S.values_by_name["Keys_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Keys_Delete"]._options = None
  _S.values_by_name["Keys_Delete"]._serialized_options = b'\360\233\'\001\370\233\'/\370\233\'0'
  _S.values_by_name["Collaborators_Add"]._options = None
  _S.values_by_name["Collaborators_Add"]._serialized_options = b'\360\233\'\001\370\233\'2'
  _S.values_by_name["Collaborators_Get"]._options = None
  _S.values_by_name["Collaborators_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Collaborators_Delete"]._options = None
  _S.values_by_name["Collaborators_Delete"]._serialized_options = b'\360\233\'\001\370\233\'3\370\233\'2'
  _S.values_by_name["Metrics_Add"]._options = None
  _S.values_by_name["Metrics_Add"]._serialized_options = b'\360\233\'\001\370\233\'5'
  _S.values_by_name["Metrics_Get"]._options = None
  _S.values_by_name["Metrics_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Metrics_Delete"]._options = None
  _S.values_by_name["Metrics_Delete"]._serialized_options = b'\360\233\'\001\370\233\'6\370\233\'5'
  _S.values_by_name["Tasks_Add"]._options = None
  _S.values_by_name["Tasks_Add"]._serialized_options = b'\360\233\'\001\370\233\'8'
  _S.values_by_name["Tasks_Get"]._options = None
  _S.values_by_name["Tasks_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Tasks_Delete"]._options = None
  _S.values_by_name["Tasks_Delete"]._serialized_options = b'\360\233\'\001\370\233\'7\370\233\'8'
  _S.values_by_name["PasswordPolicies_Add"]._options = None
  _S.values_by_name["PasswordPolicies_Add"]._serialized_options = b'\360\233\'\001\370\233\':'
  _S.values_by_name["PasswordPolicies_Get"]._options = None
  _S.values_by_name["PasswordPolicies_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["PasswordPolicies_Delete"]._options = None
  _S.values_by_name["PasswordPolicies_Delete"]._serialized_options = b'\360\233\'\001\370\233\'9\370\233\':'
  _S.values_by_name["LabelOrders_Get"]._options = None
  _S.values_by_name["LabelOrders_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["LabelOrders_Add"]._options = None
  _S.values_by_name["LabelOrders_Add"]._serialized_options = b'\360\233\'\001\370\233\'C'
  _S.values_by_name["LabelOrders_Delete"]._options = None
  _S.values_by_name["LabelOrders_Delete"]._serialized_options = b'\360\233\'\001\370\233\'D\370\233\'C'
  _S.values_by_name["UserFeatureConfigs_Get"]._options = None
  _S.values_by_name["UserFeatureConfigs_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["FindDuplicateAnnotationsJobs_Add"]._options = None
  _S.values_by_name["FindDuplicateAnnotationsJobs_Add"]._serialized_options = b'\360\233\'\001\370\233\'g'
  _S.values_by_name["FindDuplicateAnnotationsJobs_Get"]._options = None
  _S.values_by_name["FindDuplicateAnnotationsJobs_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["FindDuplicateAnnotationsJobs_Delete"]._options = None
  _S.values_by_name["FindDuplicateAnnotationsJobs_Delete"]._serialized_options = b'\360\233\'\001\370\233\'f\370\233\'g'
  _S.values_by_name["Datasets_Get"]._options = None
  _S.values_by_name["Datasets_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Datasets_Add"]._options = None
  _S.values_by_name["Datasets_Add"]._serialized_options = b'\360\233\'\001\370\233\'i'
  _S.values_by_name["Datasets_Delete"]._options = None
  _S.values_by_name["Datasets_Delete"]._serialized_options = b'\360\233\'\001\370\233\'i\370\233\'j'
  _S.values_by_name["Modules_Add"]._options = None
  _S.values_by_name["Modules_Add"]._serialized_options = b'\360\233\'\001\370\233\'m'
  _S.values_by_name["Modules_Get"]._options = None
  _S.values_by_name["Modules_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Modules_Delete"]._options = None
  _S.values_by_name["Modules_Delete"]._serialized_options = b'\360\233\'\001\370\233\'l\370\233\'m'
  _S.values_by_name["InstalledModuleVersions_Add"]._options = None
  _S.values_by_name["InstalledModuleVersions_Add"]._serialized_options = b'\360\233\'\001\370\233\'p\370\233\'m'
  _S.values_by_name["InstalledModuleVersions_Get"]._options = None
  _S.values_by_name["InstalledModuleVersions_Get"]._serialized_options = b'\360\233\'\001\370\233\'m'
  _S.values_by_name["InstalledModuleVersions_Delete"]._options = None
  _S.values_by_name["InstalledModuleVersions_Delete"]._serialized_options = b'\360\233\'\001\370\233\'o\370\233\'p\370\233\'m'
  _S.values_by_name["Search"]._options = None
  _S.values_by_name["Search"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["SavedSearch_Get"]._options = None
  _S.values_by_name["SavedSearch_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["SavedSearch_Add"]._options = None
  _S.values_by_name["SavedSearch_Add"]._serialized_options = b'\360\233\'\001\370\233\'r'
  _S.values_by_name["SavedSearch_Delete"]._options = None
  _S.values_by_name["SavedSearch_Delete"]._serialized_options = b'\360\233\'\001\370\233\'r\370\233\'s'
  _S.values_by_name["ModelVersionPublications_Add"]._options = None
  _S.values_by_name["ModelVersionPublications_Add"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["ModelVersionPublications_Delete"]._options = None
  _S.values_by_name["ModelVersionPublications_Delete"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["WorkflowPublications_Add"]._options = None
  _S.values_by_name["WorkflowPublications_Add"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["WorkflowPublications_Delete"]._options = None
  _S.values_by_name["WorkflowPublications_Delete"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["BulkOperation_Add"]._options = None
  _S.values_by_name["BulkOperation_Add"]._serialized_options = b'\360\233\'\001\370\233\'z'
  _S.values_by_name["BulkOperation_Get"]._options = None
  _S.values_by_name["BulkOperation_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["BulkOperation_Delete"]._options = None
  _S.values_by_name["BulkOperation_Delete"]._serialized_options = b'\360\233\'\001\370\233\'y\370\233\'z'
  _S.values_by_name["Uploads_Get"]._options = None
  _S.values_by_name["Uploads_Get"]._serialized_options = b'\360\233\'\001'
  _S.values_by_name["Uploads_Add"]._options = None
  _S.values_by_name["Uploads_Add"]._serialized_options = b'\360\233\'\001\370\233\'\200\001'
  _S.values_by_name["Uploads_Delete"]._options = None
  _S.values_by_name["Uploads_Delete"]._serialized_options = b'\360\233\'\001\370\233\'\200\001\370\233\'\201\001'
  _S._serialized_start=169
  _S._serialized_end=2831
  _SCOPELIST._serialized_start=96
  _SCOPELIST._serialized_end=166
# @@protoc_insertion_point(module_scope)
