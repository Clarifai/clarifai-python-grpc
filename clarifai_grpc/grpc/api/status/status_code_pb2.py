# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/clarifai/api/status/status_code.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n+proto/clarifai/api/status/status_code.proto\x12\x13\x63larifai.api.status*\xfb_\n\nStatusCode\x12\x08\n\x04ZERO\x10\x00\x12\x0c\n\x07SUCCESS\x10\x90N\x12\x11\n\x0cMIXED_STATUS\x10\x9aN\x12\x0c\n\x07\x46\x41ILURE\x10\xa4N\x12\x0e\n\tTRY_AGAIN\x10\xaeN\x12\x14\n\x0fNOT_IMPLEMENTED\x10\xb8N\x12\n\n\x05MOVED\x10\xc2N\x12\x0b\n\x06TEAPOT\x10\xccN\x12\x18\n\x13\x43ONN_ACCOUNT_ISSUES\x10\xf8U\x12\x1b\n\x12\x43ONN_TOKEN_INVALID\x10\xf9U\x1a\x02\x08\x01\x12\x1d\n\x18\x43ONN_CREDENTIALS_INVALID\x10\xfaU\x12\x1d\n\x18\x43ONN_EXCEED_HOURLY_LIMIT\x10\xfbU\x12\x1e\n\x19\x43ONN_EXCEED_MONTHLY_LIMIT\x10\xfcU\x12\x13\n\x0e\x43ONN_THROTTLED\x10\xfdU\x12\x18\n\x13\x43ONN_EXCEEDS_LIMITS\x10\xfeU\x12\x1d\n\x18\x43ONN_INSUFFICIENT_SCOPES\x10\xffU\x12\x15\n\x10\x43ONN_KEY_INVALID\x10\x80V\x12\x17\n\x12\x43ONN_KEY_NOT_FOUND\x10\x81V\x12\x1c\n\x17\x43ONN_BAD_REQUEST_FORMAT\x10\xdcV\x12\x18\n\x13\x43ONN_DOES_NOT_EXIST\x10\xddV\x12\x19\n\x14\x43ONN_INVALID_REQUEST\x10\xdeV\x12\x1c\n\x17\x43ONN_METHOD_NOT_ALLOWED\x10\xdfV\x12\x19\n\x14\x43ONN_NO_GDPR_CONSENT\x10\xe0V\x12\x17\n\x12\x43ONN_INVALID_RANGE\x10\xe1V\x12\x1e\n\x19\x43ONN_AUTH_METHOD_DISABLED\x10\xc0W\x12\x13\n\rMODEL_TRAINED\x10\xec\xa4\x01\x12\x14\n\x0eMODEL_TRAINING\x10\xed\xa4\x01\x12\x15\n\x0fMODEL_UNTRAINED\x10\xee\xa4\x01\x12\x1f\n\x19MODEL_QUEUED_FOR_TRAINING\x10\xef\xa4\x01\x12\x15\n\x0fMODEL_UPLOADING\x10\xf0\xa4\x01\x12\x1c\n\x16MODEL_UPLOADING_FAILED\x10\xf1\xa4\x01\x12\x1b\n\x15MODEL_TRAINING_FAILED\x10\xf2\xa4\x01\x12\x14\n\x0eMODEL_BUILDING\x10\xf3\xa4\x01\x12\x1b\n\x15MODEL_BUILDING_FAILED\x10\xf4\xa4\x01\x12\"\n\x1cMODEL_BUILD_UNEXPECTED_ERROR\x10\xf5\xa4\x01\x12\x1c\n\x16MODEL_TRAINING_NO_DATA\x10\xf6\xa4\x01\x12!\n\x1bMODEL_TRAINING_NO_POSITIVES\x10\xf7\xa4\x01\x12*\n$MODEL_TRAINING_ONE_VS_N_SINGLE_CLASS\x10\xf8\xa4\x01\x12\x1e\n\x18MODEL_TRAINING_TIMED_OUT\x10\xf9\xa4\x01\x12\"\n\x1cMODEL_TRAINING_WAITING_ERROR\x10\xfa\xa4\x01\x12\"\n\x1cMODEL_TRAINING_UNKNOWN_ERROR\x10\xfb\xa4\x01\x12&\n\x1cMODEL_TRAINING_MSG_REDELIVER\x10\xfc\xa4\x01\x1a\x02\x08\x01\x12&\n MODEL_TRAINING_INSUFFICIENT_DATA\x10\xfd\xa4\x01\x12#\n\x1dMODEL_TRAINING_INVALID_PARAMS\x10\xfe\xa4\x01\x12\x34\n.MODEL_TRAINING_INVALID_DATA_TOLERANCE_EXCEEDED\x10\xff\xa4\x01\x12\x1a\n\x14MODEL_MODIFY_SUCCESS\x10\x9e\xa5\x01\x12\x1a\n\x14MODEL_MODIFY_PENDING\x10\x9f\xa5\x01\x12\x19\n\x13MODEL_MODIFY_FAILED\x10\xa0\xa5\x01\x12\x1a\n\x14MODEL_DOES_NOT_EXIST\x10\xd0\xa5\x01\x12\x1d\n\x17MODEL_PERMISSION_DENIED\x10\xd1\xa5\x01\x12\x1c\n\x16MODEL_INVALID_ARGUMENT\x10\xd2\xa5\x01\x12\x1b\n\x15MODEL_INVALID_REQUEST\x10\xd3\xa5\x01\x12\x15\n\x0fMODEL_EVALUATED\x10\xb4\xa6\x01\x12\x16\n\x10MODEL_EVALUATING\x10\xb5\xa6\x01\x12\x19\n\x13MODEL_NOT_EVALUATED\x10\xb6\xa6\x01\x12!\n\x1bMODEL_QUEUED_FOR_EVALUATION\x10\xb7\xa6\x01\x12 \n\x1aMODEL_EVALUATION_TIMED_OUT\x10\xbe\xa6\x01\x12$\n\x1eMODEL_EVALUATION_WAITING_ERROR\x10\xbf\xa6\x01\x12$\n\x1eMODEL_EVALUATION_UNKNOWN_ERROR\x10\xc0\xa6\x01\x12\x1d\n\x17MODEL_PREDICTION_FAILED\x10\xc1\xa6\x01\x12(\n\x1eMODEL_EVALUATION_MSG_REDELIVER\x10\xc2\xa6\x01\x1a\x02\x08\x01\x12\"\n\x1cMODEL_EVALUATION_NEED_LABELS\x10\xc3\xa6\x01\x12\"\n\x1cMODEL_EVALUATION_NEED_INPUTS\x10\xc4\xa6\x01\x12\x1d\n\x17MODEL_EVALUATION_FAILED\x10\xc5\xa6\x01\x12\x1d\n\x17MODEL_DEPLOYMENT_FAILED\x10\xe6\xa6\x01\x12\x15\n\x0fMODEL_DEPLOYING\x10\xe7\xa6\x01\x12!\n\x1bMODEL_QUEUED_FOR_DEPLOYMENT\x10\xe8\xa6\x01\x12\x18\n\x12MODEL_NOT_DEPLOYED\x10\xe9\xa6\x01\x12\x1d\n\x17MODEL_BUSY_PLEASE_RETRY\x10\xea\xa6\x01\x12\x13\n\rMODEL_LOADING\x10\xeb\xa6\x01\x12&\n MODEL_REFERENCE_INVALID_ARGUMENT\x10\x98\xa7\x01\x12*\n$MODEL_EXAMPLE_INPUT_INVALID_ARGUMENT\x10\xac\xa7\x01\x12\x14\n\x0eMODEL_EXPORTED\x10\xfc\xa7\x01\x12\x15\n\x0fMODEL_EXPORTING\x10\xfd\xa7\x01\x12\x1c\n\x16MODEL_EXPORTING_FAILED\x10\xfe\xa7\x01\x12\x1a\n\x14MODEL_EXPORT_PENDING\x10\xff\xa7\x01\x12 \n\x1aWORKFLOW_NO_MATCHING_INPUT\x10\xf1\xab\x01\x12$\n\x1eWORKFLOW_REQUIRE_TRAINED_MODEL\x10\xf2\xab\x01\x12\x18\n\x12WORKFLOW_DUPLICATE\x10\xd4\xac\x01\x12!\n\x1bWORKFLOW_UNSUPPORTED_FORMAT\x10\xd5\xac\x01\x12\x1d\n\x17WORKFLOW_DOES_NOT_EXIST\x10\xd6\xac\x01\x12 \n\x1aWORKFLOW_PERMISSION_DENIED\x10\xd7\xac\x01\x12\x1f\n\x19WORKFLOW_INVALID_ARGUMENT\x10\xd8\xac\x01\x12\x1d\n\x17WORKFLOW_INVALID_RECIPE\x10\xd9\xac\x01\x12\x1f\n\x19WORKFLOW_INVALID_TEMPLATE\x10\xda\xac\x01\x12\x1c\n\x16WORKFLOW_INVALID_GRAPH\x10\xdb\xac\x01\x12\x1f\n\x19WORKFLOW_INTERNAL_FAILURE\x10\xdc\xac\x01\x12\x1e\n\x18WORKFLOW_INVALID_REQUEST\x10\xd7\xb3\x01\x12\x1d\n\x17WORKFLOW_MODIFY_SUCCESS\x10\x86\xad\x01\x12\x1d\n\x17WORKFLOW_MODIFY_PENDING\x10\x87\xad\x01\x12\x1c\n\x16WORKFLOW_MODIFY_FAILED\x10\x88\xad\x01\x12\x1d\n\x17WORKFLOW_REINDEX_FAILED\x10\x89\xad\x01\x12\x1c\n\x16\x43ONCEPT_MODIFY_SUCCESS\x10\xee\xb4\x01\x12\x1c\n\x16\x43ONCEPT_MODIFY_PENDING\x10\xef\xb4\x01\x12\x1b\n\x15\x43ONCEPT_MODIFY_FAILED\x10\xf0\xb4\x01\x12\x18\n\x12\x41NNOTATION_SUCCESS\x10\xd6\xbc\x01\x12\x18\n\x12\x41NNOTATION_PENDING\x10\xd7\xbc\x01\x12\x17\n\x11\x41NNOTATION_FAILED\x10\xd8\xbc\x01\x12\x1f\n\x19\x41NNOTATION_UNKNOWN_STATUS\x10\xda\xbc\x01\x12!\n\x1b\x41NNOTATION_INVALID_ARGUMENT\x10\xdb\xbc\x01\x12\"\n\x1c\x41NNOTATION_PERMISSION_DENIED\x10\xdc\xbc\x01\x12 \n\x1a\x41NNOTATION_AWAITING_REVIEW\x10\xdd\xbc\x01\x12*\n$ANNOTATION_AWAITING_CONSENSUS_REVIEW\x10\xdf\xbc\x01\x12\x1e\n\x18\x41NNOTATION_REVIEW_DENIED\x10\xde\xbc\x01\x12\x1f\n\x19\x41NNOTATION_MODIFY_SUCCESS\x10\xba\xbd\x01\x12\x1f\n\x19\x41NNOTATION_MODIFY_PENDING\x10\xbb\xbd\x01\x12\x1e\n\x18\x41NNOTATION_MODIFY_FAILED\x10\xbc\xbd\x01\x12&\n METADATA_INVALID_PATCH_ARGUMENTS\x10\xc4\xc2\x01\x12\x1c\n\x16METADATA_PARSING_ISSUE\x10\xc5\xc2\x01\x12!\n\x1bMETADATA_MANIPULATION_ISSUE\x10\xc6\xc2\x01\x12\x1c\n\x16TRAINER_JOB_STATE_NONE\x10\xa8\xc3\x01\x12\x1e\n\x18TRAINER_JOB_STATE_QUEUED\x10\xa9\xc3\x01\x12\x1f\n\x19TRAINER_JOB_STATE_RUNNING\x10\xaa\xc3\x01\x12 \n\x1aTRAINER_JOB_STATE_COMPLETE\x10\xab\xc3\x01\x12\x1d\n\x17TRAINER_JOB_STATE_ERROR\x10\xac\xc3\x01\x12\x17\n\x11\x44\x41TA_DUMP_SUCCESS\x10\xbe\xc4\x01\x12\x17\n\x11\x44\x41TA_DUMP_PENDING\x10\xbf\xc4\x01\x12\x16\n\x10\x44\x41TA_DUMP_FAILED\x10\xc0\xc4\x01\x12\x1b\n\x15\x44\x41TA_DUMP_IN_PROGRESS\x10\xc1\xc4\x01\x12\x1b\n\x11\x44\x41TA_DUMP_NO_DATA\x10\xc2\xc4\x01\x1a\x02\x08\x01\x12 \n\x1a\x44\x41TA_DUMP_UNEXPECTED_ERROR\x10\xc3\xc4\x01\x12\x1e\n\x18\x44\x41TA_DUMP_EXPORT_SUCCESS\x10\xd2\xc4\x01\x12\x1e\n\x18\x44\x41TA_DUMP_EXPORT_PENDING\x10\xd3\xc4\x01\x12\x1d\n\x17\x44\x41TA_DUMP_EXPORT_FAILED\x10\xd4\xc4\x01\x12\"\n\x1c\x44\x41TA_DUMP_EXPORT_IN_PROGRESS\x10\xd5\xc4\x01\x12\'\n!DATA_DUMP_EXPORT_UNEXPECTED_ERROR\x10\xd6\xc4\x01\x12\x1d\n\x17\x41PP_DUPLICATION_SUCCESS\x10\xf0\xc4\x01\x12\x1c\n\x16\x41PP_DUPLICATION_FAILED\x10\xf1\xc4\x01\x12\x1d\n\x17\x41PP_DUPLICATION_PENDING\x10\xf2\xc4\x01\x12!\n\x1b\x41PP_DUPLICATION_IN_PROGRESS\x10\xf3\xc4\x01\x12%\n\x1f\x41PP_DUPLICATION_INVALID_REQUEST\x10\xf4\xc4\x01\x12\x1b\n\x15MODULE_DOES_NOT_EXIST\x10\xd4\xc5\x01\x12\x1e\n\x18MODULE_PERMISSION_DENIED\x10\xd5\xc5\x01\x12\x1d\n\x17MODULE_INVALID_ARGUMENT\x10\xd6\xc5\x01\x12\x1c\n\x16MODULE_INVALID_REQUEST\x10\xd7\xc5\x01\x12\x1c\n\x16\x42ULK_OPERATION_SUCCESS\x10\xb8\xc6\x01\x12\x1b\n\x15\x42ULK_OPERATION_FAILED\x10\xb9\xc6\x01\x12\x1c\n\x16\x42ULK_OPERATION_PENDING\x10\xba\xc6\x01\x12 \n\x1a\x42ULK_OPERATION_IN_PROGRESS\x10\xbb\xc6\x01\x12$\n\x1e\x42ULK_OPERATION_INVALID_REQUEST\x10\xbc\xc6\x01\x12\x1e\n\x18\x42ULK_OPERATION_CANCELLED\x10\xbd\xc6\x01\x12%\n\x1f\x42ULK_OPERATION_UNEXPECTED_ERROR\x10\xbe\xc6\x01\x12\x1b\n\x15RUNNER_DOES_NOT_EXIST\x10\x80\xc8\x01\x12\x1e\n\x18RUNNER_PERMISSION_DENIED\x10\x81\xc8\x01\x12\x1d\n\x17RUNNER_INVALID_ARGUMENT\x10\x82\xc8\x01\x12\x1c\n\x16RUNNER_INVALID_REQUEST\x10\x83\xc8\x01\x12\x18\n\x12RUNNER_NEEDS_RETRY\x10\x84\xc8\x01\x12\x19\n\x13RUNNER_STREAM_START\x10\x85\xc8\x01\x12\x17\n\x11RUNNER_STREAM_END\x10\x86\xc8\x01\x12\x1b\n\x15RUNNER_ITEM_CANCELLED\x10\x87\xc8\x01\x12\x1e\n\x18RUNNER_PROCESSING_FAILED\x10\x88\xc8\x01\x12\x1d\n\x17NODEPOOL_DOES_NOT_EXIST\x10\xe4\xc8\x01\x12\x1f\n\x19NODEPOOL_INVALID_ARGUMENT\x10\xe5\xc8\x01\x12\x1e\n\x18NODEPOOL_INVALID_REQUEST\x10\xe6\xc8\x01\x12$\n\x1e\x43OMPUTE_CLUSTER_DOES_NOT_EXIST\x10\xc8\xc9\x01\x12&\n COMPUTE_CLUSTER_INVALID_ARGUMENT\x10\xc9\xc9\x01\x12%\n\x1f\x43OMPUTE_CLUSTER_INVALID_REQUEST\x10\xca\xc9\x01\x12\x1f\n\x19\x44\x45PLOYMENT_DOES_NOT_EXIST\x10\xac\xca\x01\x12!\n\x1b\x44\x45PLOYMENT_INVALID_ARGUMENT\x10\xad\xca\x01\x12 \n\x1a\x44\x45PLOYMENT_INVALID_REQUEST\x10\xae\xca\x01\x12\"\n\x1cINSTANCE_TYPE_DOES_NOT_EXIST\x10\x90\xcb\x01\x12$\n\x1eINSTANCE_TYPE_INVALID_ARGUMENT\x10\x91\xcb\x01\x12#\n\x1dINSTANCE_TYPE_INVALID_REQUEST\x10\x92\xcb\x01\x12+\n%COMPUTE_PLANE_METRICS_INVALID_REQUEST\x10\xf4\xcb\x01\x12\x13\n\rINPUT_SUCCESS\x10\xb0\xea\x01\x12\x13\n\rINPUT_PENDING\x10\xb1\xea\x01\x12\x12\n\x0cINPUT_FAILED\x10\xb2\xea\x01\x12\x17\n\x11INPUT_IN_PROGRESS\x10\xb3\xea\x01\x12 \n\x16INPUT_DOWNLOAD_SUCCESS\x10\xb0\xea\x01\x1a\x02\x08\x01\x12 \n\x16INPUT_DOWNLOAD_PENDING\x10\xb1\xea\x01\x1a\x02\x08\x01\x12\x1f\n\x15INPUT_DOWNLOAD_FAILED\x10\xb2\xea\x01\x1a\x02\x08\x01\x12$\n\x1aINPUT_DOWNLOAD_IN_PROGRESS\x10\xb3\xea\x01\x1a\x02\x08\x01\x12 \n\x1aINPUT_STATUS_UPDATE_FAILED\x10\xb4\xea\x01\x12\x19\n\x13INPUT_DELETE_FAILED\x10\xb5\xea\x01\x12\x15\n\x0fINPUT_DUPLICATE\x10\x94\xeb\x01\x12\x1e\n\x18INPUT_UNSUPPORTED_FORMAT\x10\x95\xeb\x01\x12\x1a\n\x14INPUT_DOES_NOT_EXIST\x10\x96\xeb\x01\x12\x1d\n\x17INPUT_PERMISSION_DENIED\x10\x97\xeb\x01\x12\x1c\n\x16INPUT_INVALID_ARGUMENT\x10\x98\xeb\x01\x12\x16\n\x10INPUT_OVER_LIMIT\x10\x99\xeb\x01\x12\x17\n\x11INPUT_INVALID_URL\x10\x9a\xeb\x01\x12\x1a\n\x14INPUT_MODIFY_SUCCESS\x10\xf8\xeb\x01\x12\x1a\n\x14INPUT_MODIFY_PENDING\x10\xf9\xeb\x01\x12\x19\n\x13INPUT_MODIFY_FAILED\x10\xfb\xeb\x01\x12\x1f\n\x19INPUT_STORAGE_HOST_FAILED\x10\x82\xec\x01\x12\x1d\n\x17\x41LL_INPUT_INVALID_BYTES\x10\xdc\xec\x01\x12\x1b\n\x15INPUT_CLUSTER_SUCCESS\x10\xc0\xed\x01\x12\x1b\n\x15INPUT_CLUSTER_PENDING\x10\xc1\xed\x01\x12\x1a\n\x14INPUT_CLUSTER_FAILED\x10\xc2\xed\x01\x12\x1f\n\x19INPUT_CLUSTER_IN_PROGRESS\x10\xc3\xed\x01\x12\x1b\n\x15INPUT_REINDEX_SUCCESS\x10\xa4\xee\x01\x12\x1b\n\x15INPUT_REINDEX_PENDING\x10\xa5\xee\x01\x12\x1a\n\x14INPUT_REINDEX_FAILED\x10\xa6\xee\x01\x12\x1f\n\x19INPUT_REINDEX_IN_PROGRESS\x10\xa7\xee\x01\x12\"\n\x1cINPUT_VIDEO_DOWNLOAD_SUCCESS\x10\x98\xf2\x01\x12\"\n\x1cINPUT_VIDEO_DOWNLOAD_PENDING\x10\x99\xf2\x01\x12!\n\x1bINPUT_VIDEO_DOWNLOAD_FAILED\x10\x9a\xf2\x01\x12\x1b\n\x15INPUT_VIDEO_DUPLICATE\x10\xfc\xf2\x01\x12$\n\x1eINPUT_VIDEO_UNSUPPORTED_FORMAT\x10\xfd\xf2\x01\x12 \n\x1aINPUT_VIDEO_DOES_NOT_EXIST\x10\xfe\xf2\x01\x12#\n\x1dINPUT_VIDEO_PERMISSION_DENIED\x10\xff\xf2\x01\x12\"\n\x1cINPUT_VIDEO_INVALID_ARGUMENT\x10\x80\xf3\x01\x12\x1c\n\x16INPUT_VIDEO_OVER_LIMIT\x10\x81\xf3\x01\x12\x1d\n\x17INPUT_VIDEO_INVALID_URL\x10\x82\xf3\x01\x12 \n\x1aINPUT_VIDEO_MODIFY_SUCCESS\x10\xe0\xf3\x01\x12 \n\x1aINPUT_VIDEO_MODIFY_PENDING\x10\xe1\xf3\x01\x12\x1f\n\x19INPUT_VIDEO_MODIFY_FAILED\x10\xe3\xf3\x01\x12%\n\x1fINPUT_VIDEO_STORAGE_HOST_FAILED\x10\xea\xf3\x01\x12$\n\x1e\x41LL_INPUT_VIDEOS_INVALID_BYTES\x10\xc4\xf4\x01\x12$\n\x1eINPUT_VIDEO_PROCESSING_SUCCESS\x10\xa8\xf5\x01\x12$\n\x1eINPUT_VIDEO_PROCESSING_PENDING\x10\xb2\xf5\x01\x12#\n\x1dINPUT_VIDEO_PROCESSING_FAILED\x10\xbc\xf5\x01\x12\'\n!INPUT_VIDEO_STORAGE_INCONSISTENCY\x10\xc6\xf5\x01\x12!\n\x1bINPUT_VIDEO_STORAGE_FAILURE\x10\xd0\xf5\x01\x12(\n\"INPUT_VIDEO_URL_GENERATION_FAILURE\x10\xda\xf5\x01\x12\x1d\n\x17INPUT_CONNECTION_FAILED\x10\xbc\xb8\x02\x12&\n REQUEST_DISABLED_FOR_MAINTENANCE\x10\xbd\xb8\x02\x12/\n%INPUT_WRITES_DISABLED_FOR_MAINTENANCE\x10\xbe\xb8\x02\x1a\x02\x08\x01\x12\x1b\n\x15INPUT_INVALID_REQUEST\x10\xbf\xb8\x02\x12\x1d\n\x17PREDICT_INVALID_REQUEST\x10\xc1\xb8\x02\x12\x1c\n\x16SEARCH_INVALID_REQUEST\x10\xc2\xb8\x02\x12\x1e\n\x18\x43ONCEPTS_INVALID_REQUEST\x10\xc3\xb8\x02\x12\x1b\n\x15STATS_INVALID_REQUEST\x10\xc4\xb8\x02\x12\x1c\n\x16\x44\x41TABASE_DUPLICATE_KEY\x10\xca\xb8\x02\x12 \n\x1a\x44\x41TABASE_STATEMENT_TIMEOUT\x10\xcb\xb8\x02\x12$\n\x1e\x44\x41TABASE_INVALID_ROWS_AFFECTED\x10\xcc\xb8\x02\x12 \n\x1a\x44\x41TABASE_DEADLOCK_DETECTED\x10\xcd\xb8\x02\x12\x18\n\x12\x44\x41TABASE_FAIL_TASK\x10\xce\xb8\x02\x12&\n DATABASE_FAIL_TO_GET_CONNECTIONS\x10\xcf\xb8\x02\x12\x1f\n\x19\x44\x41TABASE_TOO_MANY_CLIENTS\x10\xd0\xb8\x02\x12\"\n\x1c\x44\x41TABASE_CONSTRAINT_VIOLATED\x10\xd1\xb8\x02\x12\x17\n\x11\x44\x41TABASE_CANCELED\x10\xd5\xb8\x02\x12\x1f\n\x19\x41SYNC_WORKER_MULTI_ERRORS\x10\xd4\xb8\x02\x12\x1c\n\x16RPC_REQUEST_QUEUE_FULL\x10\xde\xb8\x02\x12\x1c\n\x16RPC_SERVER_UNAVAILABLE\x10\xdf\xb8\x02\x12\x19\n\x13RPC_REQUEST_TIMEOUT\x10\xe0\xb8\x02\x12#\n\x1dRPC_MAX_MESSAGE_SIZE_EXCEEDED\x10\xe1\xb8\x02\x12\x12\n\x0cRPC_CANCELED\x10\xe3\xb8\x02\x12\x18\n\x12RPC_UNKNOWN_METHOD\x10\xe4\xb8\x02\x12\x1e\n\x18REQUEST_CANCELED_BY_USER\x10\xe5\xb8\x02\x12\x1e\n\x18\x43LUSTER_INTERNAL_FAILURE\x10\xa0\xd0\x02\x12\x1f\n\x19\x45XTERNAL_CONNECTION_ERROR\x10\xe2\xb8\x02\x12\x1a\n\x14QUERY_INVALID_SYNTAX\x10\xf2\xb8\x02\x12\x16\n\x10QUEUE_CONN_ERROR\x10\xa8\xc0\x02\x12!\n\x1bQUEUE_CLOSE_REQUEST_TIMEOUT\x10\xaa\xc0\x02\x12\x17\n\x11QUEUE_CONN_CLOSED\x10\xab\xc0\x02\x12\x1f\n\x19QUEUE_PUBLISH_ACK_TIMEOUT\x10\xac\xc0\x02\x12\x19\n\x13QUEUE_PUBLISH_ERROR\x10\xad\xc0\x02\x12 \n\x1aQUEUE_SUBSCRIPTION_TIMEOUT\x10\xae\xc0\x02\x12\x1e\n\x18QUEUE_SUBSCRIPTION_ERROR\x10\xaf\xc0\x02\x12\x1e\n\x18QUEUE_MARSHALLING_FAILED\x10\xb0\xc0\x02\x12 \n\x1aQUEUE_UNMARSHALLING_FAILED\x10\xb1\xc0\x02\x12\'\n!QUEUE_MAX_MSG_REDELIVERY_EXCEEDED\x10\xb2\xc0\x02\x12\x17\n\x11QUEUE_ACK_FAILURE\x10\xb3\xc0\x02\x12\x13\n\rSQS_OVERLIMIT\x10\x8c\xc1\x02\x12 \n\x1aSQS_INVALID_RECEIPT_HANDLE\x10\x8d\xc1\x02\x12\x11\n\x0bSQS_UNKNOWN\x10\x8e\xc1\x02\x12\x1d\n\x17SEARCH_INTERNAL_FAILURE\x10\xf9\xcf\x02\x12\x1f\n\x19SEARCH_PROJECTION_FAILURE\x10\xfa\xcf\x02\x12\x1f\n\x19SEARCH_PREDICTION_FAILURE\x10\xfb\xcf\x02\x12\'\n!SEARCH_BY_NOT_FULLY_INDEXED_INPUT\x10\xfc\xcf\x02\x12 \n\x1aSAVED_SEARCH_MODIFY_FAILED\x10\xfd\xcf\x02\x12\x1f\n\x19SEARCH_COUNTS_UNAVAILABLE\x10\xfe\xcf\x02\x12\x17\n\x11\x45VALUATION_QUEUED\x10\xdc\xd0\x02\x12\x1c\n\x16\x45VALUATION_IN_PROGRESS\x10\xdd\xd0\x02\x12\x18\n\x12\x45VALUATION_SUCCESS\x10\xde\xd0\x02\x12(\n\"EVALUATION_FAILED_TO_RETRIEVE_DATA\x10\xdf\xd0\x02\x12!\n\x1b\x45VALUATION_INVALID_ARGUMENT\x10\xe0\xd0\x02\x12\x17\n\x11\x45VALUATION_FAILED\x10\xe1\xd0\x02\x12\x18\n\x12\x45VALUATION_PENDING\x10\xe2\xd0\x02\x12\x1a\n\x14\x45VALUATION_TIMED_OUT\x10\xe3\xd0\x02\x12!\n\x1b\x45VALUATION_UNEXPECTED_ERROR\x10\xe4\xd0\x02\x12\x16\n\x10\x45VALUATION_MIXED\x10\xe5\xd0\x02\x12\x18\n\x12STRIPE_EVENT_ERROR\x10\xe1\xd7\x02\x12\x10\n\nCACHE_MISS\x10\xc9\xdf\x02\x12&\n REDIS_SCRIPT_EXITED_WITH_FAILURE\x10\xca\xdf\x02\x12\x16\n\x10REDIS_STREAM_ERR\x10\xcb\xdf\x02\x12\x18\n\x12REDIS_NO_CONSUMERS\x10\xcc\xdf\x02\x12\x1a\n\x14REDIS_STREAM_BACKOFF\x10\xcd\xdf\x02\x12\x18\n\x12SIGNUP_EVENT_ERROR\x10\xb1\xe7\x02\x12\x14\n\x0eSIGNUP_FLAGGED\x10\xb2\xe7\x02\x12\x1a\n\x14\x46ILETYPE_UNSUPPORTED\x10\xb3\xe7\x02\x12\x1f\n\x19\x41PP_COUNT_INVALID_MESSAGE\x10\x99\xef\x02\x12\'\n!APP_COUNT_UPDATE_INCREMENT_FAILED\x10\x9a\xef\x02\x12\x1e\n\x18\x41PP_COUNT_REBUILD_FAILED\x10\x9b\xef\x02\x12 \n\x1a\x41PP_COUNT_INTERNAL_FAILURE\x10\x9c\xef\x02\x12\x17\n\x11MP_DOWNLOAD_ERROR\x10\xfd\xef\x02\x12\x1a\n\x14MP_RESOLVE_DNS_ERROR\x10\xfe\xef\x02\x12)\n#MP_DOWNLOAD_MAX_SIZE_EXCEEDED_ERROR\x10\xff\xef\x02\x12\x1b\n\x15MP_IMAGE_DECODE_ERROR\x10\x80\xf0\x02\x12\x19\n\x13MP_INVALID_ARGUMENT\x10\x81\xf0\x02\x12\x1f\n\x19MP_IMAGE_PROCESSING_ERROR\x10\x82\xf0\x02\x12\x19\n\x13\x44\x41TATIER_CONN_ERROR\x10\xe1\xf0\x02\x12\x17\n\x11USER_CONSENT_FACE\x10\xd1\x86\x03\x12\x14\n\x0eWORKER_MISSING\x10\xb8\x8e\x03\x12\x13\n\rWORKER_ACTIVE\x10\xb9\x8e\x03\x12\x15\n\x0fWORKER_INACTIVE\x10\xba\x8e\x03\x12\x17\n\x11\x43OLLECTOR_MISSING\x10\xa0\x96\x03\x12\x16\n\x10\x43OLLECTOR_ACTIVE\x10\xa1\x96\x03\x12\x18\n\x12\x43OLLECTOR_INACTIVE\x10\xa2\x96\x03\x12!\n\x1b\x43OLLECTOR_POST_INPUT_FAILED\x10\xa3\x96\x03\x12*\n$SSO_IDENTITY_PROVIDER_DOES_NOT_EXIST\x10\x89\x9e\x03\x12\x16\n\x10TASK_IN_PROGRESS\x10\xf1\xa5\x03\x12\x0f\n\tTASK_DONE\x10\xf2\xa5\x03\x12\x12\n\x0cTASK_WONT_DO\x10\xf3\xa5\x03\x12\x11\n\x0bTASK_FAILED\x10\xf5\xa5\x03\x12\x0f\n\tTASK_IDLE\x10\xf6\xa5\x03\x12\x13\n\rTASK_CONFLICT\x10\xd4\xa6\x03\x12\x1a\n\x14TASK_NOT_IMPLEMENTED\x10\xd5\xa6\x03\x12\x12\n\x0cTASK_MISSING\x10\xd6\xa6\x03\x12\x1c\n\x16TASK_PERMISSION_DENIED\x10\xd7\xa6\x03\x12\x1d\n\x17TASK_ASSIGNMENT_SUCCESS\x10\xb8\xa7\x03\x12\x1d\n\x17TASK_ASSIGNMENT_PENDING\x10\xb9\xa7\x03\x12%\n\x1fTASK_ASSIGNMENT_AWAITING_REVIEW\x10\xba\xa7\x03\x12/\n)TASK_ASSIGNMENT_AWAITING_CONSENSUS_REVIEW\x10\xbb\xa7\x03\x12\x1e\n\x18TASK_ASSIGNMENT_REJECTED\x10\xbc\xa7\x03\x12$\n\x1eTASK_ASSIGNMENT_REVIEW_SUCCESS\x10\x9c\xa8\x03\x12$\n\x1eTASK_ASSIGNMENT_REVIEW_PENDING\x10\x9d\xa8\x03\x12&\n TASK_ASSIGNMENT_REVIEW_DISMISSED\x10\x9e\xa8\x03\x12\x19\n\x13LABEL_ORDER_PENDING\x10\xd9\xad\x03\x12\x1d\n\x17LABEL_ORDER_IN_PROGRESS\x10\xda\xad\x03\x12\x19\n\x13LABEL_ORDER_SUCCESS\x10\xdb\xad\x03\x12\x1a\n\x14LABEL_ORDER_CANCELED\x10\xdc\xad\x03\x12\x14\n\x0eLICENSE_ACTIVE\x10\xe0\xd4\x03\x12\x1c\n\x16LICENSE_DOES_NOT_EXIST\x10\xe1\xd4\x03\x12\x19\n\x13LICENSE_NEED_UPDATE\x10\xe2\xd4\x03\x12\x15\n\x0fLICENSE_EXPIRED\x10\xe3\xd4\x03\x12\x15\n\x0fLICENSE_REVOKED\x10\xe4\xd4\x03\x12\x15\n\x0fLICENSE_DELETED\x10\xe5\xd4\x03\x12\x1d\n\x17LICENSE_VOLUME_EXCEEDED\x10\xe6\xd4\x03\x12!\n\x1bPASSWORD_VALIDATION_SUCCESS\x10\xc8\xdc\x03\x12 \n\x1aPASSWORD_VALIDATION_FAILED\x10\xc9\xdc\x03\x12%\n\x1fPASSWORDPOLICY_INVALID_ARGUMENT\x10\xca\xdc\x03\x12\"\n\x1c\x46\x45\x41TUREFLAG_CONFIG_NOT_FOUND\x10\xb0\xe4\x03\x12\"\n\x1c\x46\x45\x41TUREFLAG_INVALID_ARGUMENT\x10\xb1\xe4\x03\x12\x19\n\x13\x46\x45\x41TUREFLAG_BLOCKED\x10\xb2\xe4\x03\x12\x1b\n\x15\x46\x45\x41TUREFLAG_NOT_FOUND\x10\xb3\xe4\x03\x12\x19\n\x13MAINTENANCE_SUCCESS\x10\x98\xec\x03\x12\x18\n\x12MAINTENANCE_FAILED\x10\x99\xec\x03\x12\x1d\n\x17\x44\x41TASET_VERSION_PENDING\x10\x85\xf4\x03\x12!\n\x1b\x44\x41TASET_VERSION_IN_PROGRESS\x10\x8a\xf4\x03\x12\x1b\n\x15\x44\x41TASET_VERSION_READY\x10\x8f\xf4\x03\x12\x1d\n\x17\x44\x41TASET_VERSION_FAILURE\x10\x94\xf4\x03\x12&\n DATASET_VERSION_UNEXPECTED_ERROR\x10\x99\xf4\x03\x12\x1e\n\x18\x44\x41TASET_VERSION_CONFLICT\x10\x9e\xf4\x03\x12\x1b\n\x15\x44\x41TASET_INPUT_SUCCESS\x10\xe4\xf4\x03\x12\x1d\n\x17\x44\x41TASET_INPUT_DUPLICATE\x10\xe5\xf4\x03\x12$\n\x1e\x44\x41TASET_VERSION_EXPORT_SUCCESS\x10\xc8\xf5\x03\x12$\n\x1e\x44\x41TASET_VERSION_EXPORT_PENDING\x10\xc9\xf5\x03\x12#\n\x1d\x44\x41TASET_VERSION_EXPORT_FAILED\x10\xca\xf5\x03\x12(\n\"DATASET_VERSION_EXPORT_IN_PROGRESS\x10\xcb\xf5\x03\x12-\n\'DATASET_VERSION_EXPORT_UNEXPECTED_ERROR\x10\xcc\xf5\x03\x12\x10\n\nJOB_QUEUED\x10\x80\xf4\x03\x12\x11\n\x0bJOB_RUNNING\x10\x81\xf4\x03\x12\x13\n\rJOB_COMPLETED\x10\x82\xf4\x03\x12\x10\n\nJOB_FAILED\x10\x83\xf4\x03\x12\x13\n\rJOB_CANCELLED\x10\x84\xf4\x03\x12\x1a\n\x14JOB_UNEXPECTED_ERROR\x10\x86\xf4\x03\x12\x12\n\x0cJOB_CONFLICT\x10\x87\xf4\x03\x12\x1c\n\x16\x41UTH_MISSING_IDP_ASSOC\x10\xe8\xfb\x03\x12\x19\n\x13LIST_OBJECTS_FAILED\x10\xd0\x83\x04\x12\x1c\n\x16\x41RCHIVE_EXTRACT_FAILED\x10\xb8\x8b\x04\x12\x18\n\x12UPLOAD_IN_PROGRESS\x10\xa0\x93\x04\x12\x11\n\x0bUPLOAD_DONE\x10\xa1\x93\x04\x12\x13\n\rUPLOAD_FAILED\x10\xa2\x93\x04\x12\x1d\n\x17UPLOAD_UNEXPECTED_ERROR\x10\xa3\x93\x04\x12\x14\n\x0eUPLOAD_EXPIRED\x10\xa4\x93\x04\x12\x15\n\x0fUPLOAD_CANCELED\x10\xa5\x93\x04\x12\x15\n\x0fUPLOAD_CONFLICT\x10\xa6\x93\x04\x12\x1a\n\x14\x42ILLING_INVALID_INFO\x10\x88\x9b\x04\x12!\n\x1bLOG_ENTRIES_INVALID_REQUEST\x10\xf0\xa2\x04\x12\x1b\n\x15INTERNAL_SERVER_ISSUE\x10\xd4\xfd\x05\x12\x1d\n\x17INTERNAL_FETCHING_ISSUE\x10\xd5\xfd\x05\x12\x1d\n\x17INTERNAL_DATABASE_ISSUE\x10\xd6\xfd\x05\x12\x1f\n\x19INTERNAL_CONTEXT_CANCELED\x10\xd8\xfd\x05\x12!\n\x1bINTERNAL_UNEXPECTED_TIMEOUT\x10\xd9\xfd\x05\x12\x1c\n\x16INTERNAL_UNEXPECTED_V1\x10\xda\xfd\x05\x12\x1f\n\x19INTERNAL_UNEXPECTED_PANIC\x10\xdb\xfd\x05\x12\x1f\n\x19INTERNAL_UNEXPECTED_SPIRE\x10\xdc\xfd\x05\x12 \n\x1aINTERNAL_REDIS_UNAVAILABLE\x10\xdd\xfd\x05\x12!\n\x1bINTERNAL_RESOURCE_EXHAUSTED\x10\xde\xfd\x05\x12\"\n\x1cINTERNAL_REDIS_UNCATEGORIZED\x10\xdf\xfd\x05\x12 \n\x1aINTERNAL_AWS_UNCATEGORIZED\x10\xe0\xfd\x05\x12\"\n\x1cINTERNAL_AZURE_UNCATEGORIZED\x10\xe1\xfd\x05\x12%\n\x1fINTERNAL_VECTORDB_UNCATEGORIZED\x10\xe2\xfd\x05\x12\x18\n\x12\x43ONN_UNCATEGORIZED\x10\xb9\x85\x06\x12\x19\n\x13MODEL_UNCATEGORIZED\x10\xba\x85\x06\x12\x19\n\x13INPUT_UNCATEGORIZED\x10\xbb\x85\x06\x12\x1e\n\x18\x41NNOTATION_UNCATEGORIZED\x10\xbc\x85\x06\x12\x1b\n\x15\x42ILLING_UNCATEGORIZED\x10\xbd\x85\x06\x12\x1c\n\x16INTERNAL_UNCATEGORIZED\x10\xc1\x85\x06\x12\x11\n\x0b\x42\x41\x44_REQUEST\x10\xa0\xc2\x05\x12\x12\n\x0cSERVER_ERROR\x10\x84\xc3\x05\x1a\x02\x10\x01\"\x08\x08\xbf\xc6\x01\x10\xbf\xc6\x01\"\x08\x08\xe8\x81\x02\x10\xe8\x81\x02\"\x08\x08\xe9\x81\x02\x10\xe9\x81\x02\"\x08\x08\xea\x81\x02\x10\xea\x81\x02\"\x08\x08\xcc\x82\x02\x10\xcc\x82\x02\"\x08\x08\xcd\x82\x02\x10\xcd\x82\x02\"\x08\x08\xce\x82\x02\x10\xce\x82\x02\"\x08\x08\xcf\x82\x02\x10\xcf\x82\x02\"\x08\x08\xd0\x82\x02\x10\xd0\x82\x02\"\x08\x08\xd1\x82\x02\x10\xd1\x82\x02\"\x08\x08\xd2\x82\x02\x10\xd2\x82\x02\"\x08\x08\xb0\x83\x02\x10\xb0\x83\x02\"\x08\x08\xb1\x83\x02\x10\xb1\x83\x02\"\x08\x08\xb3\x83\x02\x10\xb3\x83\x02\"\x08\x08\xba\x83\x02\x10\xba\x83\x02\"\x08\x08\xbb\xb8\x02\x10\xbb\xb8\x02\"\x08\x08\xd2\xb8\x02\x10\xd2\xb8\x02\"\x08\x08\xd3\xb8\x02\x10\xd3\xb8\x02\"\x08\x08\xf0\xc1\x02\x10\xf0\xc1\x02\"\x08\x08\xf1\xc1\x02\x10\xf1\xc1\x02\"\x08\x08\xf2\xc1\x02\x10\xf2\xc1\x02\"\x08\x08\xf3\xc1\x02\x10\xf3\xc1\x02\"\x08\x08\xf4\xc1\x02\x10\xf4\xc1\x02\"\x08\x08\x9c\xc7\x01\x10\x9c\xc7\x01\"\x08\x08\x9d\xc7\x01\x10\x9d\xc7\x01\"\x08\x08\x9e\xc7\x01\x10\x9e\xc7\x01\"\x08\x08\x9f\xc7\x01\x10\x9f\xc7\x01\"\x08\x08\xa1\xc7\x01\x10\xa1\xc7\x01\"\x08\x08\xa2\xc7\x01\x10\xa2\xc7\x01\x42g\n\x1c\x63om.clarifai.grpc.api.statusP\x01Z>github.com/Clarifai/clarifai-go-grpc/proto/clarifai/api/status\xa2\x02\x04\x43\x41IPb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'proto.clarifai.api.status.status_code_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\034com.clarifai.grpc.api.statusP\001Z>github.com/Clarifai/clarifai-go-grpc/proto/clarifai/api/status\242\002\004CAIP'
  _STATUSCODE._options = None
  _STATUSCODE._serialized_options = b'\020\001'
  _STATUSCODE.values_by_name["CONN_TOKEN_INVALID"]._options = None
  _STATUSCODE.values_by_name["CONN_TOKEN_INVALID"]._serialized_options = b'\010\001'
  _STATUSCODE.values_by_name["MODEL_TRAINING_MSG_REDELIVER"]._options = None
  _STATUSCODE.values_by_name["MODEL_TRAINING_MSG_REDELIVER"]._serialized_options = b'\010\001'
  _STATUSCODE.values_by_name["MODEL_EVALUATION_MSG_REDELIVER"]._options = None
  _STATUSCODE.values_by_name["MODEL_EVALUATION_MSG_REDELIVER"]._serialized_options = b'\010\001'
  _STATUSCODE.values_by_name["DATA_DUMP_NO_DATA"]._options = None
  _STATUSCODE.values_by_name["DATA_DUMP_NO_DATA"]._serialized_options = b'\010\001'
  _STATUSCODE.values_by_name["INPUT_DOWNLOAD_SUCCESS"]._options = None
  _STATUSCODE.values_by_name["INPUT_DOWNLOAD_SUCCESS"]._serialized_options = b'\010\001'
  _STATUSCODE.values_by_name["INPUT_DOWNLOAD_PENDING"]._options = None
  _STATUSCODE.values_by_name["INPUT_DOWNLOAD_PENDING"]._serialized_options = b'\010\001'
  _STATUSCODE.values_by_name["INPUT_DOWNLOAD_FAILED"]._options = None
  _STATUSCODE.values_by_name["INPUT_DOWNLOAD_FAILED"]._serialized_options = b'\010\001'
  _STATUSCODE.values_by_name["INPUT_DOWNLOAD_IN_PROGRESS"]._options = None
  _STATUSCODE.values_by_name["INPUT_DOWNLOAD_IN_PROGRESS"]._serialized_options = b'\010\001'
  _STATUSCODE.values_by_name["INPUT_WRITES_DISABLED_FOR_MAINTENANCE"]._options = None
  _STATUSCODE.values_by_name["INPUT_WRITES_DISABLED_FOR_MAINTENANCE"]._serialized_options = b'\010\001'
  _STATUSCODE._serialized_start=69
  _STATUSCODE._serialized_end=12352
# @@protoc_insertion_point(module_scope)
