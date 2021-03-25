# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    EnumDescriptor as google___protobuf___descriptor___EnumDescriptor,
    FileDescriptor as google___protobuf___descriptor___FileDescriptor,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    List as typing___List,
    NewType as typing___NewType,
    Tuple as typing___Tuple,
    cast as typing___cast,
)


builtin___int = int
builtin___str = str


DESCRIPTOR: google___protobuf___descriptor___FileDescriptor = ...

StatusCodeValue = typing___NewType('StatusCodeValue', builtin___int)
type___StatusCodeValue = StatusCodeValue
class StatusCode(object):
    DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
    @classmethod
    def Name(cls, number: builtin___int) -> builtin___str: ...
    @classmethod
    def Value(cls, name: builtin___str) -> StatusCodeValue: ...
    @classmethod
    def keys(cls) -> typing___List[builtin___str]: ...
    @classmethod
    def values(cls) -> typing___List[StatusCodeValue]: ...
    @classmethod
    def items(cls) -> typing___List[typing___Tuple[builtin___str, StatusCodeValue]]: ...
    ZERO = typing___cast(StatusCodeValue, 0)
    SUCCESS = typing___cast(StatusCodeValue, 10000)
    MIXED_STATUS = typing___cast(StatusCodeValue, 10010)
    FAILURE = typing___cast(StatusCodeValue, 10020)
    TRY_AGAIN = typing___cast(StatusCodeValue, 10030)
    NOT_IMPLEMENTED = typing___cast(StatusCodeValue, 10040)
    CONN_ACCOUNT_ISSUES = typing___cast(StatusCodeValue, 11000)
    CONN_TOKEN_INVALID = typing___cast(StatusCodeValue, 11001)
    CONN_CREDENTIALS_INVALID = typing___cast(StatusCodeValue, 11002)
    CONN_EXCEED_HOURLY_LIMIT = typing___cast(StatusCodeValue, 11003)
    CONN_EXCEED_MONTHLY_LIMIT = typing___cast(StatusCodeValue, 11004)
    CONN_THROTTLED = typing___cast(StatusCodeValue, 11005)
    CONN_EXCEEDS_LIMITS = typing___cast(StatusCodeValue, 11006)
    CONN_INSUFFICIENT_SCOPES = typing___cast(StatusCodeValue, 11007)
    CONN_KEY_INVALID = typing___cast(StatusCodeValue, 11008)
    CONN_KEY_NOT_FOUND = typing___cast(StatusCodeValue, 11009)
    CONN_BAD_REQUEST_FORMAT = typing___cast(StatusCodeValue, 11100)
    CONN_DOES_NOT_EXIST = typing___cast(StatusCodeValue, 11101)
    CONN_INVALID_REQUEST = typing___cast(StatusCodeValue, 11102)
    CONN_METHOD_NOT_ALLOWED = typing___cast(StatusCodeValue, 11103)
    CONN_NO_GDPR_CONSENT = typing___cast(StatusCodeValue, 11104)
    CONN_AUTH_METHOD_DISABLED = typing___cast(StatusCodeValue, 11200)
    MODEL_TRAINED = typing___cast(StatusCodeValue, 21100)
    MODEL_TRAINING = typing___cast(StatusCodeValue, 21101)
    MODEL_UNTRAINED = typing___cast(StatusCodeValue, 21102)
    MODEL_QUEUED_FOR_TRAINING = typing___cast(StatusCodeValue, 21103)
    MODEL_UPLOADING = typing___cast(StatusCodeValue, 21104)
    MODEL_UPLOADING_FAILED = typing___cast(StatusCodeValue, 21105)
    MODEL_TRAINING_NO_DATA = typing___cast(StatusCodeValue, 21110)
    MODEL_TRAINING_NO_POSITIVES = typing___cast(StatusCodeValue, 21111)
    MODEL_TRAINING_ONE_VS_N_SINGLE_CLASS = typing___cast(StatusCodeValue, 21112)
    MODEL_TRAINING_TIMED_OUT = typing___cast(StatusCodeValue, 21113)
    MODEL_TRAINING_WAITING_ERROR = typing___cast(StatusCodeValue, 21114)
    MODEL_TRAINING_UNKNOWN_ERROR = typing___cast(StatusCodeValue, 21115)
    MODEL_TRAINING_MSG_REDELIVER = typing___cast(StatusCodeValue, 21116)
    MODEL_TRAINING_INSUFFICIENT_DATA = typing___cast(StatusCodeValue, 21117)
    MODEL_TRAINING_INVALID_PARAMS = typing___cast(StatusCodeValue, 21118)
    MODEL_TRAINING_INVALID_DATA_TOLERANCE_EXCEEDED = typing___cast(StatusCodeValue, 21119)
    MODEL_MODIFY_SUCCESS = typing___cast(StatusCodeValue, 21150)
    MODEL_MODIFY_PENDING = typing___cast(StatusCodeValue, 21151)
    MODEL_MODIFY_FAILED = typing___cast(StatusCodeValue, 21152)
    MODEL_DOES_NOT_EXIST = typing___cast(StatusCodeValue, 21200)
    MODEL_PERMISSION_DENIED = typing___cast(StatusCodeValue, 21201)
    MODEL_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 21202)
    MODEL_INVALID_REQUEST = typing___cast(StatusCodeValue, 21203)
    MODEL_EVALUATED = typing___cast(StatusCodeValue, 21300)
    MODEL_EVALUATING = typing___cast(StatusCodeValue, 21301)
    MODEL_NOT_EVALUATED = typing___cast(StatusCodeValue, 21302)
    MODEL_QUEUED_FOR_EVALUATION = typing___cast(StatusCodeValue, 21303)
    MODEL_EVALUATION_TIMED_OUT = typing___cast(StatusCodeValue, 21310)
    MODEL_EVALUATION_WAITING_ERROR = typing___cast(StatusCodeValue, 21311)
    MODEL_EVALUATION_UNKNOWN_ERROR = typing___cast(StatusCodeValue, 21312)
    MODEL_PREDICTION_FAILED = typing___cast(StatusCodeValue, 21313)
    MODEL_EVALUATION_MSG_REDELIVER = typing___cast(StatusCodeValue, 21314)
    MODEL_EVALUATION_NEED_LABELS = typing___cast(StatusCodeValue, 21315)
    MODEL_EVALUATION_NEED_INPUTS = typing___cast(StatusCodeValue, 21316)
    MODEL_DEPLOYMENT_FAILED = typing___cast(StatusCodeValue, 21350)
    MODEL_DEPLOYING = typing___cast(StatusCodeValue, 21351)
    MODEL_QUEUED_FOR_DEPLOYMENT = typing___cast(StatusCodeValue, 21352)
    MODEL_NOT_DEPLOYED = typing___cast(StatusCodeValue, 21353)
    WORKFLOW_NO_MATCHING_INPUT = typing___cast(StatusCodeValue, 22001)
    WORKFLOW_REQUIRE_TRAINED_MODEL = typing___cast(StatusCodeValue, 22002)
    WORKFLOW_DUPLICATE = typing___cast(StatusCodeValue, 22100)
    WORKFLOW_UNSUPPORTED_FORMAT = typing___cast(StatusCodeValue, 22101)
    WORKFLOW_DOES_NOT_EXIST = typing___cast(StatusCodeValue, 22102)
    WORKFLOW_PERMISSION_DENIED = typing___cast(StatusCodeValue, 22103)
    WORKFLOW_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 22104)
    WORKFLOW_INVALID_RECIPE = typing___cast(StatusCodeValue, 22105)
    WORKFLOW_INVALID_TEMPLATE = typing___cast(StatusCodeValue, 22106)
    WORKFLOW_INVALID_GRAPH = typing___cast(StatusCodeValue, 22107)
    WORKFLOW_INTERNAL_FAILURE = typing___cast(StatusCodeValue, 22108)
    WORKFLOW_INVALID_REQUEST = typing___cast(StatusCodeValue, 22999)
    WORKFLOW_MODIFY_SUCCESS = typing___cast(StatusCodeValue, 22150)
    WORKFLOW_MODIFY_PENDING = typing___cast(StatusCodeValue, 22151)
    WORKFLOW_MODIFY_FAILED = typing___cast(StatusCodeValue, 22152)
    WORKFLOW_REINDEX_FAILED = typing___cast(StatusCodeValue, 22153)
    CONCEPT_MODIFY_SUCCESS = typing___cast(StatusCodeValue, 23150)
    CONCEPT_MODIFY_PENDING = typing___cast(StatusCodeValue, 23151)
    CONCEPT_MODIFY_FAILED = typing___cast(StatusCodeValue, 23152)
    ANNOTATION_SUCCESS = typing___cast(StatusCodeValue, 24150)
    ANNOTATION_PENDING = typing___cast(StatusCodeValue, 24151)
    ANNOTATION_FAILED = typing___cast(StatusCodeValue, 24152)
    ANNOTATION_UNKNOWN_STATUS = typing___cast(StatusCodeValue, 24154)
    ANNOTATION_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 24155)
    ANNOTATION_PERMISSION_DENIED = typing___cast(StatusCodeValue, 24156)
    ANNOTATION_AWAITING_REVIEW = typing___cast(StatusCodeValue, 24157)
    ANNOTATION_AWAITING_CONSENSUS_REVIEW = typing___cast(StatusCodeValue, 24159)
    ANNOTATION_REVIEW_DENIED = typing___cast(StatusCodeValue, 24158)
    ANNOTATION_MODIFY_SUCCESS = typing___cast(StatusCodeValue, 24250)
    ANNOTATION_MODIFY_PENDING = typing___cast(StatusCodeValue, 24251)
    ANNOTATION_MODIFY_FAILED = typing___cast(StatusCodeValue, 24252)
    METADATA_INVALID_PATCH_ARGUMENTS = typing___cast(StatusCodeValue, 24900)
    METADATA_PARSING_ISSUE = typing___cast(StatusCodeValue, 24901)
    METADATA_MANIPULATION_ISSUE = typing___cast(StatusCodeValue, 24902)
    TRAINER_JOB_STATE_NONE = typing___cast(StatusCodeValue, 25000)
    TRAINER_JOB_STATE_QUEUED = typing___cast(StatusCodeValue, 25001)
    TRAINER_JOB_STATE_RUNNING = typing___cast(StatusCodeValue, 25002)
    TRAINER_JOB_STATE_COMPLETE = typing___cast(StatusCodeValue, 25003)
    TRAINER_JOB_STATE_ERROR = typing___cast(StatusCodeValue, 25004)
    DATA_DUMP_SUCCESS = typing___cast(StatusCodeValue, 25150)
    DATA_DUMP_PENDING = typing___cast(StatusCodeValue, 25151)
    DATA_DUMP_FAILED = typing___cast(StatusCodeValue, 25152)
    DATA_DUMP_IN_PROGRESS = typing___cast(StatusCodeValue, 25153)
    DATA_DUMP_NO_DATA = typing___cast(StatusCodeValue, 25154)
    APP_DUPLICATION_SUCCESS = typing___cast(StatusCodeValue, 25200)
    APP_DUPLICATION_FAILED = typing___cast(StatusCodeValue, 25201)
    APP_DUPLICATION_PENDING = typing___cast(StatusCodeValue, 25202)
    APP_DUPLICATION_IN_PROGRESS = typing___cast(StatusCodeValue, 25203)
    APP_DUPLICATION_INVALID_REQUEST = typing___cast(StatusCodeValue, 25204)
    INPUT_DOWNLOAD_SUCCESS = typing___cast(StatusCodeValue, 30000)
    INPUT_DOWNLOAD_PENDING = typing___cast(StatusCodeValue, 30001)
    INPUT_DOWNLOAD_FAILED = typing___cast(StatusCodeValue, 30002)
    INPUT_DOWNLOAD_IN_PROGRESS = typing___cast(StatusCodeValue, 30003)
    INPUT_STATUS_UPDATE_FAILED = typing___cast(StatusCodeValue, 30004)
    INPUT_DELETE_FAILED = typing___cast(StatusCodeValue, 30005)
    INPUT_DUPLICATE = typing___cast(StatusCodeValue, 30100)
    INPUT_UNSUPPORTED_FORMAT = typing___cast(StatusCodeValue, 30101)
    INPUT_DOES_NOT_EXIST = typing___cast(StatusCodeValue, 30102)
    INPUT_PERMISSION_DENIED = typing___cast(StatusCodeValue, 30103)
    INPUT_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 30104)
    INPUT_OVER_LIMIT = typing___cast(StatusCodeValue, 30105)
    INPUT_INVALID_URL = typing___cast(StatusCodeValue, 30106)
    INPUT_MODIFY_SUCCESS = typing___cast(StatusCodeValue, 30200)
    INPUT_MODIFY_PENDING = typing___cast(StatusCodeValue, 30201)
    INPUT_MODIFY_FAILED = typing___cast(StatusCodeValue, 30203)
    INPUT_STORAGE_HOST_FAILED = typing___cast(StatusCodeValue, 30210)
    ALL_INPUT_INVALID_BYTES = typing___cast(StatusCodeValue, 30300)
    INPUT_CLUSTER_SUCCESS = typing___cast(StatusCodeValue, 30400)
    INPUT_CLUSTER_PENDING = typing___cast(StatusCodeValue, 30401)
    INPUT_CLUSTER_FAILED = typing___cast(StatusCodeValue, 30402)
    INPUT_CLUSTER_IN_PROGRESS = typing___cast(StatusCodeValue, 30403)
    INPUT_REINDEX_SUCCESS = typing___cast(StatusCodeValue, 30500)
    INPUT_REINDEX_PENDING = typing___cast(StatusCodeValue, 30501)
    INPUT_REINDEX_FAILED = typing___cast(StatusCodeValue, 30502)
    INPUT_REINDEX_IN_PROGRESS = typing___cast(StatusCodeValue, 30503)
    INPUT_VIDEO_DOWNLOAD_SUCCESS = typing___cast(StatusCodeValue, 31000)
    INPUT_VIDEO_DOWNLOAD_PENDING = typing___cast(StatusCodeValue, 31001)
    INPUT_VIDEO_DOWNLOAD_FAILED = typing___cast(StatusCodeValue, 31002)
    INPUT_VIDEO_DUPLICATE = typing___cast(StatusCodeValue, 31100)
    INPUT_VIDEO_UNSUPPORTED_FORMAT = typing___cast(StatusCodeValue, 31101)
    INPUT_VIDEO_DOES_NOT_EXIST = typing___cast(StatusCodeValue, 31102)
    INPUT_VIDEO_PERMISSION_DENIED = typing___cast(StatusCodeValue, 31103)
    INPUT_VIDEO_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 31104)
    INPUT_VIDEO_OVER_LIMIT = typing___cast(StatusCodeValue, 31105)
    INPUT_VIDEO_INVALID_URL = typing___cast(StatusCodeValue, 31106)
    INPUT_VIDEO_MODIFY_SUCCESS = typing___cast(StatusCodeValue, 31200)
    INPUT_VIDEO_MODIFY_PENDING = typing___cast(StatusCodeValue, 31201)
    INPUT_VIDEO_MODIFY_FAILED = typing___cast(StatusCodeValue, 31203)
    INPUT_VIDEO_STORAGE_HOST_FAILED = typing___cast(StatusCodeValue, 31210)
    ALL_INPUT_VIDEOS_INVALID_BYTES = typing___cast(StatusCodeValue, 31300)
    INPUT_CONNECTION_FAILED = typing___cast(StatusCodeValue, 39996)
    REQUEST_DISABLED_FOR_MAINTENANCE = typing___cast(StatusCodeValue, 39997)
    INPUT_WRITES_DISABLED_FOR_MAINTENANCE = typing___cast(StatusCodeValue, 39998)
    INPUT_INVALID_REQUEST = typing___cast(StatusCodeValue, 39999)
    PREDICT_INVALID_REQUEST = typing___cast(StatusCodeValue, 40001)
    SEARCH_INVALID_REQUEST = typing___cast(StatusCodeValue, 40002)
    CONCEPTS_INVALID_REQUEST = typing___cast(StatusCodeValue, 40003)
    STATS_INVALID_REQUEST = typing___cast(StatusCodeValue, 40004)
    DATABASE_DUPLICATE_KEY = typing___cast(StatusCodeValue, 40010)
    DATABASE_STATEMENT_TIMEOUT = typing___cast(StatusCodeValue, 40011)
    DATABASE_INVALID_ROWS_AFFECTED = typing___cast(StatusCodeValue, 40012)
    DATABASE_DEADLOCK_DETECTED = typing___cast(StatusCodeValue, 40013)
    DATABASE_FAIL_TASK = typing___cast(StatusCodeValue, 40014)
    DATABASE_FAIL_TO_GET_CONNECTIONS = typing___cast(StatusCodeValue, 40015)
    DATABASE_TOO_MANY_CLIENTS = typing___cast(StatusCodeValue, 40016)
    DATABASE_CONSTRAINT_VIOLATED = typing___cast(StatusCodeValue, 40017)
    ASYNC_WORKER_MULTI_ERRORS = typing___cast(StatusCodeValue, 40020)
    RPC_REQUEST_QUEUE_FULL = typing___cast(StatusCodeValue, 40030)
    RPC_SERVER_UNAVAILABLE = typing___cast(StatusCodeValue, 40031)
    RPC_REQUEST_TIMEOUT = typing___cast(StatusCodeValue, 40032)
    RPC_MAX_MESSAGE_SIZE_EXCEEDED = typing___cast(StatusCodeValue, 40033)
    RPC_CANCELED = typing___cast(StatusCodeValue, 40035)
    RPC_UNKNOWN_METHOD = typing___cast(StatusCodeValue, 40036)
    CLUSTER_INTERNAL_FAILURE = typing___cast(StatusCodeValue, 43040)
    EXTERNAL_CONNECTION_ERROR = typing___cast(StatusCodeValue, 40034)
    QUEUE_CONN_ERROR = typing___cast(StatusCodeValue, 41000)
    QUEUE_CLOSE_REQUEST_TIMEOUT = typing___cast(StatusCodeValue, 41002)
    QUEUE_CONN_CLOSED = typing___cast(StatusCodeValue, 41003)
    QUEUE_PUBLISH_ACK_TIMEOUT = typing___cast(StatusCodeValue, 41004)
    QUEUE_PUBLISH_ERROR = typing___cast(StatusCodeValue, 41005)
    QUEUE_SUBSCRIPTION_TIMEOUT = typing___cast(StatusCodeValue, 41006)
    QUEUE_SUBSCRIPTION_ERROR = typing___cast(StatusCodeValue, 41007)
    QUEUE_MARSHALLING_FAILED = typing___cast(StatusCodeValue, 41008)
    QUEUE_UNMARSHALLING_FAILED = typing___cast(StatusCodeValue, 41009)
    QUEUE_MAX_MSG_REDELIVERY_EXCEEDED = typing___cast(StatusCodeValue, 41010)
    QUEUE_ACK_FAILURE = typing___cast(StatusCodeValue, 41011)
    SQS_OVERLIMIT = typing___cast(StatusCodeValue, 41100)
    SQS_INVALID_RECEIPT_HANDLE = typing___cast(StatusCodeValue, 41101)
    SQS_UNKNOWN = typing___cast(StatusCodeValue, 41102)
    KAFKA_UNKNOW = typing___cast(StatusCodeValue, 41200)
    KAFKA_MISSING_TOPIC = typing___cast(StatusCodeValue, 41201)
    KAFKA_ADMIN_ERR = typing___cast(StatusCodeValue, 41202)
    KAFKA_CONSUMER_ERR = typing___cast(StatusCodeValue, 41203)
    KAFKA_PUBLISHER_ERR = typing___cast(StatusCodeValue, 41204)
    SEARCH_INTERNAL_FAILURE = typing___cast(StatusCodeValue, 43001)
    SEARCH_PROJECTION_FAILURE = typing___cast(StatusCodeValue, 43002)
    SEARCH_PREDICTION_FAILURE = typing___cast(StatusCodeValue, 43003)
    SEARCH_BY_NOT_FULLY_INDEXED_INPUT = typing___cast(StatusCodeValue, 43004)
    SAVED_SEARCH_MODIFY_FAILED = typing___cast(StatusCodeValue, 43005)
    EVALUATION_QUEUED = typing___cast(StatusCodeValue, 43100)
    EVALUATION_IN_PROGRESS = typing___cast(StatusCodeValue, 43101)
    EVALUATION_SUCCESS = typing___cast(StatusCodeValue, 43102)
    EVALUATION_FAILED_TO_RETRIEVE_DATA = typing___cast(StatusCodeValue, 43103)
    EVALUATION_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 43104)
    EVALUATION_FAILED = typing___cast(StatusCodeValue, 43105)
    EVALUATION_PENDING = typing___cast(StatusCodeValue, 43106)
    EVALUATION_TIMED_OUT = typing___cast(StatusCodeValue, 43107)
    EVALUATION_UNEXPECTED_ERROR = typing___cast(StatusCodeValue, 43108)
    EVALUATION_MIXED = typing___cast(StatusCodeValue, 43109)
    STRIPE_EVENT_ERROR = typing___cast(StatusCodeValue, 44001)
    CACHE_MISS = typing___cast(StatusCodeValue, 45001)
    REDIS_SCRIPT_EXITED_WITH_FAILURE = typing___cast(StatusCodeValue, 45002)
    REDIS_STREAM_ERR = typing___cast(StatusCodeValue, 45003)
    REDIS_NO_CONSUMERS = typing___cast(StatusCodeValue, 45004)
    REDIS_STREAM_BACKOFF = typing___cast(StatusCodeValue, 45005)
    SIGNUP_EVENT_ERROR = typing___cast(StatusCodeValue, 46001)
    SIGNUP_FLAGGED = typing___cast(StatusCodeValue, 46002)
    FILETYPE_UNSUPPORTED = typing___cast(StatusCodeValue, 46003)
    APP_COUNT_INVALID_MESSAGE = typing___cast(StatusCodeValue, 47001)
    APP_COUNT_UPDATE_INCREMENT_FAILED = typing___cast(StatusCodeValue, 47002)
    APP_COUNT_REBUILD_FAILED = typing___cast(StatusCodeValue, 47003)
    APP_COUNT_INTERNAL_FAILURE = typing___cast(StatusCodeValue, 47004)
    MP_DOWNLOAD_ERROR = typing___cast(StatusCodeValue, 47101)
    MP_RESOLVE_DNS_ERROR = typing___cast(StatusCodeValue, 47102)
    MP_DOWNLOAD_MAX_SIZE_EXCEEDED_ERROR = typing___cast(StatusCodeValue, 47103)
    MP_IMAGE_DECODE_ERROR = typing___cast(StatusCodeValue, 47104)
    MP_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 47105)
    MP_IMAGE_PROCESSING_ERROR = typing___cast(StatusCodeValue, 47106)
    USER_CONSENT_FACE = typing___cast(StatusCodeValue, 50001)
    WORKER_MISSING = typing___cast(StatusCodeValue, 51000)
    WORKER_ACTIVE = typing___cast(StatusCodeValue, 51001)
    WORKER_INACTIVE = typing___cast(StatusCodeValue, 51002)
    COLLECTOR_MISSING = typing___cast(StatusCodeValue, 52000)
    COLLECTOR_ACTIVE = typing___cast(StatusCodeValue, 52001)
    COLLECTOR_INACTIVE = typing___cast(StatusCodeValue, 52002)
    SSO_IDENTITY_PROVIDER_DOES_NOT_EXIST = typing___cast(StatusCodeValue, 53001)
    TASK_IN_PROGRESS = typing___cast(StatusCodeValue, 54001)
    TASK_DONE = typing___cast(StatusCodeValue, 54002)
    TASK_WONT_DO = typing___cast(StatusCodeValue, 54003)
    TASK_ADD_ANNOTATIONS_FAILURE = typing___cast(StatusCodeValue, 54005)
    TASK_CONFLICT = typing___cast(StatusCodeValue, 54100)
    TASK_NOT_IMPLEMENTED = typing___cast(StatusCodeValue, 54101)
    TASK_MISSING = typing___cast(StatusCodeValue, 54102)
    LABEL_ORDER_PENDING = typing___cast(StatusCodeValue, 55001)
    LABEL_ORDER_IN_PROGRESS = typing___cast(StatusCodeValue, 55002)
    LABEL_ORDER_SUCCESS = typing___cast(StatusCodeValue, 55003)
    LABEL_ORDER_CANCELED = typing___cast(StatusCodeValue, 55004)
    LICENSE_ACTIVE = typing___cast(StatusCodeValue, 60000)
    LICENSE_DOES_NOT_EXIST = typing___cast(StatusCodeValue, 60001)
    LICENSE_NEED_UPDATE = typing___cast(StatusCodeValue, 60002)
    LICENSE_EXPIRED = typing___cast(StatusCodeValue, 60003)
    LICENSE_REVOKED = typing___cast(StatusCodeValue, 60004)
    LICENSE_DELETED = typing___cast(StatusCodeValue, 60005)
    LICENSE_VOLUME_EXCEEDED = typing___cast(StatusCodeValue, 60006)
    PASSWORD_VALIDATION_SUCCESS = typing___cast(StatusCodeValue, 61000)
    PASSWORD_VALIDATION_FAILED = typing___cast(StatusCodeValue, 61001)
    PASSWORDPOLICY_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 61002)
    FEATUREFLAG_CONFIG_NOT_FOUND = typing___cast(StatusCodeValue, 62000)
    FEATUREFLAG_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 62001)
    FEATUREFLAG_BLOCKED = typing___cast(StatusCodeValue, 62002)
    MAINTENANCE_SUCCESS = typing___cast(StatusCodeValue, 63000)
    MAINTENANCE_FAILED = typing___cast(StatusCodeValue, 63001)
    JOB_QUEUED = typing___cast(StatusCodeValue, 6400)
    JOB_RUNNING = typing___cast(StatusCodeValue, 6401)
    JOB_COMPLETED = typing___cast(StatusCodeValue, 6402)
    JOB_FAILED = typing___cast(StatusCodeValue, 6403)
    INTERNAL_SERVER_ISSUE = typing___cast(StatusCodeValue, 98004)
    INTERNAL_FETCHING_ISSUE = typing___cast(StatusCodeValue, 98005)
    INTERNAL_DATABASE_ISSUE = typing___cast(StatusCodeValue, 98006)
    INTERNAL_UNEXPECTED_TIMEOUT = typing___cast(StatusCodeValue, 98009)
    INTERNAL_UNEXPECTED_V1 = typing___cast(StatusCodeValue, 98010)
    INTERNAL_UNEXPECTED_PANIC = typing___cast(StatusCodeValue, 98011)
    INTERNAL_UNEXPECTED_SPIRE = typing___cast(StatusCodeValue, 98012)
    INTERNAL_REDIS_UNAVAILABLE = typing___cast(StatusCodeValue, 98013)
    INTERNAL_RESOURCE_EXHAUSTED = typing___cast(StatusCodeValue, 98014)
    INTERNAL_REDIS_UNCATEGORIZED = typing___cast(StatusCodeValue, 98015)
    INTERNAL_AWS_UNCATEGORIZED = typing___cast(StatusCodeValue, 98016)
    INTERNAL_AZURE_UNCATEGORIZED = typing___cast(StatusCodeValue, 98017)
    CONN_UNCATEGORIZED = typing___cast(StatusCodeValue, 99001)
    MODEL_UNCATEGORIZED = typing___cast(StatusCodeValue, 99002)
    INPUT_UNCATEGORIZED = typing___cast(StatusCodeValue, 99003)
    ANNOTATION_UNCATEGORIZED = typing___cast(StatusCodeValue, 99004)
    BILLING_UNCATEGORIZED = typing___cast(StatusCodeValue, 99005)
    INTERNAL_UNCATEGORIZED = typing___cast(StatusCodeValue, 99009)
    BAD_REQUEST = typing___cast(StatusCodeValue, 90400)
    SERVER_ERROR = typing___cast(StatusCodeValue, 90500)
ZERO = typing___cast(StatusCodeValue, 0)
SUCCESS = typing___cast(StatusCodeValue, 10000)
MIXED_STATUS = typing___cast(StatusCodeValue, 10010)
FAILURE = typing___cast(StatusCodeValue, 10020)
TRY_AGAIN = typing___cast(StatusCodeValue, 10030)
NOT_IMPLEMENTED = typing___cast(StatusCodeValue, 10040)
CONN_ACCOUNT_ISSUES = typing___cast(StatusCodeValue, 11000)
CONN_TOKEN_INVALID = typing___cast(StatusCodeValue, 11001)
CONN_CREDENTIALS_INVALID = typing___cast(StatusCodeValue, 11002)
CONN_EXCEED_HOURLY_LIMIT = typing___cast(StatusCodeValue, 11003)
CONN_EXCEED_MONTHLY_LIMIT = typing___cast(StatusCodeValue, 11004)
CONN_THROTTLED = typing___cast(StatusCodeValue, 11005)
CONN_EXCEEDS_LIMITS = typing___cast(StatusCodeValue, 11006)
CONN_INSUFFICIENT_SCOPES = typing___cast(StatusCodeValue, 11007)
CONN_KEY_INVALID = typing___cast(StatusCodeValue, 11008)
CONN_KEY_NOT_FOUND = typing___cast(StatusCodeValue, 11009)
CONN_BAD_REQUEST_FORMAT = typing___cast(StatusCodeValue, 11100)
CONN_DOES_NOT_EXIST = typing___cast(StatusCodeValue, 11101)
CONN_INVALID_REQUEST = typing___cast(StatusCodeValue, 11102)
CONN_METHOD_NOT_ALLOWED = typing___cast(StatusCodeValue, 11103)
CONN_NO_GDPR_CONSENT = typing___cast(StatusCodeValue, 11104)
CONN_AUTH_METHOD_DISABLED = typing___cast(StatusCodeValue, 11200)
MODEL_TRAINED = typing___cast(StatusCodeValue, 21100)
MODEL_TRAINING = typing___cast(StatusCodeValue, 21101)
MODEL_UNTRAINED = typing___cast(StatusCodeValue, 21102)
MODEL_QUEUED_FOR_TRAINING = typing___cast(StatusCodeValue, 21103)
MODEL_UPLOADING = typing___cast(StatusCodeValue, 21104)
MODEL_UPLOADING_FAILED = typing___cast(StatusCodeValue, 21105)
MODEL_TRAINING_NO_DATA = typing___cast(StatusCodeValue, 21110)
MODEL_TRAINING_NO_POSITIVES = typing___cast(StatusCodeValue, 21111)
MODEL_TRAINING_ONE_VS_N_SINGLE_CLASS = typing___cast(StatusCodeValue, 21112)
MODEL_TRAINING_TIMED_OUT = typing___cast(StatusCodeValue, 21113)
MODEL_TRAINING_WAITING_ERROR = typing___cast(StatusCodeValue, 21114)
MODEL_TRAINING_UNKNOWN_ERROR = typing___cast(StatusCodeValue, 21115)
MODEL_TRAINING_MSG_REDELIVER = typing___cast(StatusCodeValue, 21116)
MODEL_TRAINING_INSUFFICIENT_DATA = typing___cast(StatusCodeValue, 21117)
MODEL_TRAINING_INVALID_PARAMS = typing___cast(StatusCodeValue, 21118)
MODEL_TRAINING_INVALID_DATA_TOLERANCE_EXCEEDED = typing___cast(StatusCodeValue, 21119)
MODEL_MODIFY_SUCCESS = typing___cast(StatusCodeValue, 21150)
MODEL_MODIFY_PENDING = typing___cast(StatusCodeValue, 21151)
MODEL_MODIFY_FAILED = typing___cast(StatusCodeValue, 21152)
MODEL_DOES_NOT_EXIST = typing___cast(StatusCodeValue, 21200)
MODEL_PERMISSION_DENIED = typing___cast(StatusCodeValue, 21201)
MODEL_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 21202)
MODEL_INVALID_REQUEST = typing___cast(StatusCodeValue, 21203)
MODEL_EVALUATED = typing___cast(StatusCodeValue, 21300)
MODEL_EVALUATING = typing___cast(StatusCodeValue, 21301)
MODEL_NOT_EVALUATED = typing___cast(StatusCodeValue, 21302)
MODEL_QUEUED_FOR_EVALUATION = typing___cast(StatusCodeValue, 21303)
MODEL_EVALUATION_TIMED_OUT = typing___cast(StatusCodeValue, 21310)
MODEL_EVALUATION_WAITING_ERROR = typing___cast(StatusCodeValue, 21311)
MODEL_EVALUATION_UNKNOWN_ERROR = typing___cast(StatusCodeValue, 21312)
MODEL_PREDICTION_FAILED = typing___cast(StatusCodeValue, 21313)
MODEL_EVALUATION_MSG_REDELIVER = typing___cast(StatusCodeValue, 21314)
MODEL_EVALUATION_NEED_LABELS = typing___cast(StatusCodeValue, 21315)
MODEL_EVALUATION_NEED_INPUTS = typing___cast(StatusCodeValue, 21316)
MODEL_DEPLOYMENT_FAILED = typing___cast(StatusCodeValue, 21350)
MODEL_DEPLOYING = typing___cast(StatusCodeValue, 21351)
MODEL_QUEUED_FOR_DEPLOYMENT = typing___cast(StatusCodeValue, 21352)
MODEL_NOT_DEPLOYED = typing___cast(StatusCodeValue, 21353)
WORKFLOW_NO_MATCHING_INPUT = typing___cast(StatusCodeValue, 22001)
WORKFLOW_REQUIRE_TRAINED_MODEL = typing___cast(StatusCodeValue, 22002)
WORKFLOW_DUPLICATE = typing___cast(StatusCodeValue, 22100)
WORKFLOW_UNSUPPORTED_FORMAT = typing___cast(StatusCodeValue, 22101)
WORKFLOW_DOES_NOT_EXIST = typing___cast(StatusCodeValue, 22102)
WORKFLOW_PERMISSION_DENIED = typing___cast(StatusCodeValue, 22103)
WORKFLOW_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 22104)
WORKFLOW_INVALID_RECIPE = typing___cast(StatusCodeValue, 22105)
WORKFLOW_INVALID_TEMPLATE = typing___cast(StatusCodeValue, 22106)
WORKFLOW_INVALID_GRAPH = typing___cast(StatusCodeValue, 22107)
WORKFLOW_INTERNAL_FAILURE = typing___cast(StatusCodeValue, 22108)
WORKFLOW_INVALID_REQUEST = typing___cast(StatusCodeValue, 22999)
WORKFLOW_MODIFY_SUCCESS = typing___cast(StatusCodeValue, 22150)
WORKFLOW_MODIFY_PENDING = typing___cast(StatusCodeValue, 22151)
WORKFLOW_MODIFY_FAILED = typing___cast(StatusCodeValue, 22152)
WORKFLOW_REINDEX_FAILED = typing___cast(StatusCodeValue, 22153)
CONCEPT_MODIFY_SUCCESS = typing___cast(StatusCodeValue, 23150)
CONCEPT_MODIFY_PENDING = typing___cast(StatusCodeValue, 23151)
CONCEPT_MODIFY_FAILED = typing___cast(StatusCodeValue, 23152)
ANNOTATION_SUCCESS = typing___cast(StatusCodeValue, 24150)
ANNOTATION_PENDING = typing___cast(StatusCodeValue, 24151)
ANNOTATION_FAILED = typing___cast(StatusCodeValue, 24152)
ANNOTATION_UNKNOWN_STATUS = typing___cast(StatusCodeValue, 24154)
ANNOTATION_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 24155)
ANNOTATION_PERMISSION_DENIED = typing___cast(StatusCodeValue, 24156)
ANNOTATION_AWAITING_REVIEW = typing___cast(StatusCodeValue, 24157)
ANNOTATION_AWAITING_CONSENSUS_REVIEW = typing___cast(StatusCodeValue, 24159)
ANNOTATION_REVIEW_DENIED = typing___cast(StatusCodeValue, 24158)
ANNOTATION_MODIFY_SUCCESS = typing___cast(StatusCodeValue, 24250)
ANNOTATION_MODIFY_PENDING = typing___cast(StatusCodeValue, 24251)
ANNOTATION_MODIFY_FAILED = typing___cast(StatusCodeValue, 24252)
METADATA_INVALID_PATCH_ARGUMENTS = typing___cast(StatusCodeValue, 24900)
METADATA_PARSING_ISSUE = typing___cast(StatusCodeValue, 24901)
METADATA_MANIPULATION_ISSUE = typing___cast(StatusCodeValue, 24902)
TRAINER_JOB_STATE_NONE = typing___cast(StatusCodeValue, 25000)
TRAINER_JOB_STATE_QUEUED = typing___cast(StatusCodeValue, 25001)
TRAINER_JOB_STATE_RUNNING = typing___cast(StatusCodeValue, 25002)
TRAINER_JOB_STATE_COMPLETE = typing___cast(StatusCodeValue, 25003)
TRAINER_JOB_STATE_ERROR = typing___cast(StatusCodeValue, 25004)
DATA_DUMP_SUCCESS = typing___cast(StatusCodeValue, 25150)
DATA_DUMP_PENDING = typing___cast(StatusCodeValue, 25151)
DATA_DUMP_FAILED = typing___cast(StatusCodeValue, 25152)
DATA_DUMP_IN_PROGRESS = typing___cast(StatusCodeValue, 25153)
DATA_DUMP_NO_DATA = typing___cast(StatusCodeValue, 25154)
APP_DUPLICATION_SUCCESS = typing___cast(StatusCodeValue, 25200)
APP_DUPLICATION_FAILED = typing___cast(StatusCodeValue, 25201)
APP_DUPLICATION_PENDING = typing___cast(StatusCodeValue, 25202)
APP_DUPLICATION_IN_PROGRESS = typing___cast(StatusCodeValue, 25203)
APP_DUPLICATION_INVALID_REQUEST = typing___cast(StatusCodeValue, 25204)
INPUT_DOWNLOAD_SUCCESS = typing___cast(StatusCodeValue, 30000)
INPUT_DOWNLOAD_PENDING = typing___cast(StatusCodeValue, 30001)
INPUT_DOWNLOAD_FAILED = typing___cast(StatusCodeValue, 30002)
INPUT_DOWNLOAD_IN_PROGRESS = typing___cast(StatusCodeValue, 30003)
INPUT_STATUS_UPDATE_FAILED = typing___cast(StatusCodeValue, 30004)
INPUT_DELETE_FAILED = typing___cast(StatusCodeValue, 30005)
INPUT_DUPLICATE = typing___cast(StatusCodeValue, 30100)
INPUT_UNSUPPORTED_FORMAT = typing___cast(StatusCodeValue, 30101)
INPUT_DOES_NOT_EXIST = typing___cast(StatusCodeValue, 30102)
INPUT_PERMISSION_DENIED = typing___cast(StatusCodeValue, 30103)
INPUT_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 30104)
INPUT_OVER_LIMIT = typing___cast(StatusCodeValue, 30105)
INPUT_INVALID_URL = typing___cast(StatusCodeValue, 30106)
INPUT_MODIFY_SUCCESS = typing___cast(StatusCodeValue, 30200)
INPUT_MODIFY_PENDING = typing___cast(StatusCodeValue, 30201)
INPUT_MODIFY_FAILED = typing___cast(StatusCodeValue, 30203)
INPUT_STORAGE_HOST_FAILED = typing___cast(StatusCodeValue, 30210)
ALL_INPUT_INVALID_BYTES = typing___cast(StatusCodeValue, 30300)
INPUT_CLUSTER_SUCCESS = typing___cast(StatusCodeValue, 30400)
INPUT_CLUSTER_PENDING = typing___cast(StatusCodeValue, 30401)
INPUT_CLUSTER_FAILED = typing___cast(StatusCodeValue, 30402)
INPUT_CLUSTER_IN_PROGRESS = typing___cast(StatusCodeValue, 30403)
INPUT_REINDEX_SUCCESS = typing___cast(StatusCodeValue, 30500)
INPUT_REINDEX_PENDING = typing___cast(StatusCodeValue, 30501)
INPUT_REINDEX_FAILED = typing___cast(StatusCodeValue, 30502)
INPUT_REINDEX_IN_PROGRESS = typing___cast(StatusCodeValue, 30503)
INPUT_VIDEO_DOWNLOAD_SUCCESS = typing___cast(StatusCodeValue, 31000)
INPUT_VIDEO_DOWNLOAD_PENDING = typing___cast(StatusCodeValue, 31001)
INPUT_VIDEO_DOWNLOAD_FAILED = typing___cast(StatusCodeValue, 31002)
INPUT_VIDEO_DUPLICATE = typing___cast(StatusCodeValue, 31100)
INPUT_VIDEO_UNSUPPORTED_FORMAT = typing___cast(StatusCodeValue, 31101)
INPUT_VIDEO_DOES_NOT_EXIST = typing___cast(StatusCodeValue, 31102)
INPUT_VIDEO_PERMISSION_DENIED = typing___cast(StatusCodeValue, 31103)
INPUT_VIDEO_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 31104)
INPUT_VIDEO_OVER_LIMIT = typing___cast(StatusCodeValue, 31105)
INPUT_VIDEO_INVALID_URL = typing___cast(StatusCodeValue, 31106)
INPUT_VIDEO_MODIFY_SUCCESS = typing___cast(StatusCodeValue, 31200)
INPUT_VIDEO_MODIFY_PENDING = typing___cast(StatusCodeValue, 31201)
INPUT_VIDEO_MODIFY_FAILED = typing___cast(StatusCodeValue, 31203)
INPUT_VIDEO_STORAGE_HOST_FAILED = typing___cast(StatusCodeValue, 31210)
ALL_INPUT_VIDEOS_INVALID_BYTES = typing___cast(StatusCodeValue, 31300)
INPUT_CONNECTION_FAILED = typing___cast(StatusCodeValue, 39996)
REQUEST_DISABLED_FOR_MAINTENANCE = typing___cast(StatusCodeValue, 39997)
INPUT_WRITES_DISABLED_FOR_MAINTENANCE = typing___cast(StatusCodeValue, 39998)
INPUT_INVALID_REQUEST = typing___cast(StatusCodeValue, 39999)
PREDICT_INVALID_REQUEST = typing___cast(StatusCodeValue, 40001)
SEARCH_INVALID_REQUEST = typing___cast(StatusCodeValue, 40002)
CONCEPTS_INVALID_REQUEST = typing___cast(StatusCodeValue, 40003)
STATS_INVALID_REQUEST = typing___cast(StatusCodeValue, 40004)
DATABASE_DUPLICATE_KEY = typing___cast(StatusCodeValue, 40010)
DATABASE_STATEMENT_TIMEOUT = typing___cast(StatusCodeValue, 40011)
DATABASE_INVALID_ROWS_AFFECTED = typing___cast(StatusCodeValue, 40012)
DATABASE_DEADLOCK_DETECTED = typing___cast(StatusCodeValue, 40013)
DATABASE_FAIL_TASK = typing___cast(StatusCodeValue, 40014)
DATABASE_FAIL_TO_GET_CONNECTIONS = typing___cast(StatusCodeValue, 40015)
DATABASE_TOO_MANY_CLIENTS = typing___cast(StatusCodeValue, 40016)
DATABASE_CONSTRAINT_VIOLATED = typing___cast(StatusCodeValue, 40017)
ASYNC_WORKER_MULTI_ERRORS = typing___cast(StatusCodeValue, 40020)
RPC_REQUEST_QUEUE_FULL = typing___cast(StatusCodeValue, 40030)
RPC_SERVER_UNAVAILABLE = typing___cast(StatusCodeValue, 40031)
RPC_REQUEST_TIMEOUT = typing___cast(StatusCodeValue, 40032)
RPC_MAX_MESSAGE_SIZE_EXCEEDED = typing___cast(StatusCodeValue, 40033)
RPC_CANCELED = typing___cast(StatusCodeValue, 40035)
RPC_UNKNOWN_METHOD = typing___cast(StatusCodeValue, 40036)
CLUSTER_INTERNAL_FAILURE = typing___cast(StatusCodeValue, 43040)
EXTERNAL_CONNECTION_ERROR = typing___cast(StatusCodeValue, 40034)
QUEUE_CONN_ERROR = typing___cast(StatusCodeValue, 41000)
QUEUE_CLOSE_REQUEST_TIMEOUT = typing___cast(StatusCodeValue, 41002)
QUEUE_CONN_CLOSED = typing___cast(StatusCodeValue, 41003)
QUEUE_PUBLISH_ACK_TIMEOUT = typing___cast(StatusCodeValue, 41004)
QUEUE_PUBLISH_ERROR = typing___cast(StatusCodeValue, 41005)
QUEUE_SUBSCRIPTION_TIMEOUT = typing___cast(StatusCodeValue, 41006)
QUEUE_SUBSCRIPTION_ERROR = typing___cast(StatusCodeValue, 41007)
QUEUE_MARSHALLING_FAILED = typing___cast(StatusCodeValue, 41008)
QUEUE_UNMARSHALLING_FAILED = typing___cast(StatusCodeValue, 41009)
QUEUE_MAX_MSG_REDELIVERY_EXCEEDED = typing___cast(StatusCodeValue, 41010)
QUEUE_ACK_FAILURE = typing___cast(StatusCodeValue, 41011)
SQS_OVERLIMIT = typing___cast(StatusCodeValue, 41100)
SQS_INVALID_RECEIPT_HANDLE = typing___cast(StatusCodeValue, 41101)
SQS_UNKNOWN = typing___cast(StatusCodeValue, 41102)
KAFKA_UNKNOW = typing___cast(StatusCodeValue, 41200)
KAFKA_MISSING_TOPIC = typing___cast(StatusCodeValue, 41201)
KAFKA_ADMIN_ERR = typing___cast(StatusCodeValue, 41202)
KAFKA_CONSUMER_ERR = typing___cast(StatusCodeValue, 41203)
KAFKA_PUBLISHER_ERR = typing___cast(StatusCodeValue, 41204)
SEARCH_INTERNAL_FAILURE = typing___cast(StatusCodeValue, 43001)
SEARCH_PROJECTION_FAILURE = typing___cast(StatusCodeValue, 43002)
SEARCH_PREDICTION_FAILURE = typing___cast(StatusCodeValue, 43003)
SEARCH_BY_NOT_FULLY_INDEXED_INPUT = typing___cast(StatusCodeValue, 43004)
SAVED_SEARCH_MODIFY_FAILED = typing___cast(StatusCodeValue, 43005)
EVALUATION_QUEUED = typing___cast(StatusCodeValue, 43100)
EVALUATION_IN_PROGRESS = typing___cast(StatusCodeValue, 43101)
EVALUATION_SUCCESS = typing___cast(StatusCodeValue, 43102)
EVALUATION_FAILED_TO_RETRIEVE_DATA = typing___cast(StatusCodeValue, 43103)
EVALUATION_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 43104)
EVALUATION_FAILED = typing___cast(StatusCodeValue, 43105)
EVALUATION_PENDING = typing___cast(StatusCodeValue, 43106)
EVALUATION_TIMED_OUT = typing___cast(StatusCodeValue, 43107)
EVALUATION_UNEXPECTED_ERROR = typing___cast(StatusCodeValue, 43108)
EVALUATION_MIXED = typing___cast(StatusCodeValue, 43109)
STRIPE_EVENT_ERROR = typing___cast(StatusCodeValue, 44001)
CACHE_MISS = typing___cast(StatusCodeValue, 45001)
REDIS_SCRIPT_EXITED_WITH_FAILURE = typing___cast(StatusCodeValue, 45002)
REDIS_STREAM_ERR = typing___cast(StatusCodeValue, 45003)
REDIS_NO_CONSUMERS = typing___cast(StatusCodeValue, 45004)
REDIS_STREAM_BACKOFF = typing___cast(StatusCodeValue, 45005)
SIGNUP_EVENT_ERROR = typing___cast(StatusCodeValue, 46001)
SIGNUP_FLAGGED = typing___cast(StatusCodeValue, 46002)
FILETYPE_UNSUPPORTED = typing___cast(StatusCodeValue, 46003)
APP_COUNT_INVALID_MESSAGE = typing___cast(StatusCodeValue, 47001)
APP_COUNT_UPDATE_INCREMENT_FAILED = typing___cast(StatusCodeValue, 47002)
APP_COUNT_REBUILD_FAILED = typing___cast(StatusCodeValue, 47003)
APP_COUNT_INTERNAL_FAILURE = typing___cast(StatusCodeValue, 47004)
MP_DOWNLOAD_ERROR = typing___cast(StatusCodeValue, 47101)
MP_RESOLVE_DNS_ERROR = typing___cast(StatusCodeValue, 47102)
MP_DOWNLOAD_MAX_SIZE_EXCEEDED_ERROR = typing___cast(StatusCodeValue, 47103)
MP_IMAGE_DECODE_ERROR = typing___cast(StatusCodeValue, 47104)
MP_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 47105)
MP_IMAGE_PROCESSING_ERROR = typing___cast(StatusCodeValue, 47106)
USER_CONSENT_FACE = typing___cast(StatusCodeValue, 50001)
WORKER_MISSING = typing___cast(StatusCodeValue, 51000)
WORKER_ACTIVE = typing___cast(StatusCodeValue, 51001)
WORKER_INACTIVE = typing___cast(StatusCodeValue, 51002)
COLLECTOR_MISSING = typing___cast(StatusCodeValue, 52000)
COLLECTOR_ACTIVE = typing___cast(StatusCodeValue, 52001)
COLLECTOR_INACTIVE = typing___cast(StatusCodeValue, 52002)
SSO_IDENTITY_PROVIDER_DOES_NOT_EXIST = typing___cast(StatusCodeValue, 53001)
TASK_IN_PROGRESS = typing___cast(StatusCodeValue, 54001)
TASK_DONE = typing___cast(StatusCodeValue, 54002)
TASK_WONT_DO = typing___cast(StatusCodeValue, 54003)
TASK_ADD_ANNOTATIONS_FAILURE = typing___cast(StatusCodeValue, 54005)
TASK_CONFLICT = typing___cast(StatusCodeValue, 54100)
TASK_NOT_IMPLEMENTED = typing___cast(StatusCodeValue, 54101)
TASK_MISSING = typing___cast(StatusCodeValue, 54102)
LABEL_ORDER_PENDING = typing___cast(StatusCodeValue, 55001)
LABEL_ORDER_IN_PROGRESS = typing___cast(StatusCodeValue, 55002)
LABEL_ORDER_SUCCESS = typing___cast(StatusCodeValue, 55003)
LABEL_ORDER_CANCELED = typing___cast(StatusCodeValue, 55004)
LICENSE_ACTIVE = typing___cast(StatusCodeValue, 60000)
LICENSE_DOES_NOT_EXIST = typing___cast(StatusCodeValue, 60001)
LICENSE_NEED_UPDATE = typing___cast(StatusCodeValue, 60002)
LICENSE_EXPIRED = typing___cast(StatusCodeValue, 60003)
LICENSE_REVOKED = typing___cast(StatusCodeValue, 60004)
LICENSE_DELETED = typing___cast(StatusCodeValue, 60005)
LICENSE_VOLUME_EXCEEDED = typing___cast(StatusCodeValue, 60006)
PASSWORD_VALIDATION_SUCCESS = typing___cast(StatusCodeValue, 61000)
PASSWORD_VALIDATION_FAILED = typing___cast(StatusCodeValue, 61001)
PASSWORDPOLICY_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 61002)
FEATUREFLAG_CONFIG_NOT_FOUND = typing___cast(StatusCodeValue, 62000)
FEATUREFLAG_INVALID_ARGUMENT = typing___cast(StatusCodeValue, 62001)
FEATUREFLAG_BLOCKED = typing___cast(StatusCodeValue, 62002)
MAINTENANCE_SUCCESS = typing___cast(StatusCodeValue, 63000)
MAINTENANCE_FAILED = typing___cast(StatusCodeValue, 63001)
JOB_QUEUED = typing___cast(StatusCodeValue, 6400)
JOB_RUNNING = typing___cast(StatusCodeValue, 6401)
JOB_COMPLETED = typing___cast(StatusCodeValue, 6402)
JOB_FAILED = typing___cast(StatusCodeValue, 6403)
INTERNAL_SERVER_ISSUE = typing___cast(StatusCodeValue, 98004)
INTERNAL_FETCHING_ISSUE = typing___cast(StatusCodeValue, 98005)
INTERNAL_DATABASE_ISSUE = typing___cast(StatusCodeValue, 98006)
INTERNAL_UNEXPECTED_TIMEOUT = typing___cast(StatusCodeValue, 98009)
INTERNAL_UNEXPECTED_V1 = typing___cast(StatusCodeValue, 98010)
INTERNAL_UNEXPECTED_PANIC = typing___cast(StatusCodeValue, 98011)
INTERNAL_UNEXPECTED_SPIRE = typing___cast(StatusCodeValue, 98012)
INTERNAL_REDIS_UNAVAILABLE = typing___cast(StatusCodeValue, 98013)
INTERNAL_RESOURCE_EXHAUSTED = typing___cast(StatusCodeValue, 98014)
INTERNAL_REDIS_UNCATEGORIZED = typing___cast(StatusCodeValue, 98015)
INTERNAL_AWS_UNCATEGORIZED = typing___cast(StatusCodeValue, 98016)
INTERNAL_AZURE_UNCATEGORIZED = typing___cast(StatusCodeValue, 98017)
CONN_UNCATEGORIZED = typing___cast(StatusCodeValue, 99001)
MODEL_UNCATEGORIZED = typing___cast(StatusCodeValue, 99002)
INPUT_UNCATEGORIZED = typing___cast(StatusCodeValue, 99003)
ANNOTATION_UNCATEGORIZED = typing___cast(StatusCodeValue, 99004)
BILLING_UNCATEGORIZED = typing___cast(StatusCodeValue, 99005)
INTERNAL_UNCATEGORIZED = typing___cast(StatusCodeValue, 99009)
BAD_REQUEST = typing___cast(StatusCodeValue, 90400)
SERVER_ERROR = typing___cast(StatusCodeValue, 90500)
type___StatusCode = StatusCode
