#!/bin/bash

set -eo pipefail

FIND_SOURCE_FILES='find clarifai_grpc tests -name "*.py" -not -path "clarifai_grpc/grpc/*"'

function run_autoflake {
  echo "- autoflake: Make sure there are no unused imports and unused variables"
  FIND_SOURCE_FILES_FOR_AUTOFLAKE="${FIND_SOURCE_FILES} -not -iwholename '*/__init__.py'"
  autoflake --remove-all-unused-imports --remove-unused-variables \
          $(eval ${FIND_SOURCE_FILES_FOR_AUTOFLAKE}) \
          | tee autoflake-output.tmp

  if [ $(cat autoflake-output.tmp | wc -c) != 0 ]; then
    echo ""
    echo "  Here are affected files: "
    grep "+++" autoflake-output.tmp
    rm autoflake-output.tmp
    echo "  autoflake failed"
    exit 1
  fi

  rm autoflake-output.tmp
  echo "  Done autoflake"
}


run_autoflake

echo "SUCCESSFUL FINISH OF lint.sh"
