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


function run_isort() {
  echo ""
  echo "- isort: Make sure all imports are sorted"

  if [ "$1" == "isort" ]; then
    isort --sp .isort.cfg --ws $(eval ${FIND_SOURCE_FILES})
  fi
  # This ignores whitespace
  isort --sp .isort.cfg --ws --diff -c $(eval ${FIND_SOURCE_FILES})

  if [ $? != 0 ]; then
    echo ""
    echo "  isort failed. Run './lint.sh isort' to automatically sort the imports correctly."
    exit 1
  fi

  echo "  Done isort"
}

run_autoflake
run_isort $1

echo "SUCCESSFUL FINISH OF lint.sh"
