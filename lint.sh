FIND_SOURCE_FILES='find clarifai_grpc tests -name "*.py" -not -path "clarifai_grpc/grpc/*"'

function run_isort() {
  echo ""
  echo "- isort: Make sure all imports are sorted"

  if [ "$1" == "isort" ]; then
    isort --sp .isort.cfg --ws $(eval ${FIND_SOURCE_FILES})
  fi
  # This ignores whitespace which is crucial because yapf defines the formatting.
  isort --sp .isort.cfg --ws --diff -c $(eval ${FIND_SOURCE_FILES})

  if [ $? != 0 ]; then
    echo ""
    echo "  isort failed. Run './lint.sh isort' to automatically sort the imports correctly."
    exit 1
  fi

  echo "  Done isort"
}

run_isort $1