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
    echo "    Note: The import code style itself (besides the order) must still comply to yapf"
    exit 1
  fi

  echo "  Done isort"
}

function run_yapf {
  echo ""
  echo "- yapf: Make sure there are no code style issues"

  if [ "$1" == "yapf" ]; then
    yapf --style=.style.yapf -p -i $(eval ${FIND_SOURCE_FILES})
  fi
  yapf --style=.style.yapf -p -d $(eval ${FIND_SOURCE_FILES})

  if [ $? != 0 ]; then
    echo ""
    echo "  yapf failed. Run './lint.sh yapf' to automatically apply yapf rules."
    exit 1
  fi

  echo "  Done aypf"
}


run_isort $1
run_yapf $1

echo "SUCCESSFUL FINISH OF lint.sh"
