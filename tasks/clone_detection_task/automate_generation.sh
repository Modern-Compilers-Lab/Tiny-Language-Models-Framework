#!/bin/bash

# Check if the --num_programs argument is provided
if [ -z "$1" ] || [ "$1" != "--num_programs" ] || [ -z "$2" ]; then
  echo "Usage: $0 --num_programs <number_of_programs>"
  exit 1
fi

NUM_PROGRAMS=$2

# Run automate.py with the given number of programs
python ./tasks/clone_detection_task/automate.py --num_programs $NUM_PROGRAMS
