#!/bin/bash

# Function to display script usage
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -l, --log       Specify a log dir to sync"
}

has_argument() {
    [[ ("$1" == *=* && -n ${1#*=}) || ( ! -z "$2" && "$2" != -*)  ]];
}

extract_argument() {
  echo "${2:-${1#*=}}"
}

# Function to handle options and arguments
handle_options() {
  while [ $# -gt 0 ]; do
    case $1 in
      -h | --help)
        usage
        exit 0
        ;;

      -l | --log*)
        if ! has_argument $@; then
          echo "Log dir not specified." >&2
          usage
          exit 1
        fi
        log_dir=$(extract_argument $@)
        shift
        ;;
      *)
        echo "Invalid option: $1" >&2
        usage
        exit 1
        ;;
    esac
    shift
  done
}

# Main script execution
handle_options "$@"

# Perform the desired actions based on the provided flags and arguments


# Sync log dir
rsync -avzh tudai@ls6:/home1/08388/tudai/CityTFT/$log_dir ./