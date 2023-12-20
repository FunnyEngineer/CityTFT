#!/bin/bash

# Function to display script usage
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -x, --xml           Specify an input xml file"
 echo " -c, --climate       Specify an climate cli file"
 echo " -p, --processed_dir    The storage that you want to export the processed data to"
 echo " -e, --export_dir    The storage that you want to export the processed data to"
 echo " -h, --help          Show this help message and exit"
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

      -x | --xml*)
        if ! has_argument $@; then
          echo "Input xml file not specified." >&2
          usage
          exit 1
        fi
        input_file=$(extract_argument $@)
        shift
        ;;
        
      -c | --climate*)
        if ! has_argument $@; then
          echo "Climate cli file not specified." >&2
          usage
          exit 1
        fi
        climate_file=$(extract_argument $@)
        climate_filename=$(basename "$climate_file")
        shift
        ;;

      -p | --processed_dir*)
        if ! has_argument $@; then
          echo "Processed directory not specified." >&2
          usage
          exit 1
        fi
        processed_dir=$(extract_argument $@)
        [ ! -d "$processed_dir" ] && mkdir -p "$processed_dir" && echo "Creating directory $processed_dir" >&2
        shift
        ;;

      -n | --new_xml_dir*)
        if ! has_argument $@; then
          echo "New xml directory not specified." >&2
          usage
          exit 1
        fi
        new_xml_dir=$(extract_argument $@)
        [ ! -d "$new_xml_dir" ] && mkdir -p "$new_xml_dir" && echo "Creating directory $new_xml_dir" >&2
        shift
        ;;

      -e | --export_dir*)
        if ! has_argument $@; then
          echo "Export directory not specified." >&2
          usage
          exit 1
        fi
        export_dir=$(extract_argument $@)
        [ ! -d "$export_dir" ] && mkdir -p "$export_dir" && echo "Creating directory $export_dir" >&2
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


# Process the climate file
python3 ./script/cli_processing.py -i $climate_file -o "$processed_dir/$climate_filename"
echo "New climate file: $processed_dir/$climate_filename"
python3 ./script/new_xml.py -i $input_file -e $new_xml_dir -c "$processed_dir/$climate_filename"
new_xml_file=$(basename -- "$climate_file" .cli)
new_xml_path="${new_xml_dir}/${new_xml_file}.xml"
echo "New xml file: $new_xml_path"
./CitySim-Solver/CitySim $new_xml_path
python3 ./script/post_process.py $new_xml_dir $export_dir $new_xml_file
