#!/bin/bash
# Usage: ./check_file_sizes.sh <root_dir>
find "$1" -type f -name '*.nii.gz' -exec ls -lh {} + | sort -k 5 -h 