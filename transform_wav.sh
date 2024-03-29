#!/bin/bash

transform_wav() {
  local SOURCE_DIR="$1"
  local DEST_DIR="$2"
  local fill_length="$3"

  rm -rf "$DEST_DIR"
  mkdir -p "$DEST_DIR"

  for file in "$SOURCE_DIR"/*.wav
  do
    filename=$(basename "$file")

    # parse index from filename
    # original filename regex: number.wav
    index=$(echo "$filename" | grep -o -E '[0-9]+' | head -1)

    index_number=$(printf "%0${fill_length}d" $index) 
    new_filename="${index_number}.wav"
    sox "$file" -r 16000 -e signed-integer -b 16 "$DEST_DIR/$new_filename"
  done
}

transform_wav "datasets/downloads/test/test" "datasets/transform/test" 0
transform_wav "datasets/downloads/train/train" "datasets/transform/train" 0
