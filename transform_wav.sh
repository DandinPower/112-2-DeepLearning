#!/bin/bash

SOURCE_DIR="downloads/train/train"
DEST_DIR="transform/train"

mkdir -p "$DEST_DIR"

for file in "$SOURCE_DIR"/*.wav
do
  filename=$(basename "$file")

  # parse index from filename
  # original filename regex: number.wav
  index=$(echo "$filename" | grep -o -E '[0-9]+' | head -1)

  index_number=$(printf "%04d" $index)
  new_filename="B${index_number}.wav"
  sox "$file" -r 16000 -e signed-integer -b 16 "$DEST_DIR/$new_filename"
done