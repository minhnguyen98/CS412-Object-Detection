#!/bin/bash

function process_image {
  input=$1
  output=$2
  cd darknet
  ./darknet detect cfg/yolov3.cfg yolov3.weights ../$input
  mv predictions.jpg ../$output
  cd ..
}

for f in input/*; do
    out="output/${f:6}"
    process_image $f $out
done