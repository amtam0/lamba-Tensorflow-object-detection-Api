#!/bin/bash
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
rm ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
cd ssdlite_mobilenet_v2_coco_2018_05_09/
mkdir training/
mkdir inference_graph/
mv frozen_inference_graph.pb inference_graph/
wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt -O training/label_map.pbtxt
find '(' -name training -o -name inference_graph ')' -prune -o -exec rm -rf {} \;


