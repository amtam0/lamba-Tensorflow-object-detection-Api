#!/bin/bash
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
tar -xzvf faster_rcnn_resnet50_coco_2018_01_28.tar.gz
rm faster_rcnn_resnet50_coco_2018_01_28.tar.gz
cd faster_rcnn_resnet50_coco_2018_01_28/
mkdir training/
mkdir inference_graph/
mv frozen_inference_graph.pb inference_graph/
wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt -O training/label_map.pbtxt
find '(' -name training -o -name inference_graph ')' -prune -o -exec rm -rf {} \;


