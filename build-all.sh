
#setup Tensorflow
cd build_libraries/tensorflow_clean; bash create_zip.sh
#setup OpenCV/Pillow layer
cd ../python_layers; bash build_layer.sh
#download models
cd ../../OD_model; 
bash download_ssdlite_mobilenet_v2_coco_2018_05_09.sh
bash download_faster_rcnn_resnet50_coco_2018_01_28.sh
bash download_faster_rcnn_inception_v2_coco_2018_01_28.sh
cd ../OD_FCT; bash zip_function.sh