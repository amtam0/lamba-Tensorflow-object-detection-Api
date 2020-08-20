This example to test the notebook inference and loading of the model (optional)

You need to download the models in OD_model/ before running this script !

Run this command

```docker run -it -p 8888:8888 -v "$(dirname "$(pwd)")":/OD1 -w /OD1 tensorflow/tensorflow:1.15.2-py3-jupyter bash inference_example/run_jupyter.sh```

Open the jupyter link and run the notebook

We use the model ssdlite_mobilenet_v2_coco_2018_05_09 as default
