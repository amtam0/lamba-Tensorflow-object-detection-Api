If you want to build this folder from scratch

- Clone the Tensorflow repo with the tag : 1.13.

```git clone -b v1.13.0 https://github.com/tensorflow/models.git```

- Run :

```apt install protobuf-compiler; cd models/research; protoc object_detection/protos/*.proto --python_out=.```
