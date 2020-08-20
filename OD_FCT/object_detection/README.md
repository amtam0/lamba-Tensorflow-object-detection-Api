Clone the Tensorflow repo with the tag : 1.13.

apt install protobuf-compiler

cd models/research

protoc object_detection/protos/*.proto --python_out=.
