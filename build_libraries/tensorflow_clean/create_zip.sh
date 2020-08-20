docker run --rm -it -v ${PWD}:/var/task lambci/lambda:build-python3.6 bash build-tf1.sh
mv requirements.zip ../../OD_FCT/
