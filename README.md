### Setup Tensorflow Object detection in AWS Lambda

We will be using Python 3.6 and Tensorflow==1.15.2

Input:
![Alt inout](inference_example/examples/image1.jpg?raw=true "Title")

Output:
![Alt output](inference_example/examples/image1-result.jpg?raw=true "Title")

### Process :

1 - Build packages and Tensorflow > check build_libraries/ folder

2 - Download models > check OD_model/ folder

3 - Test function (optional) > check inference_example/ folder

3 - Zip function > check OF_FCT/ folder

0 - To build all at once, run
```bash build-all.sh```

Files to upload to S3 :
- Lambda function : od_function.zip
- Model folder(s)
- Lambda layer : mypythonlibs36.zip

Each folder has its own README for more details to setup

For AWS setup, check the Medium [link](https://medium.com/p/39ea18754313/edit)

### Ref :

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md
