#inspired from this link https://github.com/Accenture/serverless-ephemeral/blob/master/docs/build-tensorflow-package.md

yum -y install epel-release -y
yum -y install gcc gcc-c++ python-pip python-devel atlas atlas-devel gcc-gfortran openssl-devel libffi-devel -y

virtualenv ~/venvs/tensorflow
source ~/venvs/tensorflow/bin/activate

pip install --upgrade pip
pip install --no-cache-dir tensorflow==1.15.2

touch $VIRTUAL_ENV/lib/python3.6/site-packages/google/__init__.py

cd $VIRTUAL_ENV/lib/python3.6/site-packages

rm -rf easy_install* pip* setup_tools* wheel* tensorboard*
rm -rf tensorflow_core/contrib/* tensorflow_core/include/unsupported/*

find . -name "*.so" | xargs strip
pushd .
zip -r9q /var/task/requirements.zip * --exclude \*.pyc
popd
