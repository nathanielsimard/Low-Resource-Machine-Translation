ENV_NAME=onmt_helios
module load python  #
export TF_ADDONS_PY_OPS=1  #
cd
virtualenv $ENV_NAME
source $ENV_NAME/bin/activate
#pip install --user tensorflow==2.1.0
pip install tensorflow_gpu==2.1.0 #
mkdir -p tmp
cd tmp
git clone https://github.com/tensorflow/addons.git 
cd addons 
git checkout r0.9
pip install  . 
cd ../..
pip install --no-deps OpenNMT-tf==2.8.1 
pip install pyyaml 
pip install rouge 
pip install pandas
pip install sacrebleu==1.4.4
pip install sentencepiece 
pip install numpy==1.16.0 #Fix for TA setup.



