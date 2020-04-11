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
pip install sacrebleu==1.3.5
pip install sentencepiece 


