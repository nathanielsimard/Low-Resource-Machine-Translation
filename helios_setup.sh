ENV_NAME=onmt_helios_2
module load python  #
export TF_ADDONS_PY_OPS=1  #
cd
virtualenv $ENV_NAME
source $ENV_NAME/bin/activate
cd -
module load python  
pip install -r requirements/onmt.txt


