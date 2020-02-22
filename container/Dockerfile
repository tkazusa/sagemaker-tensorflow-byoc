FROM tensorflow/tensorflow:nightly-gpu

RUN pip install sagemaker-containers

# Copies the training code inside the container
# opt/
# ├ /ml
#    ├ code
#      ├ tf_model.py
#      └ train.py
# └ ml/
#  ├ input/
#    └ data/
#  └ model/
COPY tf_codes /opt/ml/code/

# Defines train.py as script entry point
# SAGEMAKER_PROGRAM はデフォルトでは /opt/ml/code/ を以下を指す。
# 絶対パスを指定しても良い。
ENV SAGEMAKER_PROGRAM train.py