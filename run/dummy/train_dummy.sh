#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:/home/zhaos5/projs/wsd/wsd_code"
nohup python ../model/train.py  -env aws -out dummy>dummy_train.log
