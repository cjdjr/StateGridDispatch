#!/bin/bash
# iPipe unittest scripts

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib64
alias python=/home/Python-3.6.0/python

parentdir=$(dirname `pwd`)
export PYTHONPATH="${PYTHONPATH}:$parentdir"

for file in unittest/*_test.py; do
    python $file
done
