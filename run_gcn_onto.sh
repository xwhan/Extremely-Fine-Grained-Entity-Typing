#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python main.py $2 -lstm_type single -enhanced_mention -goal onto -gcn
