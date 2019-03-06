#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py $1 -lstm_type single -enhanced_mention -data_setup joint -add_crowd -multitask -gcn -model_debug