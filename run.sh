#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python main.py $2 -lstm_type single -enhanced_mention -data_setup joint -add_crowd -multitask -incon_w $3 -add_regu
