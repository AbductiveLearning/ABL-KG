#!/usr/bin/env bash

python main.py --dataset_division 721_5fold/1/ > kg_log1
python main.py --dataset_division 721_5fold/2/ > kg_log2
python main.py --dataset_division 721_5fold/3/ > kg_log3
python main.py --dataset_division 721_5fold/4/ > kg_log4
python main.py --dataset_division 721_5fold/5/ > kg_log5
