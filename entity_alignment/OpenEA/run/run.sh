#!/usr/bin/env bash

python main_from_args.py ./args/bootea_args_15K.json EN_FR_15K_V2 721_5fold/2/ > aligne_plus_log2
python main_from_args.py ./args/bootea_args_15K.json EN_FR_15K_V2 721_5fold/3/ > aligne_plus_log3
python main_from_args.py ./args/bootea_args_15K.json EN_FR_15K_V2 721_5fold/4/ > aligne_plus_log4
python main_from_args.py ./args/bootea_args_15K.json EN_FR_15K_V2 721_5fold/5/ > aligne_plus_log5