#!/bin/sh
python train.py --name "evaluate_exp0010-45_0-82unknown_kws20_v3_55e_adamo_0-0001lr_256b_nobias_6classes_KWSdataset" --model ai85kws20netv3 --dataset KWS  --evaluate\
 --exp-load-weights-from /home/ahwar/Desktop/vad/ai8x-training/logs/train_exp0010-45_0-82unknown_kws20_v3_55e_adamo_0-0001lr_256b_nobias_6classes_KWSdataset___2023.05.21-140516/epoch__54_train_exp0010-45_0-82unknown_kws20_v3_55e_adamo_0-0001lr_256b_nobias_6classes_KWSdataset_checkpoint.pth.tar \
 --confusion --save-confusion --device MAX78000 "$@"
