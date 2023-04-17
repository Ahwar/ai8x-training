#!/bin/sh
python train.py --name train_exp007_0-06unknown_kws20_v3_400e_adamo_0-001lr_256b_nobias_1classes_KWS-1dataset --epochs 400 \
    --optimizer Adam --lr 0.001 --wd 0 --deterministic  \
    --qat-policy None --compress policies/schedule_kws20.yaml --model ai85kws20netv3 --dataset KWS_1\
    --show-train-accuracy "full" --enable-tensorboard \
    --confusion --save-confusion --param-hist --pr-curves --embedding --print-freq 100 \
    -j 5 --device MAX78000 "$@"

systemctl suspend #--effective-train-size 0.90 --effective-valid-size 0.01 --effective-test-size 0.09\
