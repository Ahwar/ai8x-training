unknown_weight=0.86  # weight for the first experiement
echo $unknown_weight > "unknown.txt"  # save unknownweight to the unknown.txt file
epochs=1 # number of epochs

# how much the weightage should be increased after every experiment
increment=0.02


for ((i=45; i<=45; i++)); do
    
    exp=$i
    # expriment name
    exp_name="train_exp0014-${exp}_transferlearning6_0${unknown_weight}unknown_kws20_v3_kardome_${epochs}e_adamo_0-0001lr_256b_nobias_1classes_KWSkardomedataset-noothers"
    exp_name=${exp_name//./-} # replace . with -

    echo "=========== new experiement ${exp}================="
    echo $exp_name
    echo $unknown_weight

    # training command
    python train.py --name $exp_name --epochs $epochs \
        --optimizer Adam --lr 0.0001 --wd 0 --deterministic --seed 42 \
        --qat-policy None --compress policies/schedule_kws20.yaml --model ai85kws20netv3 --dataset KWS\
        --show-train-accuracy "full" --enable-tensorboard \
        --confusion --save-confusion --param-hist --pr-curves --embedding --print-freq 100 \
        --transfer-learning-from "logs/train_exp0011-45_00-82unknown_kws20_v3_220e_adamo_0-0001lr_256b_nobias_6classes_KWSdataset___2023.06.07-175352/epoch__181_train_exp0011-45_00-82unknown_kws20_v3_220e_adamo_0-0001lr_256b_nobias_6classes_KWSdataset_best.pth.tar" \
        -j 5 --device MAX78000 --model-output-shape 7 "$@"
    #!/bin/bash

    # increment unknown weightage
    unknown_weight=$(echo $unknown_weight + $increment | bc)
    echo $unknown_weight > unknown.txt

done
# systemctl suspend