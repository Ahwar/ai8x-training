unknown_weight=0.82  # weight for the first experiement
echo $unknown_weight > "unknown.txt"  # save unknownweight to the unknown.txt file
epochs=5 # number of epochs

# how much the weightage should be increased after every experiment
increment=0.02


for ((i=45; i<=45; i++)); do
    
    exp=$i
    # expriment name
    exp_name="resume_exp0010-${exp}_0${unknown_weight}unknown_kws20_v3_${epochs}e_adamo_0-0001lr_256b_nobias_6classes_KWSdataset"
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
        -j 4 --device MAX78000 "$@"
    #!/bin/bash

    # increment unknown weightage
    unknown_weight=$(echo $unknown_weight + $increment | bc)
    echo $unknown_weight > unknown.txt

done