#!/bin/bash

# cd /app/code
# python download_data.py

# One test run to show the code - all is running long here.

python pipeline.py Cifar10 --config config/Cifar10/SDKL.yaml --seed_everything $((1234 + 1)) --trainer.default_root_dir ../results/Cifar10/SDKL/

exit 1

# Repetitions
n=3

# --------------------------------------------------
# Baselines
# --------------------------------------------------

# Baseline Methods
declare -a methods=(
"SVDKL"
"AVDKL"
"DUE"
"DNN"
"SVGP"
"MCdropout"
"LLBayes"
)

dataset=Cifar10

for method in "${methods[@]}"
do
    for i in $(seq 0 $((n-1))); do
        python ref_pipeline.py $dataset --config config/$dataset/all_baselines.yaml --seed_everything $((1234 + $i)) --trainer.default_root_dir lightning_logs/Cifar10/$method/ --model.name $method
    done
done

dataset=Cifar100

for method in "${methods[@]}"
do
    for i in $(seq 0 $((n-1))); do
        if [ $method = "DUE" ]
        then
            python ref_pipeline.py $dataset --config config/$dataset/all_baselines.yaml --seed_everything $((1234 + $i)) --trainer.default_root_dir lightning_logs/Cifar100/$method/ --model.name $method --data.batch_size 64

        else
            python ref_pipeline.py $dataset --config config/$dataset/all_baselines.yaml --seed_everything $((1234 + $i)) --trainer.default_root_dir lightning_logs/Cifar100/$method/ --model.name $method

        fi
    done
done

dataset=TinyImagenet

for method in "${methods[@]}"
do
    for i in $(seq 0 $((n-1))); do
            python ref_pipeline.py $dataset --config config/$dataset/$method.yaml --seed_everything $((15 + $i)) --trainer.default_root_dir lightning_logs/$dataset/$method/ --model.name $method

    done
done

dataset=Imagenet100

for method in "${methods[@]}"
do
    for i in $(seq 0 $((n-1))); do
        python ref_pipeline.py $dataset --config config/$dataset/$method.yaml --seed_everything $((12 + $i)) --trainer.default_root_dir lightning_logs/$dataset/$method/ --model.name $method

    done
done

# --------------------------------------------------
# SDKL & CDKL
# --------------------------------------------------

declare -a datasets=(
"Imagenet100"
"TinyImagenet"
"Cifar10"
"Cifar100"
)

declare -a methods=(
"SDKL"
"CDKL"
)


for dataset in "${datasets[@]}"
do
    for method in "${methods[@]}"
    do
        for i in $(seq 0 $((n-1))); do

            python pipeline.py $dataset --config config/$dataset/$method.yaml --seed_everything $((1234 + $i)) --trainer.default_root_dir lightning_logs/$dataset/$method/

        done
    done
done
