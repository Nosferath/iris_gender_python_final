#!/bin/bash
run_vgg_lsvm_pairs(){
    conda activate iris_keras

    python ./vgg_lsvm.py 15 -p 30 --use_botheyes --no_parallel --use_pairs --out_folder "experiments/vgg_lsvm_pairs/"

}
