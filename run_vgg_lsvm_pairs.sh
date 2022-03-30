#!/bin/bash
run_vgg_lsvm_pairs(){
    conda activate iris_keras

    CUDA_VISIBLE_DEVICES="" python ./vgg_lsvm.py 15 -p 30 --use_botheyes --no_parallel --use_pairs --out_folder "experiments/vgg_lsvm_pairs_second_half/"
    # CUDA_VISIBLE_DEVICES="" python ./vgg_lsvm.py 15 -p 30 --use_botheyes --no_parallel --use_pairs --out_folder "experiments/vgg_lsvm_pairs/"

}
