#!/bin/bash
# This script is for running vgg lsvm tests using pairs, with and without removal
# using 4 specific thresholds, and 30 partitions on each
run_vgg_lsvm_pairs_for_anova(){
    conda activate iris_keras

    for thresh in 0.05 0.075 0.1 0.125; do
	    CUDA_VISIBLE_DEVICES="" python ./vgg_lsvm.py 15 -p 30 --use_botheyes --no_parallel --use_pairs -t "$thresh" --out_folder "experiments/vgg_lsvm_pairs_thresh_full/$thresh/"
	    CUDA_VISIBLE_DEVICES="" python ./vgg_lsvm.py 15 -p 30 --use_botheyes --no_parallel --use_pairs -t "$thresh" --out_folder "experiments/vgg_lsvm_pairs_thresh_full_removebad_09/$thresh/" -rm
    done
    # CUDA_VISIBLE_DEVICES="" python ./vgg_lsvm.py 15 -p 30 --use_botheyes --no_parallel --use_pairs --out_folder "experiments/vgg_lsvm_pairs/"

}
