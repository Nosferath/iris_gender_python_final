#!/bin/bash
run_vgg_lsvm_pairs(){
    conda activate iris_keras

    for thresh in $(seq 0.03 0.005 0.15); do
	    CUDA_VISIBLE_DEVICES="" python ./vgg_lsvm.py 15 -p 5 --use_botheyes --no_parallel --use_pairs -t "$thresh" --out_folder "experiments/vgg_lsvm_pairs_thresh/$thresh/"
    done
    # CUDA_VISIBLE_DEVICES="" python ./vgg_lsvm.py 15 -p 30 --use_botheyes --no_parallel --use_pairs --out_folder "experiments/vgg_lsvm_pairs/"

}
