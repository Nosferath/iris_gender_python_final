#!/bin/bash
run_all_remove_pairs_tests(){
    conda activate iris_keras

    # for bins in $(seq 1 4); do
    #     CUDA_VISIBLE_DEVICES="" python ./vgg_lsvm.py 15 -p 30 --use_botheyes --no_parallel --use_ndiris --out_folder "experiments/ndiris_vgg_lsvm_pairs_removebad_$bins/" --use_pairs -rm $bins
    #     CUDA_VISIBLE_DEVICES="" python ./vgg_lsvm.py 15 -p 30 --use_botheyes --no_parallel --out_folder "experiments/gfi_vgg_lsvm_pairs_removebad_$bins/" --use_pairs -rm $bins
    # done

    for nPart in $(seq 1 30); do  # CHANGE NUMBERS APPROPRIATELY
        for bins in 6 8; do # for bins in $(seq 1 4); do
            for d in $(seq 2 3); do
                python ./vgg_full.py -pf "experiments/ndiris_full_vgg_pairs_removebad_$bins/initial_test/params.json" -d "$d" -p "$nPart" --use_ndiris --use_pairs -rm $bins
            done
            for d in $(seq 0 3); do
                python ./vgg_full.py -pf "experiments/gfi_full_vgg_pairs_removebad_$bins/initial_test/params.json" -d "$d" -p "$nPart" --use_pairs -rm $bins
            done
        done
    done
    # for nPart in $(seq 1 1); do  # CHANGE NUMBERS APPROPRIATELY
    #     for d in $(seq 2 3); do
    #         python ./vgg_full.py -pf experiments/ndiris_full_vgg_pairs/initial_test_sbs/params.json -d "$d" -p "$nPart" -sbs --use_ndiris --use_pairs
    #     done
    # done

}
