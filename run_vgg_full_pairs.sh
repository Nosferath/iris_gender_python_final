#!/bin/bash
run_vgg_full_pairs(){
    conda activate iris_keras

    for nPart in $(seq 1 30); do  # CHANGE NUMBERS APPROPRIATELY
        for d in $(seq 0 3); do
            python ./vgg_full.py -pf experiments/full_vgg_pairs/initial_test/params.json -d "$d" -p "$nPart" --use_pairs
        done
    done
    # for nPart in $(seq 1 1); do  # CHANGE NUMBERS APPROPRIATELY
    #     for d in $(seq 0 3); do
    #         python ./vgg_full.py -pf experiments/full_vgg_pairs/initial_test_sbs_2/params.json -d "$d" -p "$nPart" --use_pairs -sbs
    #     done
    # done

}
