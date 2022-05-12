#!/bin/bash
run_vgg_full_ndiris(){
    conda activate iris_keras

    for nPart in $(seq 1 30); do  # CHANGE NUMBERS APPROPRIATELY
        for d in $(seq 2 3); do
            python ./vgg_full.py -pf experiments/ndiris_full_vgg/initial_test/params.json -d "$d" -p "$nPart" --use_ndiris
        done
    done
    # for nPart in $(seq 1 1); do  # CHANGE NUMBERS APPROPRIATELY
    #     for d in $(seq 2 3); do
    #         python ./vgg_full.py -pf experiments/ndiris_full_vgg/initial_test_sbs/params.json -d "$d" -p "$nPart" -sbs --use_ndiris
    #     done
    # done

}
