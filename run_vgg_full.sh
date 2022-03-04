#!/bin/bash
run_vgg_full(){
    conda activate iris_keras

    for nPart in $(seq 1 1); do  # CHANGE NUMBERS APPROPRIATELY
        for d in $(seq 0 3); do
            python ./vgg_full.py -pf experiments/full_vgg/normalized_postfix_step_by_step/params.json -d "$d" -p "$nPart" -sbs
        done
        # python ./vgg_full.py -pf experiments/full_vgg/step_by_step_peri_fix_2_lesslr/params.json -d 0 -p "$nPart" --use_fix -sbs
    done

}
