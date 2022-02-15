#!/bin/bash
run_vgg_full(){
    conda activate iris_keras

    for nPart in $(seq 1 1); do  # CHANGE NUMBERS APPROPRIATELY
        # for d in $(seq 0 3); do
            # python ./vgg_full.py -pf step_by_step_vgg_4_longer2/params.json -d "$d" -p "$nPart" -sbs
        # done
        python ./vgg_full.py -pf step_by_step_vgg_peri_4/params.json -d 0 -p "$nPart" -sbs --use_peri
    done

}
