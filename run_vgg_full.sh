#!/bin/bash
run_vgg_full(){
    conda activate iris_keras

    for nPart in $(seq 1 1); do  # CHANGE NUMBERS APPROPRIATELY
        for d in $(seq 0 3); do
            # python ./vgg_full.py -d $d -p $i -e $2 -bs $1 -o step_by_step_vgg -lr 0.0005
            python ./vgg_full.py -pf step_by_step_vgg_4_longer/params.json -d $d -p "$nPart" -sbs
        done
        # python ./vgg_full.py -d 0 -p $i -e $2 --use_peri -bs $3 -o second_both_peri_vgg_results -lr 0.0005
        # python ./vgg_full.py -d 0 -p $i -e $2 --use_val --use_peri -bs $3 -o step_by_step_vgg_peri -lr 0.0005 -sbs
    done

}
