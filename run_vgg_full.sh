#!/bin/bash
run_vgg_full(){
    # FIRST ARG IS BATCH SIZE
    # SECOND ARG IS EPOCHS
    conda activate iris_keras

    for i in $(seq 1 1); do  # CHANGE NUMBERS APPROPRIATELY
        for d in $(seq 0 3); do
            # python ./vgg_full.py -d $d -p $i -e $2 -bs $1 -o step_by_step_vgg -lr 0.0005
            python ./vgg_full.py -d $d -p $i -e $2 --use_val -bs $1 -o step_by_step_vgg -lr 0.0005 -sbs
        done
        # python ./vgg_full.py -d 0 -p $i -e $2 --use_peri -bs $3 -o second_both_peri_vgg_results -lr 0.0005
        # python ./vgg_full.py -d 0 -p $i -e $2 --use_val --use_peri -bs $3 -o step_by_step_vgg_peri -lr 0.0005 -sbs
    done

  # for i in $(seq 1 10); do
  #   python ./vgg_full.py -d 0 -p $i --use_peri
  #   for d in $(seq 0 3); do
  #     # echo "Testing this thing $i"
  #     python ./vgg_full.py -d $d -p $i
  #     python ./vgg_full.py -d $d -p $i --use_val
  #   done
  #   python ./vgg_full.py -d 0 -p $i --use_peri --use_val
  # done
  # for i in $(seq 10 20); do
  #   python ./vgg_full.py -d 0 -p $i --use_peri
  #   for d in $(seq 0 3); do
  #     # echo "Testing this thing $i"
  #     python ./vgg_full.py -d $d -p $i
  #     python ./vgg_full.py -d $d -p $i --use_val
  #   done
  #   python ./vgg_full.py -d 0 -p $i --use_peri --use_val
  # done
  # for i in $(seq 20 30); do
  #   python ./vgg_full.py -d 0 -p $i --use_peri
  #   for d in $(seq 0 3); do
  #     # echo "Testing this thing $i"
  #     python ./vgg_full.py -d $d -p $i
  #     python ./vgg_full.py -d $d -p $i --use_val
  #   done
  #   python ./vgg_full.py -d 0 -p $i --use_peri --use_val
  # done
}
