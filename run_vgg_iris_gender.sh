#!/bin/bash
run_vgg_iris_gender(){
    # FIRST ARG IS BATCH SIZE
    # SECOND ARG IS EPOCHS
    conda activate iris_keras
    for i in $(seq 6 30); do
        for d in $(seq 0 3); do
            python ./vgg_iris_gender.py -d $d -p $i -e $2 -bs $1 -o initial_both_vgg_results -lr 0.0005
            python ./vgg_iris_gender.py -d $d -p $i -e $2 --use_val -bs $1 -o initial_both_vgg_results_val -lr 0.0005
        done
        # python ./vgg_iris_gender.py -d 0 -p $i -e $2 --use_peri -bs $1 -o initial_both_peri_vgg_results -lr 0.0005
        # python ./vgg_iris_gender.py -d 0 -p $i -e $2 --use_val --use_peri -bs $1 -o initial_both_peri_vgg_results_val -lr 0.0005
    done

  # for i in $(seq 1 10); do
  #   python ./vgg_iris_gender.py -d 0 -p $i --use_peri
  #   for d in $(seq 0 3); do
  #     # echo "Testing this thing $i"
  #     python ./vgg_iris_gender.py -d $d -p $i
  #     python ./vgg_iris_gender.py -d $d -p $i --use_val
  #   done
  #   python ./vgg_iris_gender.py -d 0 -p $i --use_peri --use_val
  # done
  # for i in $(seq 10 20); do
  #   python ./vgg_iris_gender.py -d 0 -p $i --use_peri
  #   for d in $(seq 0 3); do
  #     # echo "Testing this thing $i"
  #     python ./vgg_iris_gender.py -d $d -p $i
  #     python ./vgg_iris_gender.py -d $d -p $i --use_val
  #   done
  #   python ./vgg_iris_gender.py -d 0 -p $i --use_peri --use_val
  # done
  # for i in $(seq 20 30); do
  #   python ./vgg_iris_gender.py -d 0 -p $i --use_peri
  #   for d in $(seq 0 3); do
  #     # echo "Testing this thing $i"
  #     python ./vgg_iris_gender.py -d $d -p $i
  #     python ./vgg_iris_gender.py -d $d -p $i --use_val
  #   done
  #   python ./vgg_iris_gender.py -d 0 -p $i --use_peri --use_val
  # done
}
