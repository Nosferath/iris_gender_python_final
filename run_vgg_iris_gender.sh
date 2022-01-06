#!/bin/bash
run_vgg_iris_gender(){
  conda activate iris_keras
  for d in $(seq 0 1); do
    for i in $(seq 1 $1); do
      # echo "Testing this thing $i"
      python ./vgg_iris_gender.py -d $d -p $i --use_peri
    done
  done
}
