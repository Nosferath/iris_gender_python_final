#!/bin/bash
run_vgg_iris_gender(){
  conda activate iris_keras
  for i in $(seq 1 10); do
    python ./vgg_iris_gender.py -d 0 -p $i --use_peri
    for d in $(seq 0 3); do
      # echo "Testing this thing $i"
      python ./vgg_iris_gender.py -d $d -p $i
      python ./vgg_iris_gender.py -d $d -p $i --use_val
    done
    python ./vgg_iris_gender.py -d 0 -p $i --use_peri --use_val
  done
  for i in $(seq 10 20); do
    python ./vgg_iris_gender.py -d 0 -p $i --use_peri
    for d in $(seq 0 3); do
      # echo "Testing this thing $i"
      python ./vgg_iris_gender.py -d $d -p $i
      python ./vgg_iris_gender.py -d $d -p $i --use_val
    done
    python ./vgg_iris_gender.py -d 0 -p $i --use_peri --use_val
  done
  for i in $(seq 20 30); do
    python ./vgg_iris_gender.py -d 0 -p $i --use_peri
    for d in $(seq 0 3); do
      # echo "Testing this thing $i"
      python ./vgg_iris_gender.py -d $d -p $i
      python ./vgg_iris_gender.py -d $d -p $i --use_val
    done
    python ./vgg_iris_gender.py -d 0 -p $i --use_peri --use_val
  done
}
