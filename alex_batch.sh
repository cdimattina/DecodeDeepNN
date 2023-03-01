#!/bin/bash

layer_list=(1 4 7 9 11 15 17)
lambda_list=(-3 -2 -1 0 1 2 3)
sz_list=(16 24 32 64)
train_list=(MCGILL BRODATZ)

for i in "${layer_list[@]}"; do
  echo "layer=$i"
done