"""
train_decoder_shell.py
Author      : Chris DiMattina @ FGCU (cdimattina@fgcu.edu)
Description : Trains a decoder model
"""

this_dataset    = 'MCGILL'
this_model_name = 'AlexNet'
layer_list      = [1, 4, 4, 7]
pool_rate_list  = [2, 1, 2, 1]
patch_sz_list   = [16,24,32]
l2_list         = [-3, -2, -1, 0, 1, 2, 3]
fold_list       = [1,2,3,4,5]

# number of layers, l2, folds
num_layers      = len(layer_list)
num_patch_sz    = len(patch_sz_list)
num_l2          = len(l2_list)
num_folds       = len(fold_list)

import train_decoder_model_fun as tdm
import numpy as np

print("...MODEL = " + this_model_name)
print("...DATA  = " + this_dataset)

for layer_ind in range(num_layers):
    this_layer      = layer_list[layer_ind]
    this_pool_rate  = pool_rate_list[layer_ind]
    for patch_sz_ind in range(num_patch_sz):
        this_patch_sz = patch_sz_list[patch_sz_ind]
        for l2_ind in range(num_l2):
            this_l2 = np.power(10,l2_list[l2_ind])
            for fold_ind in range(num_folds):

                this_holdout_fold = fold_list[fold_ind]
                print("...Layer      = " + str(this_layer) + ", Pool rate = " + str(this_pool_rate))
                print("...Patch Size = " + str(this_patch_sz))
                print("...L2         = " + str(this_l2))
                print("...Holdback fold = " + str(this_holdout_fold))

                tdm.train_this_model(this_dataset,this_patch_sz, this_model_name, this_layer, this_pool_rate,this_l2,this_holdout_fold)