"""
train_decoder_model_fun.py
Author      : Chris DiMattina @ FGCU (cdimattina@fgcu.edu)
Description : This program trains a linear decoder on the output of a given layer of
              a specified deep neural network. This makes extensive use of code written
              by Josiah Burnham (FGCU student).


"""

import sys
sys.path.append('../CJD/')

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from Scripts.load_dataset import LoadDataset
from ModelCode.AlexNet.AlexNet import AlexNet

batch_sz        = 100
num_epochs      = 5
num_folds       = 5
learning_rate   = 1e-04

def remove_substr(str,sub):
   if str.endswith(sub):
      return str[:-len(sub)]
   return str

def train_this_model(this_dataset='MCGILL',patch_sz=64, this_model_name='AlexNet', this_layer=1, pool_rate=2,this_l2=0.01,this_holdout_fold=5):

    working_directory = os.getcwd()
    print("CPLAB: Working directory : " + working_directory)

    path_prefix = remove_substr(working_directory, 'DecodeDeepNN')

    dir_path1 = path_prefix + "CJD/TRAINSETS/" + this_dataset + "/" + str(patch_sz) + "/H"
    dir_path2 = path_prefix + "CJD/TRAINSETS/" + this_dataset + "/" + str(patch_sz) + "/V"

    print("CPLAB: Label 1 path      : " + dir_path1)
    print("CPLAB: Label 2 path      : " + dir_path2)
    print("CPLAB: Loading dataset")

    loadDB = LoadDataset(dir_path1, dir_path2, x_shape=patch_sz, y_shape=2, num_folds=num_folds, holdout_fold=this_holdout_fold)
    (x_data, y_data), (test_x, test_y) = loadDB.load_dataset()

    # just take the second column : 1 if label 2, 0 if label 1
    y_data = y_data[:,1]
    test_y = test_y[:,1]

    print("CPLAB: Dataset loaded...Contains " + str(x_data.shape[0]) + " training stimuli and " +  str(test_x.shape[0]) + " test stimuli")
    print("CPLAB: Model selected = " + this_model_name)
    print("CPLAB: Layer selected = " + str(this_layer))
    print("CPLAB: l2 penalty     = " + str(this_l2))
    print("CPLAB: pool rate      = " + str(pool_rate))

    if(this_model_name=='AlexNet'):
        anetData = np.load("../CJD/ModelCode/AlexNet/AlexNet_WD/AlexNet_WD.npy", mmap_mode=None, allow_pickle=True,
                        fix_imports=True, encoding='ASCII')
        anet = AlexNet(anetData, 1, l2=this_l2, patch_sz=patch_sz, pool_rate=pool_rate, output=True)
        this_model = anet.get_model(this_layer)
    else:
        print("CPLAB: Model not currently supported!")


    this_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy()])
    this_model.summary()

    history = this_model.fit(x_data, y_data, batch_size=batch_sz, epochs=num_epochs, verbose=1)
    print("CPLAB: EVALUATING MODEL ON TEST BATCH...")
    test_scores = this_model.evaluate(test_x, test_y, verbose=1)

    print(("CPLAB: Saving trained model..."))
    this_fname_out  = this_dataset + "_" + str(patch_sz) + "_" + this_model_name + "_" + str(this_layer) + "_" \
                                 + str(pool_rate) + "_" + str(int(np.log10(this_l2))) + "_" + str(this_holdout_fold)
    this_fname_full = "../CJD/TRAINEDMODELS/" + this_fname_out
    this_model.save(this_fname_full)
    print("CPLAB: Saved trained model to : " + this_fname_full)
