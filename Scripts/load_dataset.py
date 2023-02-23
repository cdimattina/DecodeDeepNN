#==========================================================
# File    : load_dataset.py
# Author  : Josiah Burnham (10/20/22)
# Org     : FGCU (Visual Perception Lab)
# Purpose : implementation of the LoadDataset class
#==========================================================

import numpy as np
import scipy.io
import os

class LoadDataset:
    """ Load a dataset that is split between two directories
        Whose data is split into folds by the file directories
        themselves for each of the two directores
        (ex. file dir 1:dataset1 has dirs fold1-fold5, each of 
        which contain a .mat. file dir 2: dataset2, has the 
        same layout as dataset1. I want to load one fold 
        from dataset 1, then the same fold from datset 2
        before doing the same thing again with the datasets
        for the rest of the folds )
    """

    def __init__(self, dir_path1, dir_path2, x_shape, y_shape, num_folds, holdout_fold, isSplit=True):
        """Load a dataset from .mat files and return them as a numpy array
           in the form: (features, labels)

        Args:
            dir_path (string): directory path for train data
            x_shape (int, optional): the shape of the features in the data. Defaults to 1600.
            y_shape (int, optional): the shape of the labels in the data. Defaults to 2.
        """

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.dir_path1 = dir_path1
        self.dir_path2 = dir_path2
        self.num_folds = num_folds
        self.holdout_fold = holdout_fold
        self.isSplit = isSplit

    def load_dataset(self):
        """load the dataset from .mat files

        Returns:
            (numpy array, numpy array): the x and y components to the dataset
        """

        x, y = self.__load_data()
        return (x,y)

    def __load_data(self):
        """loads the datafiles and create a numpy array

        Returns:
            numpy array: numpy array with shape: 
                            [number of images, image height , image width]
        """
        if self.isSplit:
            dataset_all_x = np.ndarray(shape=(0, self.x_shape, self.x_shape), dtype=float)
            holdout_x = np.ndarray(shape=(0, self.x_shape, self.x_shape), dtype=float)
        else:
            dataset_all_x = np.ndarray(shape=(0, self.x_shape), dtype=float)
            holdout_x = np.ndarray(shape=(0, self.x_shape), dtype=float)

        dataset_all_y = np.ndarray(shape=(0, self.y_shape), dtype=float)
        holdout_y = np.ndarray(shape=(0, self.y_shape), dtype=float)

        dirs_in_train_dir = [f for f in os.listdir(self.dir_path1)
                                    if os.path.isdir(os.path.join(self.dir_path1, f))]

        files = []
        holdouts=[]
        for directory in dirs_in_train_dir:
            if directory != f"Fold_{self.holdout_fold}":
                files_in_dir1 = [f for f in os.listdir(self.dir_path1+"/"+directory)
                                    if os.path.isfile(os.path.join(self.dir_path1+"/"+directory, f))]
                files_in_dir2 = [f for f in os.listdir(self.dir_path2+"/"+directory)
                                    if os.path.isfile(os.path.join(self.dir_path2+"/"+directory, f))]

                files.append(f"{self.dir_path1}/{directory}/{files_in_dir1[0]}")
                files.append(f"{self.dir_path2}/{directory}/{files_in_dir2[0]}")
            else:
                files_in_dir1 = [f for f in os.listdir(self.dir_path1+"/"+directory)
                                    if os.path.isfile(os.path.join(self.dir_path1+"/"+directory, f))]
                files_in_dir2 = [f for f in os.listdir(self.dir_path2+"/"+directory)
                                    if os.path.isfile(os.path.join(self.dir_path2+"/"+directory, f))]

                holdouts.append(f"{self.dir_path1}/{directory}/{files_in_dir1[0]}")
                holdouts.append(f"{self.dir_path2}/{directory}/{files_in_dir2[0]}")




        for i in files:
            datafile_x = np.float32(scipy.io.loadmat(i)["image_patches"])
            datafile_y = np.float32(scipy.io.loadmat(i)["category_labels"])

            dataset_all_x = np.append(datafile_x, dataset_all_x, axis=0)
            dataset_all_y = np.append(datafile_y, dataset_all_y, axis=0)

        for i in holdouts:
            holdout_x_data = np.float32(scipy.io.loadmat(i)["image_patches"])
            holdout_y_data = np.float32(scipy.io.loadmat(i)["category_labels"])

            holdout_x = np.append(holdout_x_data, holdout_x, axis=0)
            holdout_y = np.append(holdout_y_data, holdout_y, axis=0)

        return (dataset_all_x, dataset_all_y), (holdout_x, holdout_y)