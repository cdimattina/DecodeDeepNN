"""
make_edge_train_set.py
by Chris DiMattina

Description     : makes a training set for texture-based edge orientation identification task
Usage           : make_edge_train_set.py <base_set> <patch_sz> <num_train> <num_folds> <set_num>
Inputs          : <base_set>  : 'MCGILL' or 'BRODATZ'
                  <patch_sz>  : size of image patches
                  <num_train> : number of training samples
                  <num_folds> : how many folds
                  <set_num>   : set number
Data            : Assumes directories comprised of 256x256 grayscale images in either .png or .tif format

"""

import sys
import os
import numpy as np
from PIL import Image
from scipy.io import savemat

# Constants
inputs_expected = 6
img_dim         = 256
max_sz          = 128
label_dict      = ['H','V']

# location of texture databases
mcgill_dir      = '../CJD/MCGILL256/'
brodatz_dir     = '../CJD/BRODATZ256/'
train_set_dir   = '../CJD/TRAINSETS/'

def check_inputs(argin):

    if(len(argin) != inputs_expected):
        error_code = 1
    elif(argin[1]!='MCGILL' and argin[1]!='BRODATZ'):
        error_code = 2
    elif(int(argin[2]) > max_sz):
        error_code = 3
    else:
        error_code = 0

    return error_code

def print_error_message(error_code):
    if(error_code==1):
        error_message = "CPLAB: Usage: make_edge_train_set.py <text_set> <patch_sz> <num_train> <num_folds> <set_num>"
    elif(error_code==2):
        error_message = "CPLAB: Invalid texture set. Permissible value = 'MCGILL', 'BRODATZ'"
    elif(error_code==3):
        error_message = "CPLAB: Image patch size must be less than 128 pixels"

    print(error_message)

def merge_patches(tex_patch_1,tex_patch_2,patch_sz, this_label):

    patch_sz2 = int(patch_sz/2)
    this_patch = np.zeros((patch_sz,patch_sz))
    if(this_label==0):       # horizontal edge
        t_half = tex_patch_1[0:(patch_sz2),:]
        b_half = tex_patch_2[(patch_sz2):patch_sz,:]
        t_half = t_half - np.mean(t_half)
        t_half = t_half/np.std(t_half)
        b_half = b_half - np.mean(b_half)
        b_half = b_half/np.std(b_half)
        this_patch[0:(patch_sz2),:] = t_half
        this_patch[(patch_sz2):patch_sz,:] = b_half
    elif(this_label==1):     # vertical edge
        l_half = tex_patch_1[:,0:(patch_sz2)]
        r_half = tex_patch_2[:,(patch_sz2):patch_sz]
        l_half = l_half - np.mean(l_half)
        l_half = l_half/np.std(l_half)
        r_half = r_half - np.mean(r_half)
        r_half = r_half/np.std(r_half)
        this_patch[:,0:(patch_sz2)] = l_half
        this_patch[:,(patch_sz2):patch_sz] =  r_half

    return this_patch

def make_training_set(text_dir,fold_ind,this_label,patch_sz,num_train):

    # initialize outputs to empty
    train_images = []
    train_cat    = []

    # get the complete file list
    file_list = os.listdir(text_dir)
    # get size of fold
    num_raw_images = len(fold_ind)
    # load the set of images comprising this fold into an np.array
    full_image_array = np.zeros((num_raw_images, img_dim, img_dim),dtype=np.float32)
    i = 0
    for this_ind in fold_ind:
        this_file_path  = text_dir + file_list[this_ind]
        this_image_obj  = Image.open(this_file_path)
        this_image      = np.asarray(this_image_obj)
        full_image_array[i,:,:] = this_image
        i = i + 1

    # create train_images and training category (one-hot)
    train_images = np.zeros((num_train,patch_sz,patch_sz),dtype=np.float32)
    train_cat    = np.zeros((num_train,2),dtype=int)
    train_cat[:,this_label] = 1

    for this_patch in range(num_train):
        # For every patch choose two different textures to sample from
        temp_permute = np.random.permutation(num_raw_images)
        ind_1 = temp_permute[0]
        ind_2 = temp_permute[1]

        # For each of the two chosen images, sample from it randomly.
        # This is done by finding random location for the upper left
        # corner in the range 0 to (img_dim - patch_sz -1) for both row
        # and column dimensions
        r_1 = np.random.randint(0,img_dim-patch_sz)
        c_1 = np.random.randint(0,img_dim-patch_sz)
        r_2 = np.random.randint(0,img_dim-patch_sz)
        c_2 = np.random.randint(0,img_dim-patch_sz)

        # Now, pick two random images from the full_image_array and then sample
        tex_patch_1 = full_image_array[ind_1,r_1:(r_1 + patch_sz),c_1:(c_1 + patch_sz)]
        tex_patch_2 = full_image_array[ind_2,r_2:(r_2 + patch_sz),c_2:(c_2 + patch_sz)]

        this_tex_edge = merge_patches(tex_patch_1,tex_patch_2,patch_sz,this_label)

        train_images[this_patch,:,:] = this_tex_edge

    return (train_images, train_cat)

# Check for the correct number of inputs
def main():
    error_code = check_inputs(sys.argv)
    if(error_code == 0):
        print("CPLAB: Valid inputs...")
        text_set    = sys.argv[1]
        patch_sz    = int(sys.argv[2])
        num_train   = int(sys.argv[3])
        num_folds   = int(sys.argv[4])
        set_num     = int(sys.argv[5])

        print("CPLAB: " + text_set + " texture set...")
        if(text_set == 'MCGILL'):
            text_dir = mcgill_dir
        elif(text_set == 'BRODATZ'):
            text_dir = brodatz_dir

        file_list   = os.listdir(text_dir)
        num_files   = len(file_list)
        files_per_fold = int(num_files/num_folds)

        print("CPLAB: Texture directory: " + text_dir )
        print("CPLAB: Patch size: " + str(patch_sz))
        print("CPLAB: Number of files: " + str(num_files))
        print("CPLAB: Number of folds: " + str(num_folds))
        print("CPLAB: Set number: " + str(set_num))
        print("CPLAB: File per fold: " + str(files_per_fold))
        print("CPLAB: " + str(num_train) + " training stimuli per fold..." )

        for this_fold in range(num_folds):
            st_ind = this_fold*files_per_fold
            sp_ind = (this_fold + 1)*files_per_fold
            fold_ind = range(st_ind,sp_ind)
            print("CPLAB: Making fold " + str(this_fold+1))
            for this_label in range(2):
                this_out_dir = train_set_dir + text_set + "/" + str(patch_sz) +"/" + label_dict[this_label] + "/Fold_" + str(this_fold+1) + "/"
                (train_images,train_cat) = make_training_set(text_dir,fold_ind,this_label,patch_sz,num_train)
                outfile_name = label_dict[this_label] + "_" + str(patch_sz) + "_" + str(set_num) + ".mat"
                full_outfile_name = this_out_dir + outfile_name
                mdict = {'image_patches': train_images, 'category_labels': train_cat}
                savemat(full_outfile_name,mdict)

    else:
        print("CPLAB: INPUT ERROR...")
        print_error_message(error_code)

# call main program
main()





