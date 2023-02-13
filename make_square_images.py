"""
make_square_images.py
by Chris DiMattina @ FGCU
Description    : This script reads in RGB .tif images from the McGill Database,
                 which are cropped to square and resized to 256x256.
Usage          : python make_square_images.py
Input images   : ./TexturesUniform/
Output images  : ./MCGILL256/
"""

#  Crop images to their smaller dimensions (try to keep most details)
from PIL import Image
import os

# Directories containing input and output images
input_path      = "../CJD/STORAGE/TexturesUniform/"
output_path     = "../CJD/MCGILL256/"

# Get filenames
file_names      = os.listdir(input_path)  # make list of files in the input path
num_files       = len(file_names)

# Announce number of files
print("CPLAB: Found N = " + str(num_files) + " image files in " + input_path)

# Loop through files. Load each one as grayscale, crop to square, resize tp 256x256
for this_file in file_names:
    fullpath = os.path.join(input_path,this_file)
    if(os.path.isfile(fullpath)):
        im          = Image.open(fullpath).convert('L')
        image_crop  = im.crop((0,0,576,576))
        image_resz  = image_crop.resize((256,256))
        outfname     = os.path.join(output_path,this_file)
        image_resz.save(outfname,'TIFF')
        print("CPLAB: Image saved to " + outfname)
