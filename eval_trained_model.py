"""
eval_trained_model.py

"""
from keras.models import load_model
import numpy as np
from scipy.io import savemat

model_dir       = './/TRAINEDMODELS//'
model_fname     = 'MCGILL_64_AlexNet_4_2_5'

this_model      = load_model(model_dir + model_fname)
this_model.summary()

linear_layer    = this_model.get_layer(name='Output')
linear_weights  = np.array(linear_layer.get_weights()[0],dtype=np.float32)

mdict           = {'linear_weights':linear_weights}

savemat("test.mat",mdict)
