from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Activation, Flatten
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from sklearn import linear_model
import click
from pathlib import Path

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def extract_feature(input, output = None, verbose = False):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    model.load_weights('./weights/vgg_face_weights.h5')
    
    vgg_face_descriptor = Model(
    inputs=model.layers[0].input,
    outputs=model.layers[-2].output)

    if verbose:
    	print('converting images from %s'%(input))
    face_array_representation = {}
    if os.path.isdir(input):
        for img_name in tqdm(os.listdir(input)):
            face_array_representation[img_name] = vgg_face_descriptor.predict(preprocess_image('{}/{}'.format(input,img_name)))[0,:]
    if os.path.isfile(input):
        img_name = input.split('/')[-1]
        face_array_representation[img_name] = vgg_face_descriptor.predict(preprocess_image('{}'.format(input)))[0,:]
    if output == None:
        return face_array_representation
    # save array to pickle
    if verbose:
    	print('output array to {}'.format(output))
    with open(output,'wb') as f:
        pickle.dump(face_array_representation, f)

def load_object(file):
    with open(file, 'rb') as f:
        res = pickle.load(f)
    return res

def save_object(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def predict_bmi(face_array, model_bmi = None, model_age = None):
    """ predcit bmi from face array or face array path (pickle file)
    Args: face_array (dict or path string)
    Return: BMI in float
    """
    if type(face_array) == str:
        with open(face_array,'rb') as f:
            face_array = pickle.load(f)
    
    df_test = pd.DataFrame(face_array).T
    res = pd.DataFrame({'name': df_test.index.tolist()})
    if model_bmi != None:
    	bmi = model_bmi.predict(df_test.values)
    	res['bmi'] = bmi
    if model_age != None:
    	age = model_age.predict(df_test.values)
    	res['age'] = age

    return res


@click.command()
@click.option('--image_path', default='./test/0', help='image path for prediction')
@click.option('--model_bmi', default='./saved_model/lasso_model_bmi.pkl', help='model pickle file(BMI)')
@click.option('--model_age', default='./saved_model/lasso_model_age.pkl', help='model pickle file(AGE)')

def main(image_path, model_bmi, model_age):
	
	print('input path: [%s]'%(image_path))

	face_array = extract_feature(input = image_path)
	res = predict_bmi(face_array, 
		load_object(Path(model_bmi)), 
		load_object(Path(model_age)))

	print(res)

if __name__ == '__main__':
	main()