import re
import os
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression,LogisticRegressionCV
from sklearn.svm import SVC, SVR
from sklearn.metrics.regression import mean_squared_error, mean_absolute_error
from sklearn.metrics.classification import f1_score
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer
from tqdm import tqdm

import os
import numpy as np
import pandas as pd
import pickle
from keras.engine import Model
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import tensorflow as tf

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_object(filename):
	with open(filename, 'rb') as f:
		res = pickle.load(f)
	return res

def save_object(obj, filename):
	with open(filename, 'wb') as f:
		pickle.dump(obj, f)

def print_ln(x, length = 40):
	left_len = (length - len(x) - 2) // 2
	right_len = length - left_len - len(x) - 2
	print('='*left_len + ' '+ str(x) + ' '+ '='*right_len)

def process_image(img,model):
	if model == 'vgg16':
		version = 1
	else:
		version = 2
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = utils.preprocess_input(x, version=version) # or version=2
	return x

def _encode_label(x, transform_label = True):
	if transform_label:
		return np.log1p(x)
	else:
		return x

def _decode_label(x, transform_label = True):
	if transform_label:
		return np.expm1(x)
	else:
		return x

def _convert_height(x):
	""" convert string height to inches(int)
	Args:
		@x: str
	Return:
		height in inches (int)
	"""
	try:
		x = re.sub('[^0-9]','',x)
		x = int(x[0]) * 12 + int(x[1:])
	except:
		x = None
	return x

def _combine_face_feature(face_array, data_path, label):

	face_array = pd.DataFrame(face_array).T
	data = pd.read_csv(data_path).set_index('index')
	data = data[[label]].join(face_array, how = 'inner')
	
	X = data.iloc[:,1:].values
	y = data.iloc[:,0].values
	
	return X, y

class tasks():

	def __init__(self, config):

		self.config = config
		self.config['train_path'] = './train.csv'
		self.config['valid_path'] = './valid.csv'
		self.config['face_array_path'] = './features/face_{}_{}.pkl'.format(
			config['model'], 
			config['layer']) 

	def extract_face_feature(self, input_face, save = False):
		""" extract face array from VGG-Face model
		Args:
			@input(str): image path or directory
			@model(str): {'vgg16','resnet50'}
			@layer(str): default to 'fc6'
			@save(bool): whether to save array to file
		Return:
			face array(dict)
		"""
		model = self.config['model']
		layer = self.config['layer']

		# build model to specify the layer of feature extraction
		vggface = VGGFace(model = model, input_shape=(224, 224, 3))
		vggface = Model(vggface.input, vggface.get_layer(layer).output)
		
		# extract face feature
		face_array = {}
		
		# for single image
		if os.path.isfile(input_face):
			img = image.load_img(input_face, target_size=(224, 224))
			res = vggface.predict(process_image(img, model))[0,:].reshape(-1)
			face_array[input_face.split('/')[-1]] = res
		
		# for image directory
		if os.path.isdir(input_face):
			for i in tqdm(os.listdir(input_face)):
				img = image.load_img('%s/%s'%(input_face,i), target_size=(224, 224))
				res = vggface.predict(process_image(img, model))[0,:].reshape(-1)
				face_array[i] = res
				
		if save:
			save_object(face_array, self.config['face_array_path'])
			
		return face_array

	def preprocess_split_meta_data(self, input_meta, split_ratio = 0.8):

		data = pd.read_csv(input_meta)

		data['height'] = data.height.map(_convert_height)
		data['bmi'] = data.weight / data.height / data.height * 703
		data['index'] = data.bookid.map(lambda i: str(i)+'.jpg')
		data['gender'] = data.sex.map(lambda i: 1 if i == 'Male' else 0)
		
		# filter unreasonable value
		data = data.loc[(data.weight > 0) & (data.height <80),:]

		# split train/valid
		in_train = np.random.uniform(size = len(data)) <= split_ratio
		train = data.loc[in_train,:]
		valid = data.loc[~in_train,:]
		train.to_csv('./train.csv', index = False)
		valid.to_csv('./valid.csv', index = False)
		print('> saved train set %s to [./train.csv]'%(str(train.shape)))
		print('> saved valid set %s to [./valid.csv]'%(str(valid.shape)))

		data.to_csv('./full_recoded.csv', index = False)

	def combine_face_feature(self, label):

		face_array = load_object(self.config['face_array_path'])
		train = _combine_face_feature(face_array, self.config['train_path'], label)
		valid = _combine_face_feature(face_array, self.config['valid_path'], label)
		print('> train set dimension: ' + str(train[0].shape))
		print('> valid set dimension: ' + str(valid[0].shape))
		
		return train, valid

	def train(self, model, train, valid, transform_label = True, type = 'reg'):

		score_auc = make_scorer(roc_auc_score)
		score_cor = make_scorer(lambda x,y:pearsonr(x,y)[0])
		x_train, y_train = train
		x_valid, y_valid = valid
		_ = model.fit(x_train, _encode_label(y_train, transform_label))

		if type == 'reg':
			y_pred_train = _decode_label(model.predict(x_train), transform_label)
			print('> train rmse: %5.3f'%(np.sqrt(mean_squared_error(y_pred_train, y_train))))
			print('> train mae: %5.3f'%(mean_absolute_error(y_pred_train, y_train)))
			print('> train corr: %5.3f with p-value: %5.3f'%(pearsonr(y_pred_train, y_train)))
			y_pred_valid = _decode_label(model.predict(x_valid), transform_label)
			print('> valid rmse: %5.3f'%(np.sqrt(mean_squared_error(y_pred_valid, y_valid))))
			print('> valid mae: %5.3f'%(mean_absolute_error(y_pred_valid, y_valid)))
			print('> valid corr: %5.3f with p-value: %5.3f'%(pearsonr(y_pred_valid, y_valid)))

		if type == 'bin':
			y_pred_train = model.predict_proba(x_train)[:,1]
			y_pred_valid = model.predict_proba(x_valid)[:,1]
			print('> train auc: %5.3f'%(roc_auc_score(y_train, y_pred_train)))
			print('> valid auc: %5.3f'%(roc_auc_score(y_valid, y_pred_valid)))

		return model

	def predict(self, models, face_array, transform_label = True):

		face_test = pd.DataFrame(face_array).T
		#age_model, gender_model, bmi_model = models
		nameid = face_test.index
		res = pd.DataFrame({'index':nameid})
		for k, model in models.items():
			if k.split('_')[0] == 'gender':
				res[k] = model.predict_proba(face_test)[:,1]
			else:
				res[k] = _decode_label(model.predict(face_test))

		return res

