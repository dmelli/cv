import pandas as pd
import numpy as np
from tasks import *

import click

@click.command()
@click.option('--layer', default='fc6', help='face model layer, defaults to "fc6"')
@click.option('--model', default='vgg16', help='face model, defaults to "vgg16"')
@click.option('--input_face', default='./face', help='face path for training, defaults to "./face"')
@click.option('--input_meta', default='./full.csv', help='meta data path, defaults to "./full.csv"')
@click.option('--test_face', default='./test', help='face path for testing, defaults to "./test"')
@click.option('--task', default='retrain', help='task option, [train, retrain, predict], defaults to "retrain"')
@click.option('--split_ratio', default=0.8, help='split ratio for train/retrain, defaults to 0.8')

def main(layer, model, input_face, input_meta, test_face, task, split_ratio):
	from tasks import tasks
	config = { 
	'layer': layer,
	'model': model
	}

	face_feature_path = './features/face_%s_%s.pkl'%(model, layer)
	models_path = './saved_model/models_%s_%s.pkl'%(model, layer)

	tasks = tasks(config)

	if task == 'train':

		# extract and save face feature
		face_array = tasks.extract_face_feature(input_face)
		if not os.path.exists('features'):
			os.mkdir('features')
		save_object(face_array, face_feature_path)
		print('saved feature to [%s]'%(face_feature_path))

		# split train and valid set
		tasks.preprocess_split_meta_data(input_meta, split_ratio)

	if task == 'retrain':

		# load extracted face feature
		try:
			print('load face feature from [%s]'%(face_feature_path))
			face_array = load_object(face_feature_path)
		except:
			print('no face feature found, extract new feature.')
			face_array = tasks.extract_face_feature(input_face) 
			if not os.path.exists('features'):
				os.mkdir('features')
			save_object(face_array, face_feature_path)
			print('saved feature to [%s]'%(face_feature_path))

		# make sure train/valid exists
		if all([i in os.listdir('./') for i in ['train.csv', 'valid.csv']]):
			pass
		else:
			tasks.preprocess_split_meta_data(input_meta, split_ratio)

	if task in ['train','retrain']:

		# train/retrain model
		
		models_all = {}
		
		print_ln('BMI')
		label = 'bmi'
		models = {}
		train, valid = tasks.combine_face_feature(label)
		models['%s_svm'%(label)] = SVR(kernel='rbf', C=0.1, gamma = 'scale')
		models['%s_lasso'%(label)] = LassoCV(max_iter = 5000, alphas = [0.3,0.5,0.9], cv = 5)
		for k, model in models.items():
			print_ln('train %s'%(k))
			models[k] = tasks.train(model, train, valid)
		models_all.update(models)

		print_ln('AGE')
		label = 'age'
		models = {}
		train, valid = tasks.combine_face_feature(label)
		models['%s_svm'%(label)] = SVR(kernel='rbf', C=0.1, gamma = 'scale')
		models['%s_lasso'%(label)] = LassoCV(max_iter = 5000, alphas = [0.3,0.5,0.9], cv = 5)
		for k, model in models.items():
			print_ln('train %s'%(k))
			models[k] = tasks.train(model, train, valid)
		models_all.update(models)

		print_ln('GENDER')
		label = 'gender'
		models = {}
		train, valid = tasks.combine_face_feature(label)
		models['%s_svm'%(label)] = SVC(kernel='rbf', C=1.0, gamma = 'scale', probability = True)
		models['%s_lr'%(label)] = LogisticRegressionCV(max_iter = 5000, cv = 5)
		for k, model in models.items():
			print_ln('train %s'%(k))
			models[k] = tasks.train(model, train, valid, transform_label = False, type = 'bin')
		models_all.update(models)

		save_object(models_all, models_path)

	if task == 'predict':

		try:
			models = load_object(models_path)
		except:
			print('no models found')
			return None

		face_array_test = tasks.extract_face_feature(test_face)
		res = tasks.predict(models, face_array_test)
		print(res)


if __name__ == '__main__':
	main()
