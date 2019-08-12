# for data i/o
import numpy as np
import pandas as pd
import re
import shutil
from pathlib import Path
# for model training
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# for image file
from PIL import ImageFile
from tensorflow.python.client import device_lib

TASK = 'predict' # [train, predict]
TRAIN_NEW = False

TRAIN_PATH = './gender/train'
VALID_PATH = './gender/valid'
IN_TRAIN_RATIO = 0.9
DROP_RATE = 0.5
NUM_EPOCHS = 10
HIDDEN_DENSE = [1024,128]
NUM_FIXED_LAYER = None
PATIENCE = 10


model_name = '{}_{}_{}'.format(
    NUM_FIXED_LAYER, 
    '_'.join(map(str, HIDDEN_DENSE)),
    str(DROP_RATE).replace('.',''))
model_path = './gender/saved_model/{}.h5'.format(model_name)

def create_train_valid(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents = True)

def move_image(bookids, path):
    for bookid in bookids:
        try:
            shutil.copy(Path('./face','{}.jpg'.format(bookid)), path)
        except:
            print('./face/{}.jpg missing'.format(bookid))
            continue

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def my_print(text):
    print('='*10 + ' ' +str(text)+ ' ' +'='*10)

if TASK == 'predict':
    my_print('predict')
    try:
        model = load_model(model_path)
        print('load model')
    except:
        print('no model found')
        exit()
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory('./test',
                                                     target_size=(224,224),
                                                     color_mode='rgb',
                                                     batch_size=1,
                                                     class_mode='categorical',
                                                     shuffle=False)

    filenames = test_generator.filenames
    nb_samples = len(filenames)
    predict = model.predict_generator(test_generator,steps = nb_samples)
    predict_class = np.argmax(predict,1)
    predict_score = predict[:,-1]
    res = pd.DataFrame({'face':filenames, 
        'score':predict_score,
        'prediction':predict_class})
    print(res)

    exit()

if TRAIN_NEW:
    data = pd.read_csv('./full.csv')
    #data['height'] = data.height.map(lambda i: re.sub('[^0-9]','',i)).map(lambda i: float(i[0])*12 + float(i[1:])/10)
    #data['bmi'] = data.weight / data.height.map(lambda i: i**2) * 703
    #data = data.loc[(data.weight >0) & (data.height > 0),:]
    data = data.loc[data.sex.isin(['Male','Female']),:]
    #data['label'] = data.bmi.map(lambda i: '1_bmi_over_30' if i > 30 else '0_bmi_below_30')
    data['label'] = data.sex.map(lambda i: 1 if i == 'Male' else 0)

    in_train = np.random.uniform(size = len(data)) < IN_TRAIN_RATIO

    # create folders
    create_train_valid(Path(TRAIN_PATH, '0'))
    create_train_valid(Path(TRAIN_PATH, '1'))
    create_train_valid(Path(VALID_PATH, '0'))
    create_train_valid(Path(VALID_PATH, '1'))

    # move image to folders
    print('1) moving train set - label 1')
    move_image(data.loc[(data.label == 1) & (in_train),'bookid'].values, Path(TRAIN_PATH,'1'))
    print('2) moving train set - label 0')
    move_image(data.loc[(data.label == 0) & (in_train),'bookid'].values, Path(TRAIN_PATH,'0'))
    print('3) moving valid set - label 1')
    move_image(data.loc[(data.label == 1) & (~in_train),'bookid'].values, Path(VALID_PATH,'1'))
    print('4) moving valid set - label 0')
    move_image(data.loc[(data.label == 0) & (~in_train),'bookid'].values, Path(VALID_PATH,'0'))

print('check if gpu is present...')


device_lib.list_local_devices()

print('start training...')

try:
    model = load_model(model_path)
    print('load model')
except:
    print('start a new model')
    base_model = ResNet50(weights = 'imagenet', include_top = False)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(DROP_RATE)(x)
    for h in HIDDEN_DENSE:
        x = Dense(h, activation='relu')(x)
        x = Dropout(DROP_RATE)(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(base_model.input, x)

    if NUM_FIXED_LAYER == None:
        num_base_model_layers = len(base_model.layers)
    else: 
        num_base_model_layers = NUM_FIXED_LAYER
    for layer in model.layers[:num_base_model_layers]:
        layer.trainable = False
    for layer in model.layers[num_base_model_layers:]:
        layer.trainable = True
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen=ImageDataGenerator(
	preprocessing_function=preprocess_input,
	rotation_range = 30,
	shear_range = 10,
	zoom_range = [1,1.25],
	horizontal_flip = True, 
	vertical_flip = True) 

ckp = ModelCheckpoint(model_path,
    save_best_only=True, 
    save_weights_only=False)  
es = EarlyStopping(patience = PATIENCE)
callbacks = [ckp, es]

train_generator=train_datagen.flow_from_directory(TRAIN_PATH,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

valid_generator=train_datagen.flow_from_directory(VALID_PATH,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

ImageFile.LOAD_TRUNCATED_IMAGES = True
res = get_available_gpus()
print(res)

if len(res) >0:

	step_size_train=train_generator.n//train_generator.batch_size

	model.fit_generator(generator=train_generator,
                        callbacks = callbacks,
	                   steps_per_epoch=step_size_train,
	                    validation_data=valid_generator,
	                   epochs=NUM_EPOCHS)
else:
	print('no GPU found')