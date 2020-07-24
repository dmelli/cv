from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import keras.backend as K

def multi_task_model(hidden_dim):
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    for layer in vgg_model.layers:
        layer.trainable = False

    last_layer = vgg_model.get_layer('pool5').output
    flatten = Flatten(name='flatten')(last_layer)

    x = Dense(hidden_dim, activation='relu', name='bmi_fc1')(flatten)
    x = Dense(hidden_dim, activation='relu', name='bmi_fc2')(x)
    out_bmi = Dense(1, activation='linear', name='bmi')(x)

    x = Dense(hidden_dim, activation='relu', name='age_fc1')(flatten)
    x = Dense(hidden_dim, activation='relu', name='age_fc2')(x)
    out_age = Dense(1, activation = 'linear', name = 'age')(x)

    x = Dense(hidden_dim, activation='relu', name='sex_fc1')(flatten)
    x = Dense(hidden_dim, activation='relu', name='sex_fc2')(x)
    out_sex = Dense(1, activation = 'sigmoid', name = 'sex')(x)

    custom_vgg_model = Model(vgg_model.input, [out_bmi, out_age, out_sex])
    custom_vgg_model.compile('adam', 
                         {'bmi':'mean_squared_error','age':'mean_squared_error','sex':'binary_crossentropy'},
                         {'bmi': 'mae','age':'mae','sex': 'accuracy'})
    
    return custom_vgg_model

def preprocess(train):
    all_images = os.listdir('./face/')
    train = train.loc[train['index'].isin(all_images),:]
    train.sex = train.sex.map(lambda i: 1 if i == 'Male' else 0).values
    train = train.rename(columns = {'index':'img'})
    train = train.reset_index(drop = True).reset_index()
    return train

def multilabel_flow_from_dataframe(datagen, df):
    for x, y in datagen:
        indices = y.astype(np.int).tolist()
        y_multi = [df.loc[df['index'].isin(indices),'bmi'].values, 
                   df.loc[df['index'].isin(indices),'age'].values,
                   df.loc[df['index'].isin(indices),'sex'].values]
        yield x, y_multi


#epochs = 20
#batch_size = 8
#hidden_dim = 256
#model_path = './saved_model/multi_task.h5'

import click

@click.command()
@click.option('--epochs', default=20, help='face model layer, defaults to "fc6"')
@click.option('--batch_size', default=8, help='face model, defaults to "vgg16"')
@click.option('--hidden_dim', default=256, help='face path for training, defaults to "./face"')
@click.option('--model_path', default='./saved_model/multi_task.h5', help='meta data path, defaults to "./full.csv"')
def main(epochs, batch_size, hidden_dim, model_path):

    ckp = ModelCheckpoint(
        model_path,
        monitor= 'val_loss',
        mode = 'min',
        verbose=1,
        save_best_only=True,
        save_weights_only = True)

    es = EarlyStopping(
        monitor= 'val_loss',
        patience=3,
        verbose=0,
        mode='min')

    callbacks = [ckp, es]

    try:
        model = load_model(model_path)
    except:
        model = multi_task_model(hidden_dim = hidden_dim)
    
    model.summary()

    train = pd.read_csv('./train.csv', usecols = ['index','bmi','age','sex'])
    train = preprocess(train)
    print('train data:')
    train.head()
    valid = pd.read_csv('./valid.csv', usecols = ['index','bmi','age','sex'])
    valid = preprocess(valid)
    print('valid data:')
    valid.head()

    train_datagen=ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range = 30,
        shear_range = 10,
        zoom_range = [1,1.25],
        horizontal_flip = True, 
        vertical_flip = True) 

    valid_datagen=ImageDataGenerator(
        preprocessing_function=preprocess_input) 

    train_generator=train_datagen.flow_from_dataframe(train, directory = './face', 
                                                     x_col = 'img', y_col = 'index', 
                                                     target_size=(224,224),
                                                     color_mode='rgb',
                                                     batch_size=batch_size,
                                                     class_mode='other',
                                                     shuffle=True)

    valid_generator=valid_datagen.flow_from_dataframe(valid, directory = './face', 
                                                     x_col = 'img', y_col = 'index',
                                                     target_size=(224,224),
                                                     color_mode='rgb',
                                                     batch_size=batch_size,
                                                     class_mode='other')
    train_generator = multilabel_flow_from_dataframe(train_generator, train)
    valid_generator = multilabel_flow_from_dataframe(valid_generator, valid)

    model.fit_generator(train_generator,
                        steps_per_epoch=train.shape[0] // batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=valid_generator,
                        validation_steps=valid.shape[0] // batch_size,
                        callbacks=callbacks,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        shuffle=True)
    print('test predictions:')
    test = valid.sample(10)

    # make predictions
    test_generator=valid_datagen.flow_from_dataframe(test, directory = './face', x_col = 'img', y_col = 'index',
                                                     target_size=(224,224),
                                                     color_mode='rgb',
                                                     batch_size=batch_size,
                                                     class_mode='other')
    steps = ((test.shape[0] + 0.1) // batch_size) + 1

    preds = custom_vgg_model.predict_generator(test_generator, steps = steps)
    preds_bmi, preds_age, preds_sex = preds

    test['preds_bmi'] = preds_bmi[:,0]
    test['preds_age'] = preds_age[:,0].astype(int)
    test['preds_sex'] = preds_sex[:,0].astype(int)

    print(test)

    test = pd.read_csv('./test.csv')
    # make predictions
    test_generator=valid_datagen.flow_from_dataframe(test, directory = './test', x_col = 'img', y_col = 'index',
                                                     target_size=(224,224),
                                                     color_mode='rgb',
                                                     batch_size=batch_size,
                                                     class_mode='other')
    #test_generator = multilabel_flow_from_dataframe(test_generator, test)
    steps = int(((test.shape[0] + 0.1) // batch_size) + 1)

    preds = custom_vgg_model.predict_generator(test_generator, steps = steps)
    preds_bmi, preds_age, preds_sex = preds

    test['preds_bmi'] = preds_bmi[:,0]
    test['preds_age'] = preds_age[:,0].astype(int)
    test['preds_sex'] = preds_sex[:,0].astype(int)

    print(test)

if __name__ == '__main__':
    main()