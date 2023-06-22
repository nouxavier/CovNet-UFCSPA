import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import cv2
import os

from sklearn.metrics import classification_report, auc, confusion_matrix, roc_curve
import tensorflow as tf
from datetime import date
from sklearn.utils import class_weight
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from models.models import cnn_basic, resnet50v2, resnet101v2, inceptionv3

import datetime

from processing.grafic_data import plot_balance_data, plot_accuracy, plot_loss, plot_confusion_matix, plot_roc
from processing.pre_processing import resize_files

print('COMECEI A PROCESSAR')
data_e_hora_atuais = datetime.datetime.now()
data_e_hora_em_texto = data_e_hora_atuais.strftime('%d/%m/%Y %H:%M')
print(data_e_hora_em_texto)

'''Verificando se GPU habilitada'''
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
num_gpu = int(len(tf.config.list_physical_devices('GPU')))
print('TIPO:' + str(type(num_gpu)))

'''Carregando as config do sistema'''
cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

epochs = cfg['TRAIN']['EPOCHS']
batch = cfg['TRAIN']['BATCH_SIZE']

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]
'''Fazendo o ppre-processamento das Images'''
images_dim_x = cfg['IMAGES']['DIM_X']
images_dim_y = cfg['IMAGES']['DIM_Y']
images_channel_z = cfg['IMAGES']['QUANT_CHANNELS']
images_dim = (images_dim_x, images_dim_y)

'''Get datas '''
data_local = pd.read_csv(cfg['PATHS']['METADATA_ALL_INFO'], sep=',')
x_data_local = data_local[['filename']].pop('filename')
y_data_local = data_local[['label']].pop('label')

'''Dados de testes'''
data_local = pd.read_csv(cfg['PATHS']['METADATA_ALL_INFO_HCPA_CLAHE'], sep=',')
x_test_clahe = data_local[['filename']].pop('filename')
y_test_clahe = data_local[['label']].pop('label')

data_local = pd.read_csv(cfg['PATHS']['METADATA_ALL_INFO_HCPA_RESIZE'], sep=',')
x_test_resize = data_local[['filename']].pop('filename')
y_test_resize = data_local[['label']].pop('label')

data_local = pd.read_csv(cfg['PATHS']['METADATA_ALL_INFO_HCPA_EQUAL'], sep=',')
x_test_equal = data_local[['filename']].pop('filename')
y_test_equal = data_local[['label']].pop('label')

data_local = pd.read_csv(cfg['PATHS']['METADATA_ALL_INFO_HCPA_ORIGINAL'], sep=',')
x_test_original = data_local[['filename']].pop('filename')
y_test_original = data_local[['label']].pop('label')

print('1')
X = []
print('1')
Y = []
print('1')
xy = resize_files(x_data_local, y_data_local, images_dim)
print(xy)
X = np.array(xy[0])
Y = np.array(xy[1])

X_clahe = []
Y_clahe = []
xy_clahe = resize_files(x_test_clahe, y_test_clahe, images_dim)
X_clahe = np.array(xy_clahe[0])
Y_clahe = np.array(xy_clahe[1])

X_resize = []
Y_resize = []
xy_resize = resize_files(x_test_resize, y_test_resize, images_dim)
X_resize = np.array(xy_resize[0])
Y_resize = np.array(xy_resize[1])

X_equal = []
Y_equal = []
xy_equal = resize_files(x_test_equal, y_test_equal, images_dim)
X_equal = np.array(xy_equal[0])
Y_equal = np.array(xy_equal[1])

X_origin =[]
Y_origin = []
xy_origin = resize_files(x_test_original, y_test_original, images_dim)
X_origin = np.array(xy_origin[0])
Y_origin = np.array(xy_origin[1])


kf = KFold(n_splits=6, shuffle=True)
skf = StratifiedKFold(n_splits=5, shuffle=True)
x_train = []
x_val = []
y_train = []
y_val = []

fold = 0
figura = 0
for train, validation in skf.split(X, Y):
    fold = fold + 1
    tf.keras.backend.clear_session()
    x_train, y_train, x_val, y_val = X[train], Y[train], X[validation], Y[validation]

    print('TAMANHO DOS ARRAY - DADOS TREINAMENTO')
    print('x_train')
    print('tipo do dados: ' + str(type(x_train)))
    print('shape: ' + str(x_train.shape))
    print('x_val')
    print('tipo do dados: ' + str(type(x_val)))
    print('shape: ' + str(x_val.shape))


    print('X_clahe, ')
    print('tipo do dados: ' + str(type(X_clahe)))
    print('shape: ' + str(X_clahe.shape))

    print('X_resize, ')
    print('tipo do dados: ' + str(type(X_resize)))
    print('shape: ' + str(X_resize.shape))

    print('X_equal, ')
    print('tipo do dados: ' + str(type(X_equal)))
    print('shape: ' + str(X_equal.shape))

    print('X_origin, ')
    print('tipo do dados: ' + str(type(X_origin)))
    print('shape: ' + str(X_origin.shape))

    x_train = np.array(x_train)
    x_val = np.array(x_val)

    X_clahe = np.array(X_clahe)
    X_resize = np.array(X_resize)
    X_equal = np.array(X_equal)
    X_origin = np.array(X_origin)

    y_train = np.array(y_train).astype('float32').reshape((-1, 1))
    y_val = np.array(y_val).astype('float32').reshape((-1, 1))

    Y_clahe = np.array(Y_clahe).astype('float32').reshape((-1, 1))
    Y_resize = np.array(Y_resize ).astype('float32').reshape((-1, 1))
    Y_equal = np.array(Y_equal ).astype('float32').reshape((-1, 1))
    Y_origin = np.array(Y_origin ).astype('float32').reshape((-1, 1))

    x_train = np.reshape(x_train, (len(x_train), images_dim_x, images_dim_y, images_channel_z)).astype('float32')
    x_val = np.reshape(x_val, (len(x_val), images_dim_x, images_dim_y, images_channel_z)).astype('float32')

    X_clahe = np.reshape(X_clahe, (len(X_clahe), images_dim_x, images_dim_y, images_channel_z)).astype('float32')
    X_resize = np.reshape(X_resize, (len(X_resize), images_dim_x, images_dim_y, images_channel_z)).astype('float32')
    X_equal = np.reshape(X_equal, (len(X_equal), images_dim_x, images_dim_y, images_channel_z)).astype('float32')
    X_origin = np.reshape(X_origin, (len(X_origin), images_dim_x, images_dim_y, images_channel_z)).astype('float32')

    plot_balance_data(y_train, 'Imagens de treinamento', 'balance_y_train' + '_fold_' + str(fold))
    plot_balance_data(y_val, 'Imagens de validacao', 'balance_y_validation' + '_fold_' + str(fold))

    plot_balance_data(Y_clahe, 'Imagens de teste', 'balance_y_tes_clahe' + '_fold_' + str(fold))
    plot_balance_data(Y_resize, 'Imagens de teste', 'balance_y_test_resize' + '_fold_' + str(fold))
    plot_balance_data(Y_equal, 'Imagens de teste', 'balance_y_test_equal' + '_fold_' + str(fold))
    plot_balance_data(Y_origin, 'Imagens de teste', 'balance_y_test_origin' + '_fold_' + str(fold))




    print('TAMANHO DOS ARRAY - DADOS TREINAMENTO')
    print('x_train')
    print('tipo do dados: ' + str(type(x_train)))
    print('shape: ' + str(x_train.shape))
    print('x_val')
    print('tipo do dados: ' + str(type(x_val)))
    print('shape: ' + str(x_val.shape))


    cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if cfg['MODELS']['TYPE'] == 'cnn_basic':
        base_model = cnn_basic(cfg)
    elif cfg['MODELS']['TYPE'] == 'resnet50v2':
        base_model = resnet50v2(cfg, METRICS, num_gpu)
    elif cfg['MODELS']['TYPE'] == 'resnet101v2':
        base_model = resnet101v2(cfg, METRICS, num_gpu)
    elif cfg['MODELS']['TYPE'] == 'inceptionv3':
        base_model = inceptionv3(cfg, METRICS, num_gpu)

    '''CREATE CALLBACKS'''
    '''https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback'''
    '''https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint'''
    '''ModelCheckpointO retorno de chamada ao usado em conjunto com o treinamento model.fit()
    para salvar um modelo ou pesos (em um arquivo de ponto de verifica) em algum intervalo, 
    para que o modelo ou pesos possam ser carregados posteriormente para continuar o treinamento
     a partir do estado salvo.'''
    checkpoint = tf.keras.callbacks.ModelCheckpoint(str(date.today()) + str(fold) + '.h5',
                                                    monitor='val_accuracy',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode='max')
    # sche = tf.keras.callbacks.LearningRateScheduler(scheduler)
    callbacks_list = [checkpoint]

    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train.reshape(-1))
    weights = {l: c for l, c in zip(np.unique(y_train), weights)}
    print(weights)

    print('INICIO DO TREINAMENTO')
    data_e_hora_atuais = datetime.datetime.now()
    data_e_hora_em_texto = data_e_hora_atuais.strftime('%d/%m/%Y %H:%M')
    print(data_e_hora_em_texto)

    history = base_model.fit(x_train,
                             y_train,
                             batch_size=batch,
                             epochs=epochs,
                             callbacks=callbacks_list,
                             class_weight={0: weights[0], 1: weights[1]},
                             validation_data=(x_val, y_val),
                             shuffle=True)

    print('DADOS TREINAMENTO')
    arquivoResultado = open('resultados_fold_' + str(fold) + '.txt', "w")
    arquivoResultado.write("Resumo da rede \n")
    arquivoResultado.write(str(base_model.summary()) + " \n")

    # list all data in history
    print(history.history.keys())
    arquivoResultado.write(str(history.history.keys()))
    print(history.history)
    arquivoResultado.write(str(history.history))

    plot_accuracy(history, fold)
    plot_loss(history, fold)



    print('PREDIcao CLAHE')
    Y_pred = base_model.predict(X_clahe)
    y_pred = np.round(Y_pred)
    print('depois da predi')

    print('Confusion Matrix - Y clahe')
    arquivoResultado.write('\n Confusion Matrix \n')
    CM = plot_confusion_matix(Y_clahe, y_pred)
    arquivoResultado.write(str(CM))
    print(CM)
    print('Classification Report')
    arquivoResultado.write('\n Classification Report \n')
    print(classification_report(Y_clahe, y_pred))
    arquivoResultado.write(str(classification_report(Y_clahe , y_pred)))

    print('y predito')
    arquivoResultado.write('\n y predito \n')
    print(y_pred)
    arquivoResultado.write(str(y_pred))
    print('y real')
    arquivoResultado.write('\n y real \n')
    print(Y_clahe )
    arquivoResultado.write(str(Y_clahe ))
    plot_roc(Y_clahe , y_pred, fold, 'Y_clahe')

    print('PREDIcao RESIZE')
    Y_pred = base_model.predict(X_resize)
    y_pred = np.round(Y_pred)
    print('depois da predicao')

    print('Confusion Matrix - Y RESIZE')
    arquivoResultado.write('\n Confusion Matrix \n')
    CM = plot_confusion_matix(Y_resize, y_pred)
    arquivoResultado.write(str(CM))
    print(CM)
    print('Classification Report')
    arquivoResultado.write('\n Classification Report \n')
    print(classification_report(Y_resize, y_pred))
    arquivoResultado.write(str(classification_report(Y_resize, y_pred)))

    print('y predito')
    arquivoResultado.write('\n y predito \n')
    print(y_pred)
    arquivoResultado.write(str(y_pred))
    print('y real')
    arquivoResultado.write('\n y real \n')
    print(Y_resize)
    arquivoResultado.write(str(Y_resize))
    plot_roc(Y_resize, y_pred, fold, 'Y_RESIZE')

    print('PREDICAO EQUAL')
    Y_pred = base_model.predict(X_equal)
    y_pred = np.round(Y_pred)
    print('depois da prediCAO')

    print('Confusion Matrix - Y EQUALIZATION')
    arquivoResultado.write('\n Confusion Matrix \n')
    CM = plot_confusion_matix(Y_equal, y_pred)
    arquivoResultado.write(str(CM))
    print(CM)
    print('Classification Report')
    arquivoResultado.write('\n Classification Report \n')
    print(classification_report(Y_equal, y_pred))
    arquivoResultado.write(str(classification_report(Y_equal, y_pred)))

    print('y predito')
    arquivoResultado.write('\n y predito \n')
    print(y_pred)
    arquivoResultado.write(str(y_pred))
    print('y real')
    arquivoResultado.write('\n y real \n')
    print(Y_equal)
    arquivoResultado.write(str(Y_equal))
    plot_roc(Y_equal, y_pred, fold, 'Y_EQUAL')

    print('PREDICAO ORIGINAL')
    Y_pred = base_model.predict(X_origin)
    y_pred = np.round(Y_pred)
    print('depois da prediCAO')

    print('Confusion Matrix - Y ORIGINAL')
    arquivoResultado.write('\n Confusion Matrix \n')
    CM = plot_confusion_matix(Y_origin, y_pred)
    arquivoResultado.write(str(CM))
    print(CM)
    print('Classification Report')
    arquivoResultado.write('\n Classification Report \n')
    print(classification_report(Y_origin, y_pred))
    arquivoResultado.write(str(classification_report(Y_origin, y_pred)))

    print('y predito')
    arquivoResultado.write('\n y predito \n')
    print(y_pred)
    arquivoResultado.write(str(y_pred))
    print('y real')
    arquivoResultado.write('\n y real \n')
    print(Y_origin)
    arquivoResultado.write(str(Y_origin))
    plot_roc(Y_origin, y_pred, fold, 'Y_ORIGIN')



    arquivoResultado.close()
    tf.keras.backend.clear_session()

    print('FIM DO TREINAMENTO')
    data_e_hora_atuais = datetime.datetime.now()
    data_e_hora_em_texto = data_e_hora_atuais.strftime('%d/%m/%Y %H:%M')
    print(data_e_hora_em_texto)
