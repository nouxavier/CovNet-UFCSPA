import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import cv2
import os

from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, auc, confusion_matrix, roc_curve
import tensorflow as tf
from datetime import date
from sklearn.utils import class_weight
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from models.models import cnn_basic, resnet50v2, resnet101v2, inceptionv3

import datetime

from src.processing.grafic_data import plot_balance_data, plot_accuracy, plot_loss, plot_confusion_matix, plot_roc

data_e_hora_atuais = datetime.datetime.now()
data_e_hora_em_texto = data_e_hora_atuais.strftime('%d/%m/%Y %H:%M')
print(data_e_hora_em_texto)

'''Verificando se GPU esÃ¡ habilitada'''
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
num_gpu = int(len(tf.config.list_physical_devices('GPU')))
print('TIPO:' + str(type(num_gpu)))

'''Carregando as configuraÃ§Ãµes do sistema'''
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
'''Fazendo o prÃ©-processamento das Images'''
images_dim_x = cfg['IMAGES']['DIM_X']
images_dim_y = cfg['IMAGES']['DIM_Y']
images_channel_z = cfg['IMAGES']['QUANT_CHANNELS']
images_dim = (images_dim_x, images_dim_y)

'''Get datas '''
data_local = pd.read_csv(cfg['PATHS']['METADATA_ALL_INFO'], sep=',')
x_data_local = data_local[['filename']].pop('filename')
y_data_local = data_local[['label']].pop('label')

'''Dados de testes'''
x_test = []
y_test = []


X = []
Y = []

for imagePath in x_data_local:
    try:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, images_dim, interpolation=cv2.INTER_AREA)
        X.append(image)
        print('OK: ' + imagePath)
    except Exception as e:
        print('ERRO: ' + imagePath)

for label in y_data_local:
    try:
        Y.append(label)
    except Exception as e:
        print('ERRO Y: ' + imagePath)

X = np.array(X)
Y = np.array(Y)

x_train_valid, x_test, y_train_valid, y_test = train_test_split(X, Y, test_size=0.3, random_state=42,
                                                                shuffle=True, stratify=Y)

kf = KFold(n_splits=6, shuffle=True)
skf = StratifiedKFold(n_splits=5, shuffle=True)
x_train = []
x_val = []
y_train = []
y_val = []

fold = 0
figura = 0
for train, validation in skf.split(x_train_valid, y_train_valid):
    fold = fold + 1
    tf.keras.backend.clear_session()
    x_train, y_train, x_val, y_val = x_train_valid[train], y_train_valid[train], x_train_valid[validation], \
                                     y_train_valid[validation]

    print('TAMANHO DOS ARRAY - DADOS TREINAMENTO')
    print('x_train')
    print('tipo do dados: ' + str(type(x_train)))
    print('shape: ' + str(x_train.shape))
    print('x_val')
    print('tipo do dados: ' + str(type(x_val)))
    print('shape: ' + str(x_val.shape))
    print('x_test')
    print('tipo do dados: ' + str(type(x_test)))
    print('shape: ' + str(x_test.shape))

    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)

    y_train = np.array(y_train).astype('float32').reshape((-1, 1))
    y_val = np.array(y_val).astype('float32').reshape((-1, 1))
    y_test = np.array(y_test).astype('float32').reshape((-1, 1))

    x_train = np.reshape(x_train, (len(x_train), images_dim_x, images_dim_y, images_channel_z)).astype('float32')
    x_val = np.reshape(x_val, (len(x_val), images_dim_x, images_dim_y, images_channel_z)).astype('float32')
    x_test = np.reshape(x_test, (len(x_test), images_dim_x, images_dim_y, images_channel_z)).astype('float32')

    plot_balance_data(y_train, 'Imagens de treinamento', 'balance_y_train' + '_fold_' + str(fold))
    plot_balance_data(y_val, 'Imagens de validação', 'balance_y_validation' + '_fold_' + str(fold))
    plot_balance_data(y_test, 'Imagens de teste', 'balance_y_test' + '_fold_' + str(fold))



    print('TAMANHO DOS ARRAY - DADOS TREINAMENTO')
    print('x_train')
    print('tipo do dados: ' + str(type(x_train)))
    print('shape: ' + str(x_train.shape))
    print('x_val')
    print('tipo do dados: ' + str(type(x_val)))
    print('shape: ' + str(x_val.shape))
    print('x_test')
    print('tipo do dados: ' + str(type(x_test)))
    print('shape: ' + str(x_test.shape))

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
    '''ModelCheckpointO retorno de chamada Ã© usado em conjunto com o treinamento model.fit()
    para salvar um modelo ou pesos (em um arquivo de ponto de verificaÃ§Ã£o) em algum intervalo, 
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


    # Resultados - prediÃ§Ã£o
    print('antes do evolute ')
    results = base_model.evaluate(x_test, y_test)
    print('depois do evolutee ')

    print('antes da predição')
    Y_pred = base_model.predict(x_test)
    y_pred = np.round(Y_pred)
    print('depois da predição')

    print('Confusion Matrix')
    arquivoResultado.write('\n Confusion Matrix \n')
    CM = plot_confusion_matix(y_test, y_pred)
    arquivoResultado.write(str(CM))
    print(CM)
    print('Classification Report')
    arquivoResultado.write('\n Classification Report \n')
    print(classification_report(y_test, y_pred))
    arquivoResultado.write(str(classification_report(y_test, y_pred)))

    print('y predito')
    arquivoResultado.write('\n y predito \n')
    print(y_pred)
    arquivoResultado.write(str(y_pred))
    print('y real')
    arquivoResultado.write('\n y real \n')
    print(y_test)
    arquivoResultado.write(str(y_test))

    'https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html'
    plot_roc(y_test, y_pred, fold)



    arquivoResultado.close()
    tf.keras.backend.clear_session()

    print('FIM DO TREINAMENTO')
    data_e_hora_atuais = datetime.datetime.now()
    data_e_hora_em_texto = data_e_hora_atuais.strftime('%d/%m/%Y %H:%M')
    print(data_e_hora_em_texto)
