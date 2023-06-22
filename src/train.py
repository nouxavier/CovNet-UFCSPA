import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import cv2
import os
from sklearn.metrics import classification_report, auc, confusion_matrix, roc_curve
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import date
from sklearn.utils import class_weight
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from models.models import cnn_basic, resnet50v2

import datetime

'''checking if LNCC gpus are enabled'''
num_gpu = len(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ",num_gpu)


'''loading system settings'''
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

'''pre processing'''
images_dim_x = cfg['IMAGES']['DIM_X']
images_dim_y = cfg['IMAGES']['DIM_Y']
images_channel_z = cfg['IMAGES']['QUANT_CHANNELS']
images_dim = (images_dim_x, images_dim_y)

'''Get dats - put label '''
data_local = pd.read_csv(cfg['PATHS']['METADATA_ALL_INFO'], sep=',')
x_data_local = data_local[['filename']].pop('filename')
y_data_local = data_local[['label']].pop('label')

'''Test data'''
x_test = []
y_test = []

'''Train and validation data'''
X = []
Y = []

'''separate data in array for train and validation'''
for imagePath in x_data_local:
    try:
        image = cv2.imread(imagePath, 0)
        image = cv2.resize(image, images_dim, 1, interpolation=cv2.INTER_AREA)
        X.append(image)
        print('OK: ' + imagePath)
    except Exception as e:
        print('ERROR X: ' + imagePath)

for label in y_data_local:
    try:
        Y.append(label)
    except Exception as e:
        print('ERROR Y: ' + imagePath)
'''Normalization data'''
X = np.array(X)
Y = np.array(Y)

'''70/30 division'''
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

    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)

    y_train = np.array(y_train).astype('float32').reshape((-1, 1))
    y_val = np.array(y_val).astype('float32').reshape((-1, 1))
    y_test = np.array(y_test).astype('float32').reshape((-1, 1))

    x_train = np.reshape(x_train, (len(x_train), images_dim_x, images_dim_y, images_channel_z)).astype('float32')
    x_val = np.reshape(x_val, (len(x_val), images_dim_x, images_dim_y, images_channel_z)).astype('float32')
    x_test = np.reshape(x_test, (len(x_test), images_dim_x, images_dim_y, images_channel_z)).astype('float32')

    cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if cfg['PATHS']['METADATA_ALL_INFO'] == 'cnn_basic':
        base_model = cnn_basic(cfg)
    elif cfg['PATHS']['METADATA_ALL_INFO'] == 'resnet50v2':
        base_model = resnet50v2(cfg,images_dim, METRICS, num_gpu)

    base_model.summary()

    '''CREATE CALLBACKS'''
    '''https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback'''
    '''https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint'''
    '''ModelCheckpointO retorno de chamada Ã© usado em conjunto com o treinamento model.fit()
    para salvar um modelo ou pesos (em um arquivo de ponto de verificaÃ§Ã£o) em algum intervalo, 
    para que o modelo ou pesos possam ser carregados posteriormente para continuar o treinamento
     a partir do estado salvo.'''
    checkpoint = tf.keras.callbacks.ModelCheckpoint(str(date.today()) + '.h5',
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

    history = base_model.fit(x_train,
                             y_train,
                             batch_size=batch,
                             epochs=epochs,
                             callbacks=callbacks_list,
                             class_weight={0: weights[0], 1: weights[1]},
                             validation_data=(x_val, y_val),
                             shuffle=True)
    arquivoResultado = open('result_fold_' + str(fold) + '.txt', "w")
    arquivoResultado.write("Resume Network \n")
    arquivoResultado.write(str(base_model.summary()) + " \n")

    # list all data in history
    print(history.history.keys())
    arquivoResultado.write(str(history.history.keys()))
    print(history.history)
    arquivoResultado.write(str(history.history))

    print('PLOT Graphs')
    # summarize history for accuracy
    plt.figure(figura)
    figura += 1
    plt.figure(figura)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('model_accuracy_fold_' + str(fold), format='png')

    # summarize history for loss
    plt.figure(figura)
    figura += 1
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    # Results Prediction
    results = base_model.evaluate(x_test, y_test)
    Y_pred = base_model.predict(x_test)
    y_pred = np.round(Y_pred)

    print('Confusion Matrix')
    arquivoResultado.write('\n Confusion Matrix \n')
    CM = confusion_matrix(y_test, y_pred, labels=[0, 1])
    arquivoResultado.write(str(CM))
    print(CM)
    print('Classification Report')
    arquivoResultado.write('\n Classification Report \n')
    print(classification_report(y_test, y_pred))
    arquivoResultado.write(str(classification_report(y_test, y_pred)))

    print('y predict')
    arquivoResultado.write('\n y predito \n')
    print(y_pred)
    arquivoResultado.write(str(y_pred))
    print('y real')
    arquivoResultado.write('\n y real \n')
    print(y_test)
    arquivoResultado.write(str(y_test))

    'https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html'
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure(figura)
    figura += 1
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig('roc_curve_fold_' + str(fold), format='png')
    arquivoResultado.close()
    tf.keras.backend.clear_session()
