from keras.applications.resnet_v2 import ResNet50V2, ResNet101V2
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import EfficientNetB0
from keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras import Sequential, Model, layers, models
from tensorflow.keras.layers import Dense, Dropout, Input, MaxPool2D, Conv2D, Flatten, LeakyReLU, BatchNormalization, \
    Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import Constant

def efficientNetB0(cfg, metrics, gpus, output_bias=None):
    '''
              Defines a model based on a pretrained ResNet101V2 for multiclass X-ray classification.
              Note that batch size per GPU should be >= 12 to prevent NaN in batch normalization.
              :param model_config: A dictionary of parameters associated with the model architecture
              :param input_shape: The shape of the model input
              :param metrics: Metrics to track model's performance
              :return: a Keras Model object with the architecture defined in this method
              '''

    # Set hyperparameters
    print('MODELAGEM')
    nodes_dense0 = cfg['P_CNN']['NODES_DENSE0']
    lr = cfg['P_CNN']['LR']
    dropout = cfg['P_CNN']['DROPOUT']
    l2_lambda = cfg['P_CNN']['L2_LAMBDA']
    input_shape = cfg['DATA']['IMG_DIM'] + [3]
    if cfg['P_CNN']['OPTIMIZER'] == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif cfg['P_CNN']['OPTIMIZER'] == 'sgd':
        optimizer = SGD(learning_rate=lr)
    else:
        optimizer = Adam(learning_rate=lr)  # For now, Adam is default option

    # Set output bias
    if output_bias is not None:
        output_bias = Constant(output_bias)

        # Set output bias
    if output_bias is not None:
        output_bias = Constant(output_bias)

        # Start with pretrained ResNet50V2
    '''Include_top permite selecionar se você quer as camadas densas finais ou não'''
    '''as camadas convolucionais funcionam como extratores de características. 
    Eles identificam uma série de padrões na imagem, e cada camada pode identificar padrões mais elaborados vendo padrões de padrões.
    as camadas densas são capazes de interpretar os padrões encontrados para classificar: esta imagem contém gatos, cães, carros, etc. 
    Por isso usei false - quero somente reaproveitar a extração de caracteríticas'''
    '''https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b'''
    X_input = Input(input_shape, name='input_img')
    res_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)

    model = models.Sequential()
    model.add(res_model)

    '''Camada Densa'''
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))
    if gpus >= 2:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics, )
    model.summary()

    return model


def inceptionv3(cfg, metrics, gpus, output_bias=None):
    '''
          Defines a model based on a pretrained ResNet101V2 for multiclass X-ray classification.
          Note that batch size per GPU should be >= 12 to prevent NaN in batch normalization.
          :param model_config: A dictionary of parameters associated with the model architecture
          :param input_shape: The shape of the model input
          :param metrics: Metrics to track model's performance
          :return: a Keras Model object with the architecture defined in this method
          '''

    # Set hyperparameters
    print('MODELAGEM')
    nodes_dense0 = cfg['P_CNN']['NODES_DENSE0']
    lr = cfg['P_CNN']['LR']
    dropout = cfg['P_CNN']['DROPOUT']
    l2_lambda = cfg['P_CNN']['L2_LAMBDA']
    input_shape = cfg['DATA']['IMG_DIM'] + [3]
    if cfg['P_CNN']['OPTIMIZER'] == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif cfg['P_CNN']['OPTIMIZER'] == 'sgd':
        optimizer = SGD(learning_rate=lr)
    else:
        optimizer = Adam(learning_rate=lr)  # For now, Adam is default option

    # Set output bias
    if output_bias is not None:
        output_bias = Constant(output_bias)

        # Set output bias
    if output_bias is not None:
        output_bias = Constant(output_bias)

        # Start with pretrained ResNet50V2
    '''Include_top permite selecionar se você quer as camadas densas finais ou não'''
    '''as camadas convolucionais funcionam como extratores de características. 
    Eles identificam uma série de padrões na imagem, e cada camada pode identificar padrões mais elaborados vendo padrões de padrões.
    as camadas densas são capazes de interpretar os padrões encontrados para classificar: esta imagem contém gatos, cães, carros, etc. 
    Por isso usei false - quero somente reaproveitar a extração de caracteríticas'''
    '''https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b'''
    X_input = Input(input_shape, name='input_img')
    res_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)

    model = models.Sequential()
    model.add(res_model)

    '''Camada Densa'''
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))
    if gpus >= 2:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics, )
    model.summary()

    return model


def resnet101v2(cfg, metrics, gpus, output_bias=None):
    '''
       Defines a model based on a pretrained ResNet101V2 for multiclass X-ray classification.
       Note that batch size per GPU should be >= 12 to prevent NaN in batch normalization.
       :param model_config: A dictionary of parameters associated with the model architecture
       :param input_shape: The shape of the model input
       :param metrics: Metrics to track model's performance
       :return: a Keras Model object with the architecture defined in this method
       '''

    # Set hyperparameters
    print('MODELAGEM')
    nodes_dense0 = cfg['P_CNN']['NODES_DENSE0']
    lr = cfg['P_CNN']['LR']
    dropout = cfg['P_CNN']['DROPOUT']
    l2_lambda = cfg['P_CNN']['L2_LAMBDA']
    input_shape = cfg['DATA']['IMG_DIM'] + [3]
    if cfg['P_CNN']['OPTIMIZER'] == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif cfg['P_CNN']['OPTIMIZER'] == 'sgd':
        optimizer = SGD(learning_rate=lr)
    else:
        optimizer = Adam(learning_rate=lr)  # For now, Adam is default option

    # Set output bias
    if output_bias is not None:
        output_bias = Constant(output_bias)

        # Set output bias
    if output_bias is not None:
        output_bias = Constant(output_bias)

        # Start with pretrained ResNet50V2
    '''Include_top permite selecionar se você quer as camadas densas finais ou não'''
    '''as camadas convolucionais funcionam como extratores de características. 
    Eles identificam uma série de padrões na imagem, e cada camada pode identificar padrões mais elaborados vendo padrões de padrões.
    as camadas densas são capazes de interpretar os padrões encontrados para classificar: esta imagem contém gatos, cães, carros, etc. 
    Por isso usei false - quero somente reaproveitar a extração de caracteríticas'''
    '''https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b'''
    X_input = Input(input_shape, name='input_img')
    res_model = ResNet101V2(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)

    model = models.Sequential()
    model.add(res_model)

    '''Camada Densa'''
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))
    if gpus >= 2:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics, )
    model.summary()

    return model


def resnet50v2(cfg, metrics, gpus, output_bias=None):
    """
    Defines a model based on a pretrained ResNet50V2 for multiclass X-ray classification.
    :param gpus:
    :param output_bias:
    :param cfg: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :return: a Keras Model object with the architecture defined in this method
    """

    # Set hyperparameters
    print('MODELAGEM')
    nodes_dense0 = cfg['P_CNN']['NODES_DENSE0']
    lr = cfg['P_CNN']['LR']
    dropout = cfg['P_CNN']['DROPOUT']
    l2_lambda = cfg['P_CNN']['L2_LAMBDA']
    input_shape = cfg['DATA']['IMG_DIM'] + [3]
    if cfg['P_CNN']['OPTIMIZER'] == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif cfg['P_CNN']['OPTIMIZER'] == 'sgd':
        optimizer = SGD(learning_rate=lr)
    else:
        optimizer = Adam(learning_rate=lr)  # For now, Adam is default option

    # Set output bias
    if output_bias is not None:
        output_bias = Constant(output_bias)

    # Start with pretrained ResNet50V2
    '''Include_top permite selecionar se você quer as camadas densas finais ou não'''
    '''as camadas convolucionais funcionam como extratores de características. 
    Eles identificam uma série de padrões na imagem, e cada camada pode identificar padrões mais elaborados vendo padrões de padrões.
    as camadas densas são capazes de interpretar os padrões encontrados para classificar: esta imagem contém gatos, cães, carros, etc. 
    Por isso usei false - quero somente reaproveitar a extração de caracteríticas'''
    '''https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b'''
    X_input = Input(input_shape, name='input_img')
    res_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)

    model = models.Sequential()
    model.add(res_model)

    '''Camada Densa'''
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))
    if gpus >= 2:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    model.summary()

    return model


def cnn_basic(cfg):
    lr = cfg['TEMPLATES_CNN_BASIC']['LR']
    dropout: object = cfg['TEMPLATES_CNN_BASIC']['DROPOUT']
    if cfg['TEMPLATES_CNN_BASIC']['DROPOUT'] == 'adam':
        optimizer = Adam(learning_rate=lr)
    else:
        optimizer = Adam(learning_rate=lr)
    image_shape = cfg['IMAGES']['DIM_X'], cfg['IMAGES']['DIM_Y'], cfg['IMAGES']['QUANT_CHANNELS']

    base_model = models.Sequential()
    base_model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=image_shape))
    base_model.add(layers.MaxPooling2D((2, 2)))
    base_model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    base_model.add(layers.MaxPooling2D((2, 2)))
    base_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    base_model.add(layers.MaxPooling2D((2, 2)))
    base_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    base_model.add(layers.MaxPooling2D((2, 2)))
    base_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    base_model.add(layers.MaxPooling2D((2, 2)))

    '''Camada Densa'''
    '''base_model.add(Flatten())'''
    base_model.add(GlobalAveragePooling2D())
    base_model.add(BatchNormalization())
    base_model.add(Dropout(dropout))
    base_model.add(Dense(128, activation='relu'))
    base_model.add(Dropout(dropout))
    base_model.add(Dense(1, activation='sigmoid'))

    base_model.summary()
    '''Para quando tiver mais de uma GPU na mÃ¡quina'''
    '''if gpus >= 2:
        model = 
        (model, gpus=gpus)'''

    base_model.compile(loss=cfg['TEMPLATES_CNN_BASIC']['LOSS'],
                       optimizer=optimizer,
                       metrics=cfg['TEMPLATES_CNN_BASIC']['METRICS'])
    base_model.summary()

    return base_model
