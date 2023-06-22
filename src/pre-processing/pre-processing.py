import csv
import os

import cv2
import numpy as np
import yaml
from matplotlib import pyplot as plt


def histogram(path_image):
    im1 = cv2.imread(path_image)
    img = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)  # converte P&B

    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


# histogram('C:/Mestrado/Imagens/hcpa/processed/hcpa_covid_I173.png')
def equalization(path_image, path_image_salve):
    # https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
    im1 = cv2.imread(path_image)
    img = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)  # converte P&B
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)  # For masked array, all operations are performed on non-masked elements.
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (
            cdf_m.max() - cdf_m.min())  # ver função de transferência em detalhes: Gonzalez & Woods (capítulo 3).
    cdf2 = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = img.copy()

    # Agora temos a tabela de consulta que nos dá as informações sobre qual é o valor do pixel de saída para
    # cada valor do pixel de entrada. Então, apenas aplicamos a transformação:

    img2 = cdf2[img]
    cv2.imwrite(path_image_salve, img2)


# equalizalization('C:/Mestrado/Imagens/hcpa/processed/sub-S03089_ses-E06747_run-1_bp-chest_vp-pa_dx.png')

def paths_equalizalization(input_path, output_path_final):
    cont = 0
    for paths, subpaths, files in os.walk(input_path):
        for name_image in files:
            explod_file = name_image.split('.')
            extension = explod_file[-1]
            if extension == 'png' or extension == 'jpg' or extension == 'jpeg':
                equalization(input_path + name_image, output_path_final + name_image)
                cont = cont + 1
                print(str(cont))


# paths_equalizalization('C:/Mestrado/Imagens/hcpa/png-cut/', 'C:/Mestrado/Imagens/hcpa/equalization/')

def clahe(input_path, output_path_final):
    cont = 0
    for paths, subpaths, files in os.walk(input_path):
        for name_image in files:
            explod_file = name_image.split('.')
            extension = explod_file[-1]
            if extension == 'png' or extension == 'jpg' or extension == 'jpeg':
                # https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
                im1 = cv2.imread(input_path + name_image)
                img = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)  # converte P&B
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                cl1 = clahe.apply(img)
                cv2.imwrite(output_path_final + name_image, cl1)
                cont = cont + 1
                print(str(cont))


# clahe('C:/Mestrado/Imagens/hcpa/png-cut/', 'C:/Mestrado/Imagens/hcpa/clahe/')


def reshape(path_image):
    X = []
    image = cv2.imread(path_image)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    print(image.shape)
    X.append(image)
    print(X)
    X = np.array(X)
    print(X)
    x_train = np.reshape(X, (len(X), 224, 224, 3)).astype('float32')
    print(x_train)


# reshape('C:/Mestrado/Imagens/sub-S04521_ses-E08982_run-1_bp-chest_vp-pa_cr.png')


def transform_img_1channel_to_3channel(path_image):
    try:
        image = cv2.imread(path_image)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        print(image.shape)
        cv2.imwrite('teste1.jpg', image)
        image = cv2.imread(path_image)
        image = cv2.resize(image, (224, 224), 1, interpolation=cv2.INTER_AREA)
        cv2.imwrite('teste2.jpg', image)
        print(image.shape)
        print('OK: ' + path_image)
    except Exception as e:
        print('ERRO: ' + path_image)

    img = cv2.imread(path_image)
    print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img2 = np.zeros_like(img)
    img2[:, :, 0] = gray
    img2[:, :, 1] = gray
    img2[:, :, 2] = gray
    cv2.imwrite('10524.jpg', img2)


# transform_img_1channel_to_3channel('C:/Mestrado/Imagens/sub-S04521_ses-E08982_run-1_bp-chest_vp-pa_cr.png')


def cut_images_hcpa():
    cfg = yaml.full_load(open("C:/Users/nouar/PycharmProjects/LNCC-UFCSPA-COVNET/src/config.yml", 'r'))
    path_image = cfg['PATHS']['LOCAL_IMAGES_HCPA']
    path_images = cv2.imread(path_image, 0)
    img_grey1 = path_images
    # Binarizando
    # T = mahotas.thresholding.rc(img_grey1)
    T = 255
    temp1 = img_grey1.copy()
    temp1[temp1 > T] = 255
    temp1[temp1 < T] = 0
    temp1 = cv2.bitwise_not(temp1)
    # Calculando os momentos da imagem
    M = cv2.moments(temp1)
    print(M)

    # Calculando o centroide
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # M['m00'] --> momento não normalizado: www.lapix.ufsc.br/ensino/reconhecimento-de-padroes/gerando-padroes-analise-de-sinais-e-imagens/

    lado = cv2.countNonZero(temp1[cy])

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.imshow(temp1[::4, ::4], cmap='gray')
    plt.title("Imagem Binarizada")

    plt.subplot(1, 3, 2)
    plt.imshow(img_grey1[::4, ::4], cmap='gray')
    plt.title("Resultado do Recorte a ser realizado")

    coor_1 = cy - lado / 2
    coor_2 = cy + lado / 2
    coor_3 = cx - lado / 2
    coor_4 = cx + lado / 2

    recorte = img_grey1[int(coor_1):int(coor_2), int(coor_3):int(coor_4)]
    plt.subplot(1, 3, 3)
    plt.imshow(recorte, cmap='gray')
    plt.title("imagem Recortada")


def process_clahe(image, output_path_clahe):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(image)
    cv2.imwrite(output_path_clahe, cl1)


def process_remove_text(local_image):
    print('Lendo:' + local_image)
    image = cv2.imread(local_image)
    mask = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)[1][:, :, 0].astype(np.uint8)
    image = image.astype(np.uint8)
    result = cv2.inpaint(image, mask, 10, cv2.INPAINT_NS).astype(np.float32)
    return result


def process_images(input_path, output_path_clahe, output_path_final):
    for paths, subpaths, files in os.walk(input_path):
        for name_image in files:
            explod_file = name_image.split('.')
            extension = explod_file[-1]
            if extension == 'png' or extension == 'jpg' or extension == 'jpeg':
                image = cv2.imread(input_path + name_image, 0)
                print(input_path + name_image)
                print('-----')
                process_clahe(image, output_path_clahe + name_image)
                print(output_path_clahe + name_image)
                print('-----')
                image_processed = process_remove_text(output_path_clahe + name_image)
                print(output_path_final + name_image)
                print('ffffffff')
                cv2.imwrite(output_path_final + name_image, image_processed)


def process_images_BIMCV(input_path, output_path_8bit):
    try:
        for paths, subpaths, files in os.walk(input_path):
            for file in files:
                explod_file = file.split('.')
                extension = explod_file[-1]
                if extension == 'png' or extension == 'jpg' or extension == 'jpeg':
                    try:
                        file_local = os.path.join(paths, file)
                        print(file)
                        print(file_local)
                        img = cv2.imread(file_local)
                        img = np.array(img, dtype=float)
                        img = (img - img.min()) / (img.max() - img.min()) * 255.0
                        img = img.astype(np.uint8)
                        cv2.imwrite(os.path.join(output_path_8bit, file), img)  # write png image
                    except Exception as e:
                        print("ERRO: " + file_local)
    except Exception as e:
        print(str(e))


def remove_error_8bit(input_path):
    try:
        for paths, subpaths, files in os.walk(input_path):
            for file in files:
                explod_file = file.split('.')
                extension = explod_file[-1]
                if extension == 'png' or extension == 'jpg' or extension == 'jpeg':
                    try:
                        if '8bit' in str(file):
                            file_local = input_path + file
                            if os.path.exists(file_local):
                                os.remove(file_local)
                                print(file_local)
                                print('----')
                    except Exception as e:
                        print("ERRO: " + file_local)
    except Exception as e:
        print(str(e))


def create_metadata(path, flag_covid, name_csv):
    f = open(name_csv, 'w')
    file_csv = csv.writer(f)
    data = ['filename, label, label_str, filename_ro']
    file_csv.writerow(data)
    for diretorio, subpastas, arquivos in os.walk(path):
        for arquivo in arquivos:
            print(os.path.join(diretorio, arquivo))
            explod_file = arquivo.split('.')
            extension = explod_file[-1]
            if extension == 'png' or extension == 'jpg' or extension == 'jpeg':
                if flag_covid is True:
                    data = [os.path.join(diretorio, arquivo), '1', 'COVID-19']
                    file_csv.writerow(data)
                else:
                    data = [os.path.join(diretorio, arquivo), '0', 'non-COVID-19']
                    file_csv.writerow(data)

