import cv2


def resize_files(x_data_local, y_data_local, images_dim):
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
    return X,Y