from keras_preprocessing.image import ImageDataGenerator


def get_test_val_generator(X_train, y_train, X_test, y_test, X_val, y_val,
                           batch_size, seed, sample_size):
    raw_train_generator = ImageDataGenerator().flow(
        X_train, y_train,
        batch_size=sample_size,
        shuffle=False)

    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)

    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    test_generator = image_generator.flow(
        X_test,
        y=y_test,
        batch_size=batch_size,
        shuffle=False,
        seed=seed)

    # get validation generator
    val_generator = image_generator.flow(
        X_val,
        y=y_val,
        batch_size=batch_size,
        shuffle=False,
        seed=seed)

    return test_generator, val_generator