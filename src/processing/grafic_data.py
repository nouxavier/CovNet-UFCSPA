from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, auc, roc_curve


def plot_balance_data(y, title, name_fig):
    labels = ['1_NORMAL', '2_COVID']
    no_cov = 0
    cov = 0

    print('IMBALANCE')
    for label in y:
        if label == 0:
            no_cov += 1
        if label == 1:
            cov += 1
    print(f'Number of Normal images = {no_cov}')
    print(f'Number of Covid images = {cov}')
    xe = [i for i, _ in enumerate(labels)]

    numbers = [no_cov, cov]
    plt.bar(xe, numbers, color='green')
    plt.xlabel("Labels")
    plt.ylabel("No. of images")
    plt.title(title)
    plt.xticks(xe, labels)
    plt.savefig(name_fig, format='png')


def plot_accuracy(history, fold):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('model_accuracy_fold_' + str(fold), format='png')


def plot_loss(history, fold):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('model_loss_fold_' + str(fold), format='png', )


def plot_confusion_matix(y_test, y_pred):
    CM = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return CM


def plot_roc(y_test, y_pred, fold, title):
    'https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html'
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig('roc_curve_fold_' + str(fold) + '_' + title, format='png')

