import itertools
import numpy as np
from matplotlib import pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          outfile='',
                          fmt='.2f',
                          verbose=False,
                          showtext=True,
                          fontsize="small"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        normalize: One of 'precision', 'recall' or None
    """
    if normalize=='precision':
        cm = cm.astype('float') / (cm.sum(axis=0)[np.newaxis, :] + 1e-10)
        print("Normalized confusion matrix")
    elif normalize=='recall':
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    if verbose:
        print(cm)

    if outfile != '':
        plt.ioff()
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # cm = cm.astype('int') if not normalize else cm
    thresh = cm.max() / 2.
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if showtext:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i+0.15, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)

    plt.tight_layout() # I cant believe repeating this produces a better plot....
    plt.tight_layout()
    plt.tight_layout()
    
    if outfile != '':
        plt.savefig(outfile, dpi=250)
        plt.close()
    else:
        plt.show()