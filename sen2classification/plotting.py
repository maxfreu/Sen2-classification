import os
import itertools
import numpy as np
from matplotlib import pyplot as plt


def plot_confusion_matrix(cm: np.ndarray,
                          classes: list,
                          normalize: str = "",
                          title: str = 'Confusion matrix',
                          cmap = plt.cm.Blues,
                          outfile: str = '',
                          fmt: str = '.2f',
                          verbose: bool = False,
                          showtext: bool = True,
                          fontsize: int or str = "small"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        cm: Confusion matrix
        classes: List of class names
        normalize: One of 'precision', 'recall' or None
        title: Title of the plot
        cmap: Colormap
        outfile: Path to save the plot
        fmt: Format of the numbers in the plot, e.g. '.2f'
        verbose: Print the confusion matrix
        showtext: Show the values in the plot
        fontsize: Font size of the text
    """
    if normalize == 'precision':
        cm = cm.astype('float') / (cm.sum(axis=0)[np.newaxis, :] + 1e-10)
        if verbose:
            print("Normalized confusion matrix by precision")
    elif normalize == 'recall':
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        if verbose:
            print("Normalized confusion matrix by recall")
    else:
        if verbose:
            print('Confusion matrix, without normalization')

    if verbose:
        print(cm)

    if outfile != '':
        plt.ioff()

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot the confusion matrix
    cax = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(cax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.0
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    if showtext:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=fontsize)

    fig.tight_layout()

    if outfile != '':
        plt.savefig(outfile, dpi=250)
        plt.close(fig)
    else:
        plt.show()


def plot_confusion_matrices(outpath, cm, classes, dataset_name, qualifier):
    os.makedirs(outpath, exist_ok=True)

    np.savetxt(os.path.join(outpath, f"cm_{dataset_name}_{qualifier}.csv"), cm, delimiter=",")

    plot_confusion_matrix(cm, classes=classes, fmt=".0f",
                          outfile=os.path.join(outpath,
                                               f"confmat_{dataset_name}_{qualifier}_unnormalized.png"),
                          fontsize=4)

    plot_confusion_matrix(cm, classes=classes, fmt=".2f", normalize="precision", title="Precision",
                          outfile=os.path.join(outpath,
                                               f"confmat_{dataset_name}_{qualifier}_precision.png"),
                          fontsize="xx-small")

    plot_confusion_matrix(cm, classes=classes, fmt=".2f", normalize="recall", title="Recall",
                          outfile=os.path.join(outpath,
                                               f"confmat_{dataset_name}_{qualifier}_recall.png"),
                          fontsize="xx-small")
