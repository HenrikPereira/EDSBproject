import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sb
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_nn_metrics(history):
    p_metrics = ['loss', 'auc', 'precision', 'recall']
    fig = plt.figure(figsize=(12, 10))
    for n, metric in enumerate(p_metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.65, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()


def plot_cm(labels, predictions, p=0.5):
    cm = metrics.confusion_matrix(labels, predictions > p)
    sb.heatmap(cm,
               annot=True,
               cmap="Blues",
               fmt='d',
               square=True,
               annot_kws={'va': 'center',
                          'ha': 'center'},
               yticklabels=True)
    plt.title('Confusion matrix @ prob >= {:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Stayers Detected (True Negatives): ', cm[0][0])
    print('Stayers Missed (False Positives): ', cm[0][1])
    print('Leavers Missed (False Negatives): ', cm[1][0])
    print('Leavers Detected (True Positives): ', cm[1][1])
    # print('Total Leavers: ', np.sum(cm[1]))


def plot_roc(name, labels, y_score, lcolor, lwidth=2, lstyle='-'):
    fp, tp, _ = metrics.roc_curve(labels, y_score)
    auc = metrics.roc_auc_score(labels, y_score)

    plt.plot(100 * fp, 100 * tp, label='%s (auc=%.2f)' % (name, auc), color=lcolor, linewidth=lwidth, linestyle=lstyle)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_pr_curve(name, true_labels, y_probability, y_hat, lcolor, lwidth=2, lstyle='-'):
    precision, recall, _ = precision_recall_curve(true_labels, y_probability)
    f1, auc_ = f1_score(true_labels, y_hat), metrics.auc(recall, precision)

    plt.plot(recall, precision, label='%s (AUC:%.2f | f1:%.2f)' % (name, auc_, f1), color=lcolor, linewidth=lwidth, linestyle=lstyle)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def save_metrics(target_dataframe, model, labels, predictions, p=0.5, batch_size=np.nan):
    tp = metrics.confusion_matrix(labels, predictions > p)[1][1]
    fp = metrics.confusion_matrix(labels, predictions > p)[0][1]
    tn = metrics.confusion_matrix(labels, predictions > p)[0][0]
    fn = metrics.confusion_matrix(labels, predictions > p)[1][0]
    precision, recall, _ = precision_recall_curve(labels, predictions)
    target_dataframe = target_dataframe.append({'model': model,
                            'tp': tp,
                            'fp': fp,
                            'tn': tn,
                            'fn': fn,
                            'acc': metrics.accuracy_score(labels, predictions > p),
                            'prec': metrics.precision_score(labels, predictions > p),
                            'recall': metrics.recall_score(labels, predictions > p),
                            'pr_auc': metrics.auc(recall, precision),
                            'f1': metrics.f1_score(labels, predictions > p),
                            'mc_coef': metrics.matthews_corrcoef(labels, predictions > p),
                            'batch_s': batch_size}, ignore_index=True)

    return target_dataframe


def drop_var_nonobj(dataframe):
    non_objs = dataframe.describe().columns.tolist()

    noneed=0
    list_noneed = []

    for col in non_objs:
        if dataframe[col].min() == dataframe[col].max():
            dataframe.drop(columns=[col], inplace=True)
            non_objs.remove(col)
            noneed += 1
            list_noneed.append(col)

    if noneed == 1:
        print('The {0} column was droped'.format(list_noneed))
    elif noneed > 1:
        print('{0} columns, {1} were droped'.format(noneed, list_noneed))
    else:
        print('No columns were removed.')


def drop_var_obj(dataframe):
    objects = dataframe.describe(include='O').columns.tolist()

    noneed=0
    list_noneed = []

    for col in objects:
        if len(dataframe[col].unique()) == 1:
            dataframe.drop(columns=[col], inplace=True)
            objects.remove(col)
            noneed += 1
            list_noneed.append(col)

    if noneed == 1:
        print('The {0} column was droped.'.format(list_noneed))
    elif noneed > 1:
        print('{0} columns, {1} were droped.'.format(noneed, list_noneed))
    else:
        print('No columns were removed.')
