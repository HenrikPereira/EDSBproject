import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier as RForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, f1_score, matthews_corrcoef, confusion_matrix, \
    roc_auc_score, roc_curve, auc, balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
ml_metric = 'f1'


def grids_skf_xgb(data_x, data_y, grid_params, pos_weight=1, scv_folds=5):
    """

    :param data_x:
    :param data_y:
    :param grid_params:
    :param pos_weight:
    :param scv_folds:
    :return:
    """
    m_xgb = XGBClassifier(n_jobs=-1, learning_rate=0.01, scale_pos_weight=pos_weight, verbosity=0)
    minmax = MinMaxScaler()

    grid_s = GridSearchCV(
        m_xgb, grid_params, n_jobs=-1,
        cv=StratifiedKFold(n_splits=scv_folds, shuffle=True),
        scoring=ml_metric, verbose=2, refit=True
    )

    train_feat_norm = minmax.fit_transform(data_x)

    grid_s.fit(train_feat_norm, data_y)
    print(grid_s.best_estimator_)
    return grid_s.best_estimator_


def grids_skf_lsvc(data_x, data_y, grid_params, weight_classes=None, scv_folds=5):
    """

    :param data_x:
    :param data_y:
    :param grid_params:
    :param weight_classes:
    :param scv_folds:
    :return:
    """
    if weight_classes is None:
        weight_classes = {0: 1, 1: 1}
    m_linear_svc = LinearSVC(
        dual=False,
        loss='squared_hinge',
        tol=0.0001,
        multi_class='ovr',
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=weight_classes,
        random_state=0
    )

    grid_s = GridSearchCV(
        m_linear_svc, grid_params, n_jobs=-1,
        cv=StratifiedKFold(n_splits=scv_folds, shuffle=True),
        scoring=ml_metric, refit=True, verbose=1
    )

    minmax = MinMaxScaler()
    train_feat_norm = minmax.fit_transform(data_x)

    grid_s.fit(train_feat_norm, data_y)
    print(grid_s.best_estimator_)
    return grid_s.best_estimator_


def grids_skf_svc(data_x, data_y, grid_params, weight_classes=None, scv_folds=5):
    """

    :param data_x:
    :param data_y:
    :param grid_params:
    :param weight_classes:
    :param scv_folds:
    :return:
    """
    if weight_classes is None:
        weight_classes = {0: 1, 1: 1}
    m_svc = SVC(
        probability=True,
        tol=0.001,
        cache_size=200,
        decision_function_shape='ovr',
        class_weight=weight_classes,
        random_state=0,
        max_iter=5000
    )

    grid_s = GridSearchCV(
        m_svc, grid_params, n_jobs=-1,
        cv=StratifiedKFold(n_splits=scv_folds, shuffle=True),
        scoring=ml_metric, refit=True, verbose=1
    )

    minmax = MinMaxScaler()
    train_feat_norm = minmax.fit_transform(data_x)

    grid_s.fit(train_feat_norm, data_y)
    print(grid_s.best_estimator_)
    return grid_s.best_estimator_


def grids_skf_rf(data_x, data_y, grid_params, weight_classes=None, scv_folds=5):
    """

    :param data_x:
    :param data_y:
    :param grid_params:
    :param weight_classes:
    :param scv_folds:
    :return:
    """
    if weight_classes is None:
        weight_classes = {0: 1, 1: 1}
    m_rf = RForest(
        criterion='gini',
        max_features='auto',
        class_weight=weight_classes,
        n_jobs=-1,
        random_state=0
    )

    grid_s = GridSearchCV(
        m_rf, grid_params, n_jobs=-1,
        cv=StratifiedKFold(n_splits=scv_folds, shuffle=True),
        scoring=ml_metric, refit=True, verbose=1
    )

    minmax = MinMaxScaler()
    train_feat_norm = minmax.fit_transform(data_x)

    grid_s.fit(train_feat_norm, data_y)
    print(grid_s.best_estimator_)
    return grid_s.best_estimator_


def grids_skf_lr(data_x, data_y, grid_params, weight_classes=None, scv_folds=5):
    """

    :param data_x:
    :param data_y:
    :param grid_params:
    :param weight_classes:
    :param scv_folds:
    :return:
    """
    if weight_classes is None:
        weight_classes = {0: 1, 1: 1}
    m_log = LogisticRegression(
        class_weight=weight_classes,
        random_state=0,
        multi_class='ovr',
        n_jobs=-1
    )

    grid_s = GridSearchCV(
        m_log, grid_params, n_jobs=-1,
        cv=StratifiedKFold(n_splits=scv_folds, shuffle=True),
        scoring=ml_metric, refit=True, verbose=1
    )

    minmax = MinMaxScaler()
    train_feat_norm = minmax.fit_transform(data_x)

    grid_s.fit(train_feat_norm, data_y)
    print(grid_s.best_estimator_)
    return grid_s.best_estimator_


def plot_elbow_kmeans(df, max_ks=7):
    """

    :param df: Dataset to evaluate
    :param max_ks: maximum number of k nearest neighbours to be tested
    :return:
    """
    ks = np.arange(1, max_ks + 1)
    inertias = []

    for k in ks:
        # Create a KMeans instance with k clusters: model
        t_model = KMeans(n_clusters=k, n_jobs=-1, n_init=30, random_state=123, )
        # Fit model to samples
        t_model.fit(df)
        # Append the inertia to the list of inertias
        inertias.append(t_model.inertia_)

    # Calculate the distance between line k1 to kmax and ki cluster
    xi, yi = 1, inertias[0]
    xf, yf = max_ks, inertias[-1]

    distances = []
    for i, v in enumerate(inertias):
        x0 = i + 1
        y0 = v
        numerator = abs((yf - yi) * x0 - (xf - xi) * y0 + xf * yi - yf * xi)
        denominator = np.sqrt((yf - yi) ** 2 + (xf - xi) ** 2)
        distances.append(numerator / denominator)

    temp_df = pd.concat(
        [pd.Series(ks, name='ks'), pd.Series(inertias, name='Inertia'), pd.Series(distances, name='Distance')],
        axis=1).set_index('ks')

    xmax = temp_df['Distance'].idxmax()
    ymax = temp_df['Distance'].max()
    dmax = temp_df['Inertia'].loc[xmax]

    # Plot ks (x-axis) vs inertias (y-axis) using plt.plot().
    plt.figure(figsize=(10, 5))
    ax = sb.lineplot(data=temp_df.reset_index(), x='ks', y='Inertia')
    plt.axvline(xmax, c='r', ls=':')
    ax2 = ax.twinx()
    sb.lineplot(data=temp_df.reset_index(), x='ks', y='Distance', color='g', ax=ax2)

    # Annotations
    ax2.annotate('Max Distance', xy=(xmax, ymax), xytext=(xmax + 1, ymax),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('Elbow at k:{}'.format(xmax), xy=(xmax, dmax), xytext=(xmax + 1, dmax + 1),
                arrowprops=dict(facecolor='black', shrink=0.05))

    ax.set_title('K-means Inertia Graph (Elbow method)', fontsize=16)
    ax.set_xlabel('Nr of Clusters', fontsize=12)
    plt.xticks(ks)
    plt.tight_layout()

    plt.show()

    print('The best number of clusters is', xmax)

    return xmax


def plot_nn_metrics(history):
    """

    :param history:
    :return:
    """
    p_metrics = ['loss', 'auc', 'precision', 'recall']
    plt.figure(figsize=(12, 10))
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
    """

    :param labels:
    :param predictions:
    :param p:
    :return:
    """
    cm = confusion_matrix(labels, predictions > p)
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
    """

    :param name:
    :param labels:
    :param y_score:
    :param lcolor:
    :param lwidth:
    :param lstyle:
    :return:
    """
    fp, tp, _ = roc_curve(labels, y_score)
    roc_auc = roc_auc_score(labels, y_score)

    plt.plot(
        100 * fp, 100 * tp, label='%s (auc=%.2f)' % (name, roc_auc), color=lcolor, linewidth=lwidth, linestyle=lstyle)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_pr_curve(name, true_labels, y_probability, y_hat, lcolor, lwidth=2, lstyle='-'):
    """

    :param name:
    :param true_labels:
    :param y_probability:
    :param y_hat:
    :param lcolor:
    :param lwidth:
    :param lstyle:
    :return:
    """
    precision, recall, _ = precision_recall_curve(true_labels, y_probability)
    f1, auc_ = f1_score(true_labels, y_hat), auc(recall, precision)

    plt.plot(recall, precision, label='%s (AUC:%.2f | f1:%.2f)' % (name, auc_, f1), color=lcolor, linewidth=lwidth,
             linestyle=lstyle)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def save_metrics(
        target_dataframe, model_name, true_y, predicted_y, probability_y, split_data, series, batch_size=np.nan):
    """
    Function to save metrics to a pandas dataframe.
    :type model_name: str
    :param target_dataframe: the name of the target dataframe
    :param model_name: the name of the model
    :param true_y: the y_true
    :param predicted_y: y_hat
    :param probability_y: probability or decision function for y
    :param split_data: partition of the data
    :param series: the series of the pipeline
    :param batch_size: the size of the mini batch used in Keras NN
    :return: pandas dataframe
    """
    tn, fp, fn, tp = confusion_matrix(true_y, predicted_y).ravel()
    precision, recall, _ = precision_recall_curve(true_y, probability_y)
    target_dataframe = target_dataframe.append(
        {
            'model': model_name,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'bal_acc': balanced_accuracy_score(true_y, predicted_y),
            'prec': precision_score(true_y, predicted_y),
            'recall': recall_score(true_y, predicted_y),
            'pr_auc': auc(recall, precision),
            'f1': f1_score(true_y, predicted_y),
            'mc_coef': matthews_corrcoef(true_y, predicted_y),
            'batch_s': batch_size,
            'data': split_data,
            'series': series
        },
        ignore_index=True
    )

    return target_dataframe


def drop_var_nonobj(dataframe):
    """

    :param dataframe:
    :return:
    """
    non_objs = dataframe.describe().columns.tolist()

    noneed = 0
    list_noneed = []

    for col in non_objs:
        if dataframe[col].min() == dataframe[col].max():
            dataframe.drop(columns=[col], inplace=True)
            non_objs.remove(col)
            noneed += 1
            list_noneed.append(col)

    if noneed == 1:
        print('\tThe {0} column was droped'.format(list_noneed))
    elif noneed > 1:
        print('\t{0} columns, {1} were droped'.format(noneed, list_noneed))
    else:
        print('\tNo columns were removed.')


def drop_var_obj(dataframe):
    """

    :param dataframe:
    :return:
    """
    objects = dataframe.describe(include='O').columns.tolist()

    noneed = 0
    list_noneed = []

    for col in objects:
        if len(dataframe[col].unique()) == 1:
            dataframe.drop(columns=[col], inplace=True)
            objects.remove(col)
            noneed += 1
            list_noneed.append(col)

    if noneed == 1:
        print('\tThe {0} column was droped.'.format(list_noneed))
    elif noneed > 1:
        print('\t{0} columns, {1} were droped.'.format(noneed, list_noneed))
    else:
        print('\tNo columns were removed.')
