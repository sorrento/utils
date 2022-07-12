import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from IPython.core.display import display
from matplotlib import pyplot as plt
from sklearn import metrics as me
from sklearn.metrics import roc_curve, confusion_matrix


def standardize_function(X_train):
    from sklearn.preprocessing import StandardScaler
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=X_train.columns)
    return df_scaled


def Classifier(shape_, n_out, LR):
    """
entrena una wavenet
    :param LR: learning rate
    :param shape_:
    :param n_out: número de clases
    :return:
    """

    from tensorflow.keras import models, losses
    from tensorflow.python.keras import Input
    from tensorflow.python.keras.layers import Conv1D, Multiply, Add, Dense
    from tensorflow.python.keras.optimizer_v2.adam import Adam

    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2 ** i for i in range(n)]
        x = Conv1D(filters=filters,
                   kernel_size=1,
                   padding='same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters=filters,
                              kernel_size=kernel_size,
                              padding='same',
                              activation='tanh',
                              dilation_rate=dilation_rate)(x)
            sigm_out = Conv1D(filters=filters,
                              kernel_size=kernel_size,
                              padding='same',
                              activation='sigmoid',
                              dilation_rate=dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters=filters,
                       kernel_size=1,
                       padding='same')(x)
            res_x = Add()([res_x, x])
        return res_x

    inp = Input(shape=shape_)

    x = wave_block(inp, 16, 3, 12)
    x = wave_block(x, 32, 3, 8)
    x = wave_block(x, 64, 3, 4)
    x = wave_block(x, 128, 3, 1)

    out = Dense(n_out, activation='softmax', name='out')(x)

    model = models.Model(inputs=inp, outputs=out)

    opt = Adam(lr=LR)
    # opt = tfa.optimizers.SWA(opt)
    model.compile(loss=losses.CategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])
    return model


def Regressor(shape_, n_out, LR):
    """
entrena una wavenet
    :param LR: learning rate
    :param shape_:
    :param n_out: número de clases
    :return:
    """

    from tensorflow.keras import models, losses
    from tensorflow.python.keras import Input
    from tensorflow.python.keras.layers import Conv1D, Multiply, Add, Dense
    from tensorflow.python.keras.optimizer_v2.adam import Adam

    import tensorflow.keras.metrics as met


    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2 ** i for i in range(n)]
        x = Conv1D(filters=filters,
                   kernel_size=1,
                   padding='same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters=filters,
                              kernel_size=kernel_size,
                              padding='same',
                              activation='tanh',
                              dilation_rate=dilation_rate)(x)
            sigm_out = Conv1D(filters=filters,
                              kernel_size=kernel_size,
                              padding='same',
                              activation='sigmoid',
                              dilation_rate=dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters=filters,
                       kernel_size=1,
                       padding='same')(x)
            res_x = Add()([res_x, x])
        return res_x

    inp = Input(shape=shape_)

    x = wave_block(inp, 16, 3, 12)
    x = wave_block(x, 32, 3, 8)
    x = wave_block(x, 64, 3, 4)
    x = wave_block(x, 128, 3, 1)

    out = Dense(n_out, activation='linear', name='out')(x)

    model = models.Model(inputs=inp, outputs=out)

    opt = Adam(lr=LR)
    # opt = tfa.optimizers.SWA(opt)
    # model.compile(loss=losses.CategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.compile(loss=losses.MeanSquaredError(), optimizer=opt, metrics=[met.mean_squared_error])
    return model


def _best_f1(y_true, probs):
    import sklearn.metrics as me
    import numpy as np

    # y_true = test[s.target]
    # probs = preds
    precision, recall, thresholds = me.precision_recall_curve(y_true=y_true, probas_pred=probs)

    f1_scores = 2 * recall * precision / (recall + precision)
    th = thresholds[np.nanargmax(f1_scores, )]

    print('Best threshold: ', th)
    print('Best F1-Score: ', np.nanmax(f1_scores))

    return th


def plot_confusion_matrix(cm, classes, normalize=False, msg='', th=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    title = 'Confusion matrix ' + msg
    if th is not None:
        title = title + txt(th)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    # print(cm)
    fmt = '.2f' if normalize else 'd'
    pinta_cm(cm, classes, cmap, fmt, title)


def plot_confusion_matrix2(cm, ax, classes, normalize=False, msg='', th=None, cmap=plt.cm.Blues):
    """
    pensada para incorporarla a otros, plot, por eso el ax
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    # print(cm)

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    title = 'Confusion matrix ' + msg
    if th is not None:
        title = title + txt(th)

    ax.set_title(title)
    # ax.colorbar()
    tick_marks = np.arange(len(classes))
    # ax.set_xticks(tick_marks, classes, rotation=45)
    ax.set_xticks(tick_marks, classes)
    ax.set_yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    # ax.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    # plt.show()<


def pinta_cm2(y_test, y_pred, clases):
    cm1 = confusion_matrix(y_test, y_pred)
    cm = cm1
    pinta_cm(cm, clases)


def pinta_cm(cm, classes, cmap=plt.cm.Blues, fmt='d', title=''):
    import itertools

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_hist(pred, y_test, msg='', th=None, log=False):
    plt.figure(figsize=(14, 5))
    plt.hist(pred[y_test == 0]["p1"], bins=200, alpha=0.5, label="Background")
    plt.hist(pred[y_test == 0]["p0"], bins=200, alpha=0.5, label="Signal")
    if th is not None:
        plt.vlines(x=th,
                   ymin=0, ymax=100,
                   colors='red', linestyles='dashed', alpha=0.4,
                   label='F1 threshold')
        msg = msg + txt(th)
    if log:
        plt.yscale('log')
    plt.xlabel("probability")
    plt.grid()
    plt.title("Histogram of predicted probability: " + msg)
    plt.legend()


def plot_hist2(pred, y_test, ax, msg='', th=None, log=False):
    # plt.figure(figsize=(14, 5))
    ax.hist(pred[y_test == 0]["p1"], bins=200, alpha=0.5, label="Background")
    ax.hist(pred[y_test == 0]["p0"], bins=200, alpha=0.5, label="Signal")
    if th is not None:
        ax.vlines(x=th,
                  ymin=0, ymax=100,
                  colors='red', linestyles='dashed', alpha=0.4,
                  label='F1 threshold')
        msg = msg + txt(th)
    # if log:
    #     ax.yscale('log')
    ax.set_xlabel("probability")
    # ax.grid()
    ax.set_title("Histogram of predicted probability: " + msg)

    ax.legend()


def plot_roc(pred, y_test, th=None):
    cm = None
    fpr, tpr, _ = roc_curve(y_test, pred.p1)
    rocauc = me.roc_auc_score(y_true=y_test, y_score=pred.p1)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange', alpha=0.5,
             lw=3, label='ROC curve (area = %0.3f)' % rocauc)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    if th is not None:
        # cual es el eje x que le corresponde a un corte?
        y_pred = np.where(pred['p1'] > th, 1, 0)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        ejx = fp / (fp + tn)  # eje x false positive rate (qué parte de los negativos los predijimos mal)
        ejy = tp / (tp + fn)  # eje y (TRUE POSITIVE RATE) qué parte de los positivos predimos bien
        plt.plot(ejx, ejy, 'bx')
        print('** For threshold  ', txt(th), ',')
        print(str(100 * round(ejx, 2)) + '% of NEGATIVE events were predicted INCORRECTLY (False Positive Rate)')
        print(str(100 * round(ejy, 2)) + '% of POSITIVE events were predicted CORRECTLY (True Positive Rate)')
    plt.show()

    return cm


def plot_roc2(pred, pred2, y_test):
    fpr1, tpr1, _ = roc_curve(y_test, pred.p1)
    fpr2, tpr2, _ = roc_curve(y_test, pred2.p1)

    rocauc = me.roc_auc_score(y_true=y_test, y_score=pred.p1)
    rocauc2 = me.roc_auc_score(y_true=y_test, y_score=pred2.p1)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr1, tpr1, color='darkorange', alpha=0.5,
             lw=3, label='ROC model 1 (area = %0.3f)' % rocauc)
    plt.plot(fpr2, tpr2, color='darkred', alpha=0.5,
             lw=3, label='ROC model 2 (area = %0.3f)' % rocauc2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_roc3(pred, y_test, ax, th=None):
    cm = None
    fpr, tpr, _ = roc_curve(y_test, pred.p1)
    rocauc = me.roc_auc_score(y_true=y_test, y_score=pred.p1)
    print('** Model performance (ROC AUC): {}   (1 means perfect)'.format(round(rocauc, 3)))
    # ax.figure(figsize=(8, 5))
    ax.plot(fpr, tpr, color='darkorange', alpha=0.5,
            lw=3, label='ROC curve (area = %0.3f)' % rocauc)

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")

    if th is not None:
        # cual es el eje x que le corresponde a un corte?
        y_pred = np.where(pred['p1'] > th, 1, 0)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        ejx = fp / (fp + tn)  # eje x false positive rate (qué parte de los negativos los predijimos mal)
        ejy = tp / (tp + fn)  # eje y (TRUE POSITIVE RATE) qué parte de los positivos predimos bien
        ax.plot(ejx, ejy, 'bx')
        # print('** For threshold  ', txt(th), ',')
        print(str(100 * round(ejx, 2)) + '% of NEGATIVE events were predicted INCORRECTLY (False Positive Rate)')
        print(str(100 * round(ejy, 2)) + '% of POSITIVE events were predicted CORRECTLY (True Positive Rate)')
    # ax.show()

    return cm


def plot_var_imp_sns(df_impo):
    import seaborn as sns
    from matplotlib import pyplot as plt
    nfeat = len(df_impo)
    if nfeat < 30:
        l = 10
    else:
        l = 15
    plt.figure(figsize=(15, l))
    ax = sns.barplot(y=df_impo.index, x="importancia_perc", data=df_impo)
    k = 0
    for p in ax.patches:
        ax.annotate(df_impo['description'][k],
                    #                 xy=(p.get_width(), p.get_y()+p.get_height()/2),
                    xy=(0, p.get_y() + p.get_height() / 2),
                    xytext=(5, 0),
                    textcoords='offset points',
                    ha="left",
                    #                 color='darkgray',
                    size=13,
                    va="center",
                    font=dict(
                        family="Courier New",
                        size=14
                        # color="#ffffff"
                    )
                    )
        k = k + 1
    #     ax.annotate("%.2f" % p.get_width(),
    #                 xy=(p.get_width(), p.get_y()+p.get_height()/2),
    #                 xytext=(3, 0),
    #                 textcoords='offset points',
    #                 ha="left",
    #                 va="center")
    plt.title('Feature Importance')
    plt.xlabel('Relative Importance (%)')
    plt.ylabel(None)
    plt.show()


def plot_var_imp_sns2(df_impo, ax):
    import seaborn as sns
    nfeat = len(df_impo)
    if nfeat < 30:
        l = 10
    else:
        l = 15
    # ax.figure(figsize=(15, l))
    ax = sns.barplot(y=df_impo.index, x="importancia_perc", data=df_impo)
    k = 0
    for p in ax.patches:
        ax.annotate(df_impo['description'][k],
                    #                 xy=(p.get_width(), p.get_y()+p.get_height()/2),
                    xy=(0, p.get_y() + p.get_height() / 2),
                    xytext=(5, 0),
                    textcoords='offset points',
                    ha="left",
                    #                 color='darkgray',
                    size=13,
                    va="center",
                    font=dict(
                        family="Courier New",
                        size=14
                        # color="#ffffff"
                    )
                    )
        k = k + 1
    #     ax.annotate("%.2f" % p.get_width(),
    #                 xy=(p.get_width(), p.get_y()+p.get_height()/2),
    #                 xytext=(3, 0),
    #                 textcoords='offset points',
    #                 ha="left",
    #                 va="center")
    ax.set_title('Feature Importance')
    ax.set_xlabel('Relative Importance (%)')
    ax.set_ylabel(None)
    # ax.show()


def txt(th):
    return ' (th=' + str(round(th, 2)) + ')'


def train_model(X, y, lgbm_params={}):
    lbm = LightGBMCVModel(n_folds=5)
    params = {'metric':         {'rmse'}, "num_threads": 8,
              "num_iterations": 15000, 'objective': 'rmse'
              }
    params.update(lgbm_params)
    print(params)
    # TRAIN
    # features = [x for x in cols if x not in omitir]
    features = X.columns.to_list()
    lbm.train(X[features], y, params)
    return lbm


class ModelCV:
    def __init__(self, n_folds=5, random_state=42, is_cv=False):
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = None
        self.motor = None
        self.train_fold = None
        self.is_binary = None
        self.threshold = None  # para clasificación binaria
        self.is_cv = is_cv

    def train(self, X_data, Y_data, params, **kwargs):
        """
Train self.n_folds models and store them in self.models. Also, save predicts for every record
in data with the prediction of p model for its fold

        :param X_data:
        :param Y_data:
        :param params:
        :param kwargs:
        """
        from sklearn.model_selection import KFold

        self.is_binary = (len(Y_data.value_counts()) == 2)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        fold = np.zeros(X_data.shape[0])
        self.models = []
        for f, (train_ind, val_ind) in enumerate(kf.split(X_data, X_data)):
            np.random.seed(self.random_state)
            fold[val_ind] = f
            X_train, X_val = X_data.iloc[train_ind], X_data.iloc[val_ind]
            y_train, y_val = Y_data.iloc[train_ind], Y_data.iloc[val_ind]

            model = self.train_one(X_train, X_val, kwargs, params, y_train, y_val)

            self.models.append(model)
        self.train_fold = pd.Series(fold, index=X_data.index)

        # si es binario, calculamos el umbral (max f1 en este caso)
        if self.is_binary:
            if self.is_cv:
                probs = self.predict_cv(X_data)
            else:
                probs = self.predict_with_error(X_data).predict

            self.threshold = _best_f1(y_true=Y_data, probs=probs)

    def train_one(self, X_train, X_val, kwargs, params, y_train, y_val):
        return None

    def predict_cv(self, X_data, **kwargs):
        """
        Perform prediction as equally weighted ensemble of the models, and overwrite predictions for records
        in train set with the ones calculated during training.
        Note that, for this to work, X_data must be a DataFrame indexed the same way for train and predict
        and contain all the features the model has been trained with
        """
        # print('** prediciendo CV con EL modelo que no se entrenó con éL')
        y_pred = pd.Series(None, index=X_data.index, dtype="float")
        # First, perform predict for data that was used in training
        for i, m in enumerate(self.models):
            indices_fold = self.train_fold[self.train_fold == i].index
            # print('--- considerando fold {} con indices \n {}'.format(i, indices_fold))
            target_index = X_data.index.intersection(indices_fold)

            y_pred[target_index] = self.predict_one_model(X_data.loc[target_index, self.feature_name()], m, **kwargs)

        # print('-- cada registro tiene su prediccion {}'.format(y_pred))

        target_index = y_pred[y_pred.isnull()].index
        # print('-- los indices de los que son nulos son {}, is empty: {}'.format(target_index, target_index.empty))

        if not target_index.empty:
            data = X_data.loc[target_index, self.feature_name()]
            y_pred[target_index] = np.mean(
                [self.predict_one_model(data, m, **kwargs) for m in self.models],
                axis=0
            )
        return y_pred

    def predict_one_model(self, data, m, **kwargs):
        pass

    # todo explicar que puede tener dataleaking por el cv
    def predict_with_error(self, X_data, **kwargs):
        df = pd.DataFrame(None, index=X_data.index, columns=["predict", "std"], dtype="float")

        y_preds = np.array(
            [self.predict_one_model(X_data.loc[:, self.feature_name()], m, **kwargs) for m in self.models])

        df["predict"] = np.mean(y_preds, axis=0)
        df["std"] = np.std(y_preds, axis=0)

        return df

    def predict_df(self, X_data, is_cv=False):
        """
devuelve un df pandas con columna predict. En el caso de clasificación, tambien con p0 y p1
        :param X_data:
        :param is_cv:
        :return:
        """
        if is_cv:
            print('**Realizando predicciones con CV (holdout)')
            df_preds = pd.DataFrame({'predict': self.predict_cv(X_data)})

        else:
            df_preds = self.predict_with_error(X_data).drop(columns=['std'])

        if self.is_binary:
            # en caso binario, agregamos p0,p1
            df_preds = df_preds.rename(columns={'predict': 'p1'})
            df_preds['p0'] = 1 - df_preds.p1
            df_preds['predict'] = np.where(df_preds['p1'] > self.threshold, 1, 0)

        return df_preds

    def feature_name(self):
        pass

    def feature_importance(self):
        # Mean of feature importance for every internal model
        importances = self._get_importances()
        varimp = pd.DataFrame(data={"importance_value":    np.mean(importances, axis=0),
                                    "importance_variance": np.var(importances, axis=0), },
                              index=self.feature_name()).sort_values("importance_value", ascending=False)

        varimp['importancia'] = varimp.importance_value / varimp.importance_value.max()
        varimp['importancia_perc'] = varimp.importance_value / varimp.importance_value.sum()
        varimp.index.name = 'feature'

        return varimp

    def _get_importances(self):
        pass

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)


class LightGBMCVModel(ModelCV):
    def __init__(self, n_folds=5, random_state=42, is_cv=False):
        ModelCV.__init__(self, n_folds, random_state, is_cv)
        self.motor = 'MOTOR_LIGHTGBM'

    def train_one(self, X_train, X_val, kwargs, params, y_train, y_val):
        d_train = lgb.Dataset(X_train, label=y_train)
        d_test = lgb.Dataset(X_val, label=y_val, reference=d_train)
        watchlist = [d_train, d_test]
        model = lgb.train(params, d_train, valid_sets=watchlist, verbose_eval=-1,
                          early_stopping_rounds=100, **kwargs)

        return model

    def predict_one_model(self, data, m, **kwargs):
        return m.predict(data, **kwargs)

    def feature_name(self):
        return self.models[0].feature_name()

    def _get_importances(self):
        return [m.feature_importance(importance_type='gain') for m in self.models]


def on_test(Xs_test, lbm, target, ys_test):
    # agregamos métricas sobre el test
    l = list()
    for i in range(0, len(Xs_test)):
        l.append(compute_one(Xs_test, ys_test, lbm, i, target))
    te = resume(l)
    return te


def compute_one(Xs_test, ys_test, lbm, i, target):
    # print('i', i)
    # print('poto')
    # print( Xs_test)
    Xt = Xs_test[i]
    yt = ys_test[i]
    pre = lbm.predict_df(Xt)
    res = pd.concat([pre, yt], axis=1)
    err = np.sqrt(pow(res[target] - res.predict, 2).mean())
    print(err)
    return err


def get_prediction(lb, X_test):
    r1 = lb.predict_df(X_test)
    res = r1.drop(columns='p0').rename(
        columns={'p1': 'probability of signal', 'predict': 'prediction'}).sort_values('probability of signal',
                                                                                      ascending=False)
    display(res)
    return r1


def performance_and_variables(lb, pred, desc, y_test):
    df_impo = lb.feature_importance()[['importancia_perc']].join(desc)

    th = lb.threshold
    k = 1
    plt.figure(figsize=(25 * k, 12 * k))

    # Placing the plots in the plane
    plot1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    plot2 = plt.subplot2grid((3, 3), (0, 2), rowspan=1, colspan=1)  # confusion
    plot3 = plt.subplot2grid((3, 3), (1, 0), rowspan=2, colspan=1)
    plot4 = plt.subplot2grid((3, 3), (1, 1), rowspan=2, colspan=1)

    plot_hist2(pred, y_test, plot1, 'model as-it-is', th=th)
    cm = plot_roc3(pred, y_test, plot3, th)
    plot_confusion_matrix2(cm, plot2, classes=["background", "signal"], th=th, normalize=False)
    plot_var_imp_sns2(df_impo, plot4)

    # plt.tight_layout()
    plt.show()


def resume(lista):
    a, b = np.mean(lista), np.std(lista) / 2
    v1 = round(100 * b / a, 2)
    di = {'rmse': lista, 'me': a, 'std': b, 'perc': v1}
    return di
