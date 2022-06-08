import pandas as pd
from IPython.core.display import display
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from u_base import save_df, inicia, tardado

CLUSTER_DBSCAN = 'DBscan'
CLUSTER_KMEANS = 'Kmeans'


class Clusterer:
    """
    Realiza un clustering a partir de un dataframe pandas
    """

    def __init__(self, df, col_name):
        """

        Parameters
        ----------
        df: Dataframe Pandas
            Dataset para el cual se calculará el clustering
        col_name: str
            Nombre que se le dará a la columna de clustering
        """

        self._df_labels = None
        self._algoritmo = None
        self._n_clusters = None
        self._col_name = col_name
        self._df = df
        self._X = None  # Dataset listo para clustering (imputado y escalado)

    def save(self, out_):
        """
        Guarda el resultado del clustering en un fichero
        """
        if self._df_labels is not None:
            filename = 'clustering_' + self._algoritmo + '_' + str(self._n_clusters) + '_clusters'
            save_df(self._df_labels, path=out_, name=filename, save_index=True)
        else:
            print('*** WARN *** no se puede guardar porque aún no se ha calculado')

    def histogram(self):
        """
        Pinta el histograma de número de elementos por cluster
        """
        self._df_labels[self._col_name].value_counts().plot(kind='bar')

    def get_labels(self):
        """

        Returns
        -------
        labels:
            Devuelve un dataframe indice - etiqueta del clustering

        """
        if self._df_labels is not None:
            display(self._df_labels)
            self.histogram()

            return self._df_labels
        else:
            print('** WARN No se han calculado aún los clusters')
            return None

    def _set_df_labels(self, labels_):
        # indices-etiqueta del cluster
        df = pd.DataFrame(labels_, columns=[self._col_name], index=self._df.index)
        df[self._col_name] = df[self._col_name].astype(object)
        self._df_labels = df

    def get_metricas(self):
        """
        Entrega y muestra la visualización de las métricas de calidad del clustering
        """
        print('*** WARN *** No implementado aún')
        pass  # todo poner métrica
        # print("Silhouette Coefficient: %0.3f"
        #       % metrics.silhouette_score(self.X, self.df_labels[self.col_name])) requiere 2d?


class ClustererKmeans(Clusterer):
    """
    Realiza un clustering utilizando Kmeans
    """

    def __init__(self, df,
                 # s: SessionData, tipo_variables: TipoVariables,
                 usar_categoricas=False, col_name='cluster',
                 n_max=1000, n_clusters=None):
        """

        Parameters
        ----------
        df: pandas df
            Dataframe sobre el cual se realizará el clustering
        s: SessionData
            Sesión actual
        tipo_variables:TipoVariables
        usar_categoricas: bool
            Utilizar las variables categóricas? En caso afirmativo se realiza transformación a variables dummy
        col_name: str
            Nombre que se la dara a la columna que contenga el número de cluster a que pertenece el registro
        n_max: int
            Número máximo de registros. Si es mayor que esto, se hará un sampleo. Kmeans es lento con números de puntos
            mayor que 1000
        n_clusters: int
            Número de clusteres a calcular. Si se pone None, se realizará un escaneo entre 2 y 15 clusters
        """
        semilla = 12
        if df.shape[0] > n_max:
            print(
                '**Realizaremos un sampleo ya que KMEANS es lento con mayor cantidad de puntos. Utilizaremos n_max={}'.format(
                    n_max))
            df0 = df.sample(n_max, random_state=semilla)
        else:
            df0 = df.copy()

        Clusterer.__init__(self, df0, col_name)
        max_clusters = 15
        self.algoritmo = CLUSTER_KMEANS

        # df1 = _excluir_vars(df0, tipo_variables)
        # df2 = _preprocessing(df1, s, tipo_variables, usar_categoricas)
        df2 = df0
        print('falta incluir preproceso de variables')  # todo
        X = _scale_feats(df2)

        self.clusters = _compute_kmeans(X, max_clusters, n_clusters)  # se calculan desde 2 a 15

    def get_labels_n(self, n_clusters):
        """
        Devuelve un dataframe de pandas con el índice original y una columna con la etiqueta del cluster

        Parameters
        ----------
        n_clusters: int
            número de clusters

        Returns
        -------
        df_clusters: pandas df

        """
        max_clusters = len(self.clusters)
        if max_clusters < n_clusters:
            print('*** ERROR: Sólo hay calculados {} clusters'.format(max_clusters))
            return None

        self._set_df_labels(self.clusters[n_clusters - 1].labels_)
        self.n_clusters = n_clusters

        return self.get_labels()

    def get_labels(self):
        if self._df_labels is None:
            print(
                '** WARN En caso de Kmeans se debe indicar el número de clusters a utilizar. Use get_labels_n(n_clusters)')
        else:
            return Clusterer.get_labels(self)


class ClustererDBScan(Clusterer):
    """
    Realiza un clustering utilizando DBScan
    """

    def __init__(self, df,
                 # tipo_variables: TipoVariables, s: SessionData,
                 usar_categoricas=False, col_name='cluster',
                 db_eps=0.5, db_min_samp=10, skip_preprocessing=False):
        """

        Parameters
        ----------
        df: pandas df
            Dataframe sobre el cual se realizará el clustering
        usar_categoricas: bool
            Utilizar las variables categóricas? En caso afirmativo se realiza transformación a variables dummy
        col_name: str
            Nombre que se la dara a la columna que contenga el número de cluster a que pertenece el registro
        db_eps: float
            Parámetro de DBScan. Es la máxima distancia entre dos muestras para ser consideradas como vecinas. (**No
            corresponde** a la distancia máxima entre los puntos de un cluster. Es importante fijarla correctamente para
            un buen resultado.
        db_min_samp: int
            Número de muestras en una vecindad de un punto para que sea considerado como un punto central
        """
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        Clusterer.__init__(self, df, col_name)
        self._algoritmo = CLUSTER_DBSCAN

        if skip_preprocessing:
            df2 = df
        else:
            pass
            # df1 = _excluir_vars(df, tipo_variables)
            # df2 = _preprocessing(df1, s, tipo_variables, usar_categoricas)

        X = StandardScaler().fit_transform(df2)
        self.X = X

        print('** Utilizando los valores de parámetros eps={}, min_samp={}'.format(db_eps, db_min_samp))
        db = DBSCAN(eps=db_eps, min_samples=db_min_samp).fit(X)
        self._assess(X, db, db_eps)

    def _assess(self, X, db, eps, imprimir=True):
        import numpy as np
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        self._set_df_labels(labels)
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        self.n_clusters = n_clusters_
        try:
            silhouette_score = metrics.silhouette_score(X, labels)
        except Exception as ex:
            silhouette_score = -9999
            print('** ERROR calculando métrica Silhouette: {}'.format(ex))

        dic = {'eps':       [eps], 'n_clusters': [n_clusters_], 'n_noise': [n_noise_],
               'sihouette': [silhouette_score]}

        if imprimir:
            print('Número estimado de clusters: %d' % n_clusters_)
            print('Número estimado de puntos considerados ruido: %d' % n_noise_)
            # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
            # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
            # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
            # print("Adjusted Rand Index: %0.3f"
            #       % metrics.adjusted_rand_score(labels_true, labels))
            # print("Adjusted Mutual Information: %0.3f"
            #       % metrics.adjusted_mutual_info_score(labels_true, labels))

            print("Coeficiente Silhouette: %0.3f (Es mejor cercano a 1)"
                  % silhouette_score)

        return dic

        # #############################################################################
        # # Plot result
        # import matplotlib.pyplot as plt
        #
        # # Black removed and is used for noise instead.
        # unique_labels = set(labels)
        # colors = [plt.cm.Spectral(each)
        #           for each in np.linspace(0, 1, len(unique_labels))]
        # for k, col in zip(unique_labels, colors):
        #     if k == -1:
        #         # Black used for noise.
        #         col = [0, 0, 0, 1]
        #
        #     class_member_mask = (labels == k)
        #
        #     xy = X[class_member_mask & core_samples_mask]
        #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #              markeredgecolor='k', markersize=14)
        #
        #     xy = X[class_member_mask & ~core_samples_mask]
        #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #              markeredgecolor='k', markersize=6)
        #
        # plt.title('Estimated number of clusters: %d' % n_clusters_)
        # plt.show()

    def scan_parameters(self):
        """
        Realiza una exploración para diferentes valores de eps
        """
        import numpy as np
        db_min_samp = 10
        epss = list(np.arange(1, 20) / 10)
        df_res = pd.DataFrame()
        for eps in epss:
            print('** Calculando con eps={}'.format(eps))
            db = DBSCAN(eps=eps, min_samples=db_min_samp).fit(self.X)
            dic = self._assess(self.X, db, eps, imprimir=False)
            row = pd.DataFrame.from_dict(dic).set_index('eps')
            df_res = pd.concat([df_res, row])
        display(df_res)


def hbdscan(sample, cluster_min=150, epsilon_clustering=0.3):
    #     https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
    import hdbscan
    # cluster_max = cluster_min*2
    clusterer_1 = hdbscan.HDBSCAN(min_cluster_size=cluster_min,
                                  min_samples=1,  # Mientras más grande, más noise points
                                  cluster_selection_epsilon=epsilon_clustering)
    clusterer_1.fit(sample[['UMAP_x', 'UMAP_y']])
    return clusterer_1.labels_


def _preprocessing(df, s, tipo_variables, usar_categoricas):
    if not usar_categoricas:
        cat_all = tipo_variables.get_v_cat_all()
        no_categoricas = list(set(df.columns) - set(cat_all))
        print('** No se utilizarán las siguientes variables: {}'.format(cat_all))
        df = df[no_categoricas]

    df2 = _basic_imputation(df, tipo_variables, False)  # o se imputa o se quitan

    if usar_categoricas:
        df3 = _categorical_to_dummy(df2, tipo_variables, s)  # se necesita para las categóricas
    else:
        df3 = df2

    return df3


def _compute_kmeans(df, max_clusters, n_clusters, semilla=12):
    """
    Calcula kmeans
    Parameters
    ----------
    df
    max_clusters

    Returns
    -------

    """
    import matplotlib.pyplot as plt
    t = inicia('Calculando kmeans')
    sum_of_squared_distances = []

    if n_clusters is None:
        K = range(2, max_clusters + 1)
    else:
        K = [n_clusters]

    kms = []
    for k in K:
        print('  n_clusters: {}'.format(k))
        km = KMeans(n_clusters=k, random_state=semilla)
        km = km.fit(df)
        sum_of_squared_distances.append(km.inertia_)
        kms.append(km)

    if n_clusters is None:
        plt.plot(K, sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    tardado(t)
    return kms


def _scale_feats(data):
    """

    Parameters
    ----------
    data

    Returns
    -------

    """
    # mms = MinMaxScaler()
    mms = StandardScaler()
    mms.fit(data)
    # data_transformed = mms.transform(data.dropna())
    data_transformed = mms.transform(data)
    return data_transformed


def _excluir_vars(df, tipo_variables):
    excluir = tipo_variables.get_v_fecha() + tipo_variables.get_v_texto()
    excluir = [x for x in excluir if x in df.columns]
    print('** Excluimos variables de texto y de fecha: {}'.format(excluir))

    return df.drop(columns=excluir)


def _categorical_to_dummy(df, categorical_features
                          # , tipo_variables: TipoVariables, s: SessionData
                          ):
    """
    Si no tiene categóricas devuelve el mismo dataset
    """
    t = inicia('Categóricas a dummy')
    data = df.copy()

    # categorical_features = list(set(df.columns) & (set(tipo_variables.get_v_cat_all()) - set([s.target])))

    if len(categorical_features) > 0:
        for col in categorical_features:
            dummies = pd.get_dummies(data[col], prefix=col)
            data = pd.concat([data, dummies], axis=1)
            data.drop(col, axis=1, inplace=True)
        print('*** Al convertir las variables categóricas en onehot, quedamos con un dataset con {} columnas'.
              format(data.shape[1]))

    tardado(t)
    return data


def _basic_imputation(df, cat_all, numericas_all,
                      # tipo_variables: TipoVariables,
                      imputa_categoricas=True):
    """
Imputa las valores faltantes, con el valor medio o más frecuente dependiendo si es numérica o categórica
Además elimina las textuales
    :param imputa_categoricas:
    :param df:
    :param tipo_variables:
    """
    t = inicia('Imputación Básica')

    nrows_na = df.shape[0] - df.dropna().shape[0]
    if nrows_na == 0:
        return df
    print('** Aplicando imputación básica a {} filas que contienen NA (de {}). Se imputa con la media total'.format(
        nrows_na, len(df)))
    # numéricas
    # numericas_all = tipo_variables.get_v_numericas_all(df)
    print('numéricas:', numericas_all)
    df_num = _simple_imp(df, numericas_all, 'mean')
    # categóricas
    # cat_all = tipo_variables.get_v_cat_all(df)
    if imputa_categoricas:
        df_cat = _simple_imp(df, cat_all, 'most_frequent')
    else:
        df_cat = df[cat_all]
    tardado(t)

    return df_num.join(df_cat)


def _simple_imp(df, variables, strat):
    df2 = df[variables]
    imp = SimpleImputer(strategy=strat)
    r = imp.fit_transform(df2)
    df2 = pd.DataFrame(r, columns=df2.columns, index=df2.index)

    return df2
