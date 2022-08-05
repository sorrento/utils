from embedding import plot_umap
from plots import radar_plot, plot_save


class Characterizator:
    def __init__(self, df_all, df_comp, df_map, df_clasificacion, file_variable_names='survey'):
        [var_id, var_cat, var_money, var_num] = read_variables(
            file_variable_names)  # nombres de variables training separadas por tipo

        self.df_comp = df_comp
        self.df_all = df_all
        self.df_map = df_map
        self.var_icd = var_id
        self.var_cat = var_cat
        self.var_num = var_num
        self.var_money = var_money
        self.variables = var_cat + var_money + var_num

        self.df_map['cluster'] = df_clasificacion['cluster']
        self.df_all['cluster'] = df_clasificacion['cluster']
        self.df_comp['cluster'] = df_clasificacion['cluster']

        plot_cluster_sizes(df_clasificacion, cluster_col_name='cluster')

    def set_cluster_id(self, clu):
        self.df_other, self.df_cluster = self.split_dataset_by_cluster(self.df_all, cluster=clu)

        plot_umap_one_clu(self.df_map, 'CLUSTER ' + str(clu), clu)
        self.resumen = get_resumen(self.df_other, self.df_cluster)

        self.v_num, self.v_cat = variables_diferenciadoras(self.resumen, self.var_num, self.var_money, self.var_cat)

        # resumen[resumen['rango'] != 0].head(7)

    def split_dataset_by_cluster(self, df, cluster):
        # vars = self.variables + ['id']

        df_other = df[df['cluster'] != cluster].copy()  # [vars]
        df_cluster = df[df['cluster'] == cluster].copy()  # [vars]

        df_other["tipo"] = "other"
        df_cluster["tipo"] = "cluster"

        print('Individuos en cluster:', len(df_cluster), ' Other:', len(df_other))

        return df_other, df_cluster

    def plot_hists(self):
        try:
            # numericas
            histogram_comparison(self.df_other, self.df_cluster, self.v_num)  # la variables numéricas más distintas
            # categoricas
            histograms_dataframe_cat2(self.df_comp, self.v_cat, self.df_other, self.df_cluster)
        except ValueError:
            print('Primer llamar a set_cluster_id(n)')

    def save_umaps_heatmaps(self):
        plot_umap_features_file(self.df_comp, self.df_map, self.var_cat, categorical=True, prefix='Categorical')
        plot_umap_features_file(self.df_all, self.df_map, self.var_money, categorical=False, prefix='Money')
        plot_umap_features_file(self.df_all, self.df_map, self.var_num, categorical=False, prefix='Numerical')

    def plot_boxplot(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.concat([self.df_other, self.df_cluster])
        df_melt = pd.melt(df[self.v_num + ['tipo']], id_vars=['tipo'])
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
        sns.boxplot(y="variable", x="value", hue="tipo",
                    data=df_melt, palette="Set3", orient="h").set(title='Variables numéricas')
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.show()


class CharacterizatorSimple:
    def __init__(self, df_all, df_desc=None, folder=None, **tipos_var):
        """

        Parameters
        ----------
        df_all: tiene los valores de las variables, las de umap y el id de cluster

        """

        self.tipos_var = tipos_var
        self.df_all = df_all
        self.df_desc = df_desc

        self.df_other = None
        self.df_cluster = None
        self.resumen = None
        self.v_num = None
        self.v_cat = None
        self.folder = folder
        self.clu = None

        plot_cluster_sizes(df_all, cluster_col_name='cluster', write_png=True, folder=folder)

    def set_cluster_id(self, clu, solo_var_diferenciadoras=True):
        self.clu = clu
        self.df_other, self.df_cluster = self.split_dataset_by_cluster(self.df_all, cluster=clu)

        plot_umap_one_clu(self.df_all, 'CLUSTER ' + str(clu), clu, write_png=True, folder=self.folder)
        self.resumen = get_resumen(self.df_other, self.df_cluster)
        if solo_var_diferenciadoras:
            self.v_num, self.v_cat = variables_diferenciadoras(self.resumen, self.tipos_var['var_num'],
                                                               self.tipos_var['var_money'], self.tipos_var['var_cat'])
        else:
            self.v_num, self.v_cat = self.tipos_var['var_num'] + self.tipos_var['var_money'], self.tipos_var['var_cat']

        # resumen[resumen['rango'] != 0].head(7)

    def split_dataset_by_cluster(self, df, cluster):
        # vars = self.variables + ['id']

        df_other = df[df['cluster'] != cluster].copy()  # [vars]
        df_cluster = df[df['cluster'] == cluster].copy()  # [vars]

        df_other["tipo"] = "other"
        df_cluster["tipo"] = "cluster"

        print('Individuos en cluster:', len(df_cluster), ' Other:', len(df_other))

        return df_other, df_cluster

    def plot_hists(self, selected_vars=None, write_png=True):
        try:
            # numéricas
            if selected_vars is None:
                num = self.v_num
            else:
                num = selected_vars

            histogram_comparison(self.df_other, self.df_cluster, num,
                                 self.df_desc, write_png, self.folder,
                                 cluster_id=self.clu)  # la variables numéricas más distintas
            # categoricas
            if len(self.v_cat) > 0:
                histograms_dataframe_cat2(self.df_all, self.v_cat, self.df_other, self.df_cluster, self.df_desc)
        except ValueError as e:
            print(e)
            print('Primer llamar a set_cluster_id(n)')

    def save_umaps_heatmaps(self):
        plot_umap_features_file(self.df_comp, self.df_map, self.var_cat, categorical=True, prefix='Categorical')
        plot_umap_features_file(self.df_all, self.df_map, self.var_money, categorical=False, prefix='Money')
        plot_umap_features_file(self.df_all, self.df_map, self.var_num, categorical=False, prefix='Numerical')

    def plot_boxplot(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.concat([self.df_other, self.df_cluster])
        df_melt = pd.melt(df[self.v_num + ['tipo']], id_vars=['tipo'])
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
        sns.boxplot(y="variable", x="value", hue="tipo",
                    data=df_melt, palette="Set3", orient="h").set(title='Variables numéricas')
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.show()

    def plot_radar(self, desc=None):
        median_clu = self.df_cluster[self.v_num].median()
        median_oth = self.df_other[self.v_num].median()
        desc = self.df_desc.to_dict()['value']  # dictionary
        radar_plot(median_clu, median_oth, self.v_num, desc, write_png=True, folder=self.folder,
                   filename=str(self.clu) + '_radar')


def plot_umap_features_file(df, df_map, variables, categorical, prefix='categorical'):
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    buf = df.copy()
    if categorical:
        buf[variables] = buf[variables].astype('category')  # nos aseguramos para que pinte bien
    # dividimos variables en grupos de
    var6 = list(chunks(variables, 6))
    for i in range(0, len(var6)):
        vars = var6[i]
        print(vars)
        plot_umap_features6(buf, df_map, vars, categorical, prefix + '_' + str(i), show=False)


def histogram_comparison(df1, df2, varis, df_desc=None, write_png=False, folder=None, cluster_id='clu_id'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    stop = len(varis)
    filas = (stop // 3) + 1

    kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})
    sns.set_style("white")
    plt.figure(figsize=(20, 3 * filas), dpi=80)

    for i in range(0, stop):
        va = varis[i]
        x1 = df1[va]
        x2 = df2[va]

        # plt.subplot(1, stop + 1, i + 1)
        plt.subplot(filas, 3, i + 1)
        titulo = va
        if df_desc is not None:
            if va in df_desc.index:
                titulo = df_desc.loc[va].iloc[0]
        sns.distplot(x1, color="dodgerblue", label="others", **kwargs).set(title=titulo)
        sns.distplot(x2, color="orange", label="cluster", **kwargs)
        plt.legend()

    plt.tight_layout()
    plot_save(write_png, folder, str(cluster_id) + '_' + 'histogramas')
    plt.show()


def get_resumen(df_other, df_cluster):
    """
    Compara las variables entre dos grupos, ordenándo por las más distinta (medias más sepraradas) normalizado por
    el rango
    :param df_other:
    :param df_cluster:
    :return:
    """
    import pandas as pd

    # promedio de cada columna
    medias1 = df_other.mean().to_frame()
    medias1.columns = ['mean1']

    medias2 = df_cluster.mean().to_frame()
    medias2.columns = ['mean2']

    medias = pd.concat([medias1, medias2], axis=1, join='inner')

    limits = df_other.quantile([.01, .99]).transpose()

    limits = pd.concat([df_other.min(), df_other.max(), limits], axis=1, join='inner')
    limits.columns = ['min_ext', 'max_ext', 'min', 'max']

    resumen = pd.concat([limits, medias], axis=1, join='inner')
    resumen["delta"] = abs(resumen["mean2"] - resumen["mean1"])
    resumen['rango'] = resumen['max'] - resumen['min']
    resumen["delta_norm"] = resumen["delta"] / resumen['rango']

    resumen = resumen.sort_values(by=['delta_norm'], ascending=False)

    return resumen


def variables_diferenciadoras(resumen_, var_num, var_money, var_cat):
    """
    Se eligen las variablesmás distintas, de acuerdo al resumen
    :param resumen_:
    :param var_num:
    :param var_money:
    :param var_cat:
    :return:
    """
    a = resumen_[resumen_['rango'] != 0].head(20).index.values.tolist()
    # Cogemos los nombres de las variables más distintas
    primeros = [x for x in a if x not in ['UMAP_x', 'UMAP_y']]
    b = []
    for p in primeros:
        if p not in b:
            b.append(p)
    b = b[0:8]  # las 8 más relevantes
    # las separamos por numéricas y por categóricas
    v_num = list(set(var_num + var_money) & set(b))
    v_cat = list(set(var_cat) & set(b))

    return v_num, v_cat


def histograms_dataframe_cat2(df_comp, var_cat, df_other, df_cluster, df_desc=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    mask_cluster = df_comp['id'].isin(df_cluster.id)
    buf_cluster = df_comp[mask_cluster]
    mask_other = df_comp['id'].isin(df_other.id)
    buf_other = df_comp[mask_other]

    nvar = len(var_cat)

    plot_cols = 4  # cuantos histogramas de ancho

    plot_filas = (nvar // plot_cols) + 1

    fig = plt.figure(figsize=(15, 5 * plot_filas))

    for i in range(0, nvar):
        v = var_cat[i]
        res2 = test(buf_cluster, buf_other, v)
        plt.subplot(plot_filas, plot_cols, i + 1)
        ax = sns.barplot(x=v, y="value", hue='variable', data=res2)
    plt.tight_layout()


def test(buf_cluster, buf_other, v):
    resumen = resume_cluster_other(buf_other, buf_cluster, v)
    resumen[v] = resumen.index
    res2 = resumen.melt(id_vars=v)
    return res2


def resume_cluster_other(buf_other, buf_cluster, v):
    import pandas as pd
    aa = resumen(buf_other, v, 'other')
    bb = resumen(buf_cluster, v, 'cluster')

    res = pd.merge(aa, bb, how='outer', on=v).sort_values(v)

    return res


def resumen(df, v, nom):
    import pandas as pd
    df_ = pd.DataFrame({nom: df[[v]].value_counts() / len(df)})
    df_.reset_index(level=0, inplace=True)

    return df_


def plot_umap_features6(df, df_map, variables, categorical: bool, png=None, show=True):
    import matplotlib.pyplot as plt

    l1 = len(df)
    l2 = len(df_map)
    if l1 != l2:
        print('Error, los dos dataset tienen diferente largo. DF:', l1, ' df_map: ', l2)
        return None

    f = 2.5
    fig = plt.figure(figsize=(16 * f, 9 * f), dpi=130)
    n_samples = str(df.shape[0])
    fig.suptitle(
        '.                                                                                             UMAP. Samples: ' + n_samples,
        fontsize=12)

    stop = min(len(variables), 6)
    for i in range(0, stop):
        variable = variables[i]
        ax = plt.subplot(2, 3, i + 1)

        if categorical:
            df[variables] = df[variables].astype('category')  # nos aseguramos para que pinte bien
            scatter = ax.scatter(df_map['UMAP_x'], df_map['UMAP_y'], c=(df[variable].cat.codes), cmap="Set1", alpha=0.5)
        else:
            scatter = ax.scatter(df_map['UMAP_x'], df_map['UMAP_y'], c=(df[variable]), alpha=0.5, s=1)

        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*scatter.legend_elements(), loc="best")
        # legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="variable")
        ax.add_artist(legend1)

        plt.title(variable)
    plt.tight_layout()
    if png is not None:
        filename = 'data_out/' + png + '.png'
        print('...saving ', filename)
        plt.savefig(filename)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cluster_sizes(df_clasificacion, cluster_col_name='cluster', write_png=True, folder=None):
    """

    :param df_map_cluster: índice es el nombre del cluster, el valor es el size

    Parameters
    ----------
    write_png
    cluster_col_name
    df_clasificacion
    """
    import squarify
    import matplotlib.pyplot as plt
    th = 5  # Cuántos se pintan con el tamaño escrito en texto

    df_map_cluster = df_clasificacion[cluster_col_name].value_counts()  # Tamaño de los clusters en una primera ronda
    label = df_map_cluster.index.values.tolist()
    sizes = df_map_cluster.tolist()

    label2 = [str(i) + '_' + str(j) for i, j in zip(label, sizes)]
    label3 = label2[:th] + label[th:]

    print('n clusters = ' + str(len(df_map_cluster)) + '. Los más pequeños tienen estos tamaños: ' + str(sizes[-5:]))

    plt.figure(figsize=(15, 10))
    squarify.plot(sizes, label=label3, alpha=0.4)
    plt.axis('off')

    plot_save(write_png, folder)
    plt.show()


def plot_umap_one_clu(df, title, clu, x=5, y=5, write_png=False, folder=None):
    df2 = df[['UMAP_x', 'UMAP_y', 'cluster']].copy()
    df2['cluster'] = (df2['cluster'] == clu).map({True: clu, False: -2}).astype('category')

    plot_umap(df2, title, x, y, column_to_colour='cluster', write_png=write_png, filename=str(clu) + '_cluster',
              folder=folder)


def clasifica_variables(df, var_id, file='survey'):
    import pandas as pd
    import pickle
    th = 12

    # categorical
    var = df.columns.tolist()
    var_feat = set(var) - set(var_id)
    n_values = df[var_feat].nunique().to_frame().sort_values(by=[0], ascending=False)
    n_values.columns = ['n']
    # non_categoricas_actually= ['MARHT', 'JWRIP', 'DRIVESP']
    # categoricas.remove('ST') #state
    # for var in non_categoricas_actually:
    #     categoricas.remove(var)
    # consideramos como categóricas aquellas que tengan menos de 12 valores:
    var_cat = n_values[n_values.n < th].index.values.tolist()
    var_num = set(var_feat) - set(var_cat)

    # las de money las vemos por el rango
    a = pd.DataFrame({'sd': df[var_num].std()})
    var_money = a[a.sd > 1000].index.tolist()

    var_num = list(set(var_num) - set(var_money))

    f = open('data_med/' + file + '_vars.pkl', 'wb')
    pickle.dump([var_id, var_cat, var_money, var_num], f)
    f.close()

    return var_cat, var_money, var_num


def read_variables(file):
    import pickle
    pickle_file = open("data_med/" + file + "_vars.pkl", "rb")
    objects = []
    while True:
        try:
            objects.append(pickle.load(pickle_file))
        except EOFError:
            break
    pickle_file.close()

    return objects[0]
