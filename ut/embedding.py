from asyncio import exceptions

# from metrics import hellinger
import umap # OJO desistalar librería umap. la buena es la umap-learn
import pandas as pd

from ut.base import abslog, timeit, make_folder
from ut.plots import plot_save


def dim_reduction(vector_matrix, kwargs):
    """UMAP dimensional reduction

    Parameters
    ----------
    vector_matrix : NumPy array
        Embedding matrix
    kwargs : dict
        Dictionary with hyperparameters

    Returns
    -------
    NumPy array

    Raises
    ------
    Exception if metric is not 'cosine' or 'hellinger'
    """

    if kwargs.get("umap_metric") == 'hellinger':
        umap_metric = hellinger  # todo me falla, porque parece que recibe 4 argumentos...
    elif kwargs.get("umap_metric") == 'cosine':
        umap_metric = 'cosine'
    else:
        raise exceptions.ValidationError(
            f"Metric {kwargs.get('umap_metric')} is not valid. Use only 'cosine' or 'hellinger'")

    umap_reductor = UMAP(
        n_neighbors=kwargs.get("umap_n_neighbors"),
        random_state=42,
        metric=umap_metric,
        min_dist=float(kwargs.get("umap_min_dist")),
        n_components=2,
    )
    vector_matrix = umap_reductor.fit_transform(vector_matrix)

    return vector_matrix


def plot_umap_feature(df, df_map, variable):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)

    scatter = ax.scatter(df_map['UMAP_x'], df_map['UMAP_y'], c=df[variable], alpha=0.5)

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="variable")
    ax.add_artist(legend1)

    n_samples = str(df.shape[0])
    plt.title(variable + ' Samples ' + n_samples)

    plt.xlabel('X_UMAP')
    plt.ylabel('Y_UMAP')

    plt.show()


def plot_umap(df, title='', x=18.5, y=10.5, write_png=True, column_to_colour=None, filename='',
              apply_log=False, quantile_cut=None, point_size=0.2, **params):
    """
    Realiza una gráfica del embedding

    Parameters
    ----------
    df: pandas df
        Dataframe que contiene las columnas UMAP_x and UMAP_y. Si se proporcionan otras columnas
        se pintarán sobre el embedding (categóricas o numéricas)
    title: str
        Título de la gráfica
    x: float
        Dimensión x del tamaño de la gráfica
    y: float
        Dimensión y del tamaño de la gráfica
    write_png: bool
        Guardar el plot en fichero
    column_to_colour: str
        Nombre de la columna a pintar sobre el embedding. Debe ser parte de df
    apply_log. boolean
        si se colorea por variable, si se debe aplicar abslog antes
    quentile_cut: float
        para pintar ponermos los valores más allá de los cuantiles a los valores de los cuantiles. valor entre cero y uno
    params:
        folder,
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    fig.set_size_inches(x, y)

    df_2 = df.copy()
    if column_to_colour is None:
        scatter = ax.scatter(df_2['UMAP_x'], df_2['UMAP_y'], c=[0] * len(df), alpha=0.5)
    else:
        if column_to_colour not in df.columns:
            print('**ERROR: la columna {} no se encuentra en el dataset'.format(column_to_colour))
            return None
        max_values = 50
        categorical = len(df[column_to_colour].value_counts()) < max_values

        if categorical:
            print('is categorical')
            df_2[column_to_colour] = df_2[column_to_colour].astype('category')
            scatter = ax.scatter(df_2['UMAP_x'], df_2['UMAP_y'], c=df_2[column_to_colour].cat.codes, cmap="Set1",
                                 alpha=0.5)
        else:

            if quantile_cut is not None:
                qs = df_2[column_to_colour].quantile([quantile_cut, 1 - quantile_cut])
                qmin = qs.iloc[0]
                qmax = qs.iloc[1]
                print('quantiles:', qs)

                df_2.loc[df_2[column_to_colour] < qmin, column_to_colour] = qmin
                df_2.loc[df_2[column_to_colour] > qmax, column_to_colour] = qmax

            if apply_log:
                colour_ = df_2[column_to_colour].apply(abslog)
                pre = 'log10_'
            else:
                colour_ = df_2[column_to_colour]
                pre = ''

            scatter = ax.scatter(df_2['UMAP_x'], df_2['UMAP_y'], c=colour_, alpha=0.5, s=point_size,
                                 cmap='viridis')  # RdYlGn')
            legend1 = ax.legend(*scatter.legend_elements(), loc="best", title=pre + "valor")
            ax.add_artist(legend1)

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="Cluster")
    ax.add_artist(legend1)

    n_samples = str(df_2.shape[0])
    plt.title(title + ' Samples ' + n_samples)

    plt.xlabel('X_UMAP')
    plt.ylabel('Y_UMAP')

    if write_png:
        if 'folder' in params:
            folder = params['folder']
        else:
            folder = ''
        plot_save(write_png, folder, filename)
    plt.show()


@timeit
def compute_umap(df, s=None, columnas='todas', tipo_variables=None, n_vecinos=30,
                 supervisado=True,
                 save_embedding=False,
                 distancia_min=1,
                 estrategia_nan=None, ignora_categoricas=True, max_n_samples=10000, plot=True, **params):
    """
    Calcula en embedding utilizando UMAP https://umap-learn.readthedocs.io/

    Parameters
    ----------
    df: pandas df
        Dataframe a clusterizar
    s: SessionData
        Sesión actual
    columnas: list
        Columnas a considerar. Se puede utilizar 'todas'
    tipo_variables:TipoVariables
        Tipos de las variables
    n_vecinos: int
        Valores pequeños hacen una estructura muy local (estructura de clusters pequeños). Puede provocar
        patrones de ruido en vez de clusters
    supervisado: bool
        Se usa la variable target para guiar el embedding
    save_embedding
    distancia_min: float
        Una distancia pequeña "empaqueta" más los puntos cercanos. Si no separa clusters, aumentar
    estrategia_nan:
         Se requiere que no haya nan's. Si se deja None, se elige automáticamemnte (los quita si la
         proporción es pequeña, si es grande, los imputa de manera simple. Valores: None, 'simple','quita'
    ignora_categoricas: bool
        si no se ignoran, se realiza un onehotencoding
    max_n_samples: int

    Returns
    -------
    Dataframe pandas
        {índice(s) - UMAP_x - UMAP_y}

    """
    import umap
    if columnas == 'todas':
        df = df.copy()
        columnas = df.columns
    else:
        df = df[columnas]

    if tipo_variables is None:
        num = df.columns
        print('tipo_variables is None: se considerarán todas las variables como numéricas')
    else:
        num = tipo_variables.get_v_numericas_all()

    if s is None:
        sem = 12
        print('No se ha puesto Session s, así que la semilla es fija:', sem)
        if 'target' in params:
            target = params['target']
        else:
            print('no se ha especificado target, ponlo en los params')
            target = None
        if 'folder' in params:
            folder = params['folder']
        else:
            print('no se podrá guardar. poner folder en los params')
            folder = None
    else:
        sem = s.semilla
        target = s.target
        folder = s.base_folder

    if ignora_categoricas:
        numi = [x for x in num if x in columnas]
        print('** Utilizando sólo las variables numéricas: \n{}'.format(numi))
        df2 = df[numi]
    else:
        df_cat = df[tipo_variables.get_v_cat_all()]
        for c in df_cat.columns:
            df_cat.loc[c] = df_cat[c].astype('object')
        df_cat_enc = pd.get_dummies(df_cat)
        df2 = df[num].join(df_cat_enc)

    # todo escanear para diferentes parámetros automáticos o manuales
    df_imp = _imputa(df2, estrategia_nan)
    print('len imp {}'.format(df_imp.shape))

    if len(df_imp) > max_n_samples:
        print('** Sampleando {} muestras para entrenar umap: '.format(max_n_samples))
        df_sampled = df_imp.sample(n=max_n_samples, random_state=sem)
    else:
        df_sampled = df_imp.copy()

    print('** size sampled:{}'.format(df_sampled.shape))
    my_umap = umap.UMAP(n_neighbors=n_vecinos,
                        min_dist=distancia_min,
                        n_epochs=100,
                        random_state=sem,
                        verbose=True,
                        # transform_seed=s.semilla).fit(df_sampled)
                        transform_seed=sem)
    try:
        if supervisado & (target is not None):
            embedding = my_umap.fit_transform(df_sampled, y=df_sampled[target])
        else:
            embedding = my_umap.fit_transform(df_sampled)
    except ValueError as e:
        print(e)
        return None
    df_map = pd.DataFrame(embedding, columns=['UMAP_x', 'UMAP_y'], index=df_sampled.index)

    title = 'N vecinos: {} | D mínima: {} | Categóricas: {} | Supervisado: {} |'.format(n_vecinos, distancia_min,
                                                                                        not ignora_categoricas,
                                                                                        supervisado)
    supervisado_ = str(n_vecinos) + '_' + str(distancia_min) + '_' + str(int(ignora_categoricas)) + '_' + str(
        int(supervisado))

    if plot:
        plot_umap(df_map, title=title, write_png=save_embedding, filename=supervisado_)

    if save_embedding and (folder is not None):
        csv_ = folder + 'umap_' + supervisado_ + '.csv'
        print('** Guardando el embedding en {}'.format(csv_))
        df_map.to_csv(csv_)

    return {'df_map': df_map, 'map': my_umap}


def predict_umap(mapa, df, cols, concat=False):
    mapa_df = mapa.transform(df[cols])  # si no es supervisado
    df_map = pd.DataFrame(mapa_df, columns=['UMAP_x', 'UMAP_y'], index=df.index)

    if concat:
        df_map = df_map.join(df)
    return df_map


def save_umap(umap_, filename):
    # https://github.com/lmcinnes/umap/issues/178
    import pickle
    pickle.dump(umap_, open(filename, 'wb'))


def load_umap(filename):
    import pickle
    loaded_model = pickle.load((open(filename, 'rb')))
    print(type(loaded_model))
    return loaded_model


def _imputa(df, estrategia_nan, tipo_variables=None):
    """

    Parameters
    ----------
    df
        Dataframe a ser imputado
    estrategia_nan: str
         Se requiere que no haya nan's. Si se deja None, se elige automáticamemnte (los quita si la
         proporción es pequeña, si es grande, los imputa de manera simple. Valores: None, 'simple','quita'
    tipo_variables

    Returns
    -------

    """
    df_imp = None
    nrows = df.shape[0]
    nrows_na = nrows - df.dropna().shape[0]
    nrows_ok = nrows - nrows_na
    perc_ok = nrows_ok / nrows
    print('** NANs: {}/{} ({}%)'.format(nrows_na, nrows, round(100 * (1 - perc_ok), 1)))

    if estrategia_nan is None:
        # print((nrows_ok > 1000) , (perc_ok > 0.8))
        if (nrows_ok > 1000) | (perc_ok > 0.8):
            print('*** Imputación automática')
            print('*** El numero de filas ok es >1000 o más del 80%, por lo que quitamos las filas que tienen nans')
            df_imp = _imputa(df, 'quita')
        else:
            df_imp = _imputa(df, 'simple', tipo_variables)
    elif estrategia_nan == 'simple':

        if tipo_variables is None:
            print('** WARN tipo_variables no puede ser None; en este caso se debe pasar a la función como parámetro')
            return None
        df_imp = _basic_imputation(df, tipo_variables)
    elif estrategia_nan == 'quita':
        print('** Quitando las filas con NA')
        df_imp = df.dropna()
    else:
        print("*** WARN la estrategia_nan debe ser None, 'simple' o 'quita' ")

    return df_imp


@timeit
def _basic_imputation(df, tipo_variables, imputa_categoricas=True):
    """
Imputa las valores faltantes, con el valor medio o más frecuente dependiendo si es numérica o categórica
Además elimina las textuales
    :param imputa_categoricas:
    :param df:
    :param tipo_variables:
    """

    nrows_na = df.shape[0] - df.dropna().shape[0]
    if nrows_na == 0:
        return df
    print('** Aplicando imputación básica a {} filas que contienen NA (de {}). Se imputa con la media total'.format(
        nrows_na, len(df)))
    # numéricas
    numericas_all = tipo_variables.get_v_numericas_all(df)
    print('numéricas:', numericas_all)
    df_num = _simple_imp(df, numericas_all, 'mean')
    # categóricas
    if imputa_categoricas:
        df_cat = _simple_imp(df, tipo_variables.get_v_cat_all(df), 'most_frequent')
    else:
        df_cat = df[tipo_variables.get_v_cat_all(df)]

    return df_num.join(df_cat)


def _simple_imp(df, variables, strat):
    from sklearn.impute import SimpleImputer
    df2 = df[variables]
    imp = SimpleImputer(strategy=strat)
    r = imp.fit_transform(df2)
    df2 = pd.DataFrame(r, columns=df2.columns, index=df2.index)

    return df2


def umap_scan(vector_matrix, nvecs, dmins, di, path='data_med/umaps/', metrics=['cosine']):
    """
Hace un barrido por los parámetros que se le dan y guarda las gráficas.
lo almacena en el diccionario di que se le da para actualizarlo
    :param vector_matrix:
    :param nvecs:
    :param dmins:
    :param di:
    :param path:
    :param metrics:
    """
    # nvecs = [7, 15, 20, 30, 40, 50]
    # dmins = [.1, .3, .5, .7, 1]
    di2 = {}
    make_folder(path)

    for me in metrics:
        # if me == 'hellinger':
        #     mef = hellinger
        # else:
        #     mef = me

        for nve in nvecs:
            for dmin in dmins:
                name = str(nve) + '_' + str(dmin) + '_' + me
                print(name)

                params_u = {
                    "umap_metric":      me,
                    "umap_n_neighbors": nve,
                    "umap_min_dist":    dmin
                }
                print(params_u)
                umap_matrix = dim_reduction(vector_matrix, params_u)
                di2[name] = {'params': params_u, 'umap': umap_matrix}

                plot_umap(as_df(umap_matrix), title=name, filename=path + name)

    di.update(di2)


def as_df(array, index):
    return pd.DataFrame(array, columns=['UMAP_x', 'UMAP_y'], index=index)


def plot_umap_anomaly(df, title='', x=18.5, y=10.5, write_png=True, column_to_colour=None, filename='', folder='',
              apply_log=False, quantile_cut=None, point_size=0.2, anomaly=False, **params):
    """
    Realiza una gráfica del embedding

    Parameters
    ----------
    df: pandas df
        Dataframe que contiene las columnas UMAP_x and UMAP_y. Si se proporcionan otras columnas
        se pintarán sobre el embedding (categóricas o numéricas)
    title: str
        Título de la gráfica
    x: float
        Dimensión x del tamaño de la gráfica
    y: float
        Dimensión y del tamaño de la gráfica
    write_png: bool
        Guardar el plot en fichero
    column_to_colour: str
        Nombre de la columna a pintar sobre el embedding. Debe ser parte de df
    apply_log. boolean
        si se colorea por variable, si se debe aplicar abslog antes
    quentile_cut: float
        para pintar ponermos los valores más allá de los cuantiles a los valores de los cuantiles. valor entre cero y uno
    params:
        folder,
        :param apply_log:
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    fig.set_size_inches(x, y)

    df_2 = df.copy()
    if column_to_colour is None:
        scatter = ax.scatter(df_2['UMAP_x'], df_2['UMAP_y'], c=[0] * len(df), alpha=0.5)
    else:
        if column_to_colour not in df.columns:
            print('**ERROR: la columna {} no se encuentra en el dataset'.format(column_to_colour))
            return None
        max_values = 50
        n_values = len(df[column_to_colour].value_counts())
        categorical = n_values < max_values

        if categorical:
            print('is categorical')
            df_2[column_to_colour] = df_2[column_to_colour].astype('category')

            if anomaly:
                df_2[column_to_colour] = df_2[column_to_colour].astype('boolean')
                ceros = df_2[~df_2[column_to_colour]]
                unos = df_2[df_2[column_to_colour]]
                scatter = ax.scatter(ceros['UMAP_x'], ceros['UMAP_y'],
                                     c='gray',
                                     alpha=0.1, s=point_size)

                scatter = ax.scatter(unos['UMAP_x'], unos['UMAP_y'],
                                     c='red',
                                     alpha=0.9, s=point_size + 2)

            else:
                scatter = ax.scatter(df_2['UMAP_x'], df_2['UMAP_y'], c=df_2[column_to_colour].cat.codes, cmap="Set1",
                                     alpha=0.5, s=point_size)
        else:

            if quantile_cut is not None:
                qs = df_2[column_to_colour].quantile([quantile_cut, 1 - quantile_cut])
                qmin = qs.iloc[0]
                qmax = qs.iloc[1]
                print('quantiles:', qs)

                df_2.loc[df_2[column_to_colour] < qmin, column_to_colour] = qmin
                df_2.loc[df_2[column_to_colour] > qmax, column_to_colour] = qmax

            if apply_log:
                colour_ = df_2[column_to_colour].apply(abslog)
                pre = 'log10_'
            else:
                colour_ = df_2[column_to_colour]
                pre = ''

            scatter = ax.scatter(df_2['UMAP_x'], df_2['UMAP_y'], c=colour_, alpha=0.5, s=point_size,
                                 cmap='viridis')  # RdYlGn')
            legend1 = ax.legend(*scatter.legend_elements(), loc="best", title=pre + "valor")
            ax.add_artist(legend1)

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="Cluster")
    ax.add_artist(legend1)

    n_samples = str(df_2.shape[0])
    plt.title(title + ' Samples ' + n_samples)

    plt.xlabel('X_UMAP')
    plt.ylabel('Y_UMAP')

    if write_png:
        plot_save(write_png, folder, filename)
    plt.show()
