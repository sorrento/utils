import logging
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from IPython.core.display import display

from ut.io import lista_files_recursiva, get_filename

FORMAT_DATE = "%Y%m%d"
FORMAT_CLASSIC = '%Y-%m-%d'
FORMAT_GRINGO = '%m/%d/%Y'
FORMAT_DATETIME = '%Y-%m-%d %H:%M:%S.%f'
FORMAT_DATETIME2 = '%Y-%m-%d %H:%M:%S'
FORMAT_UTC = '%Y-%m-%dT%H:%M:%S.%fZ'
FORMAT_UTC2 = '%Y-%m-%d %H:%M:%S.%f+00:00'


def make_folder(path, verbose=True, delete_if_exists=False):
    import os
    path = os.path.abspath(path)
    try:
        if os.path.isdir(path):
            if verbose:
                print('Already exists: {}'.format(path))
            if delete_if_exists:
                import shutil
                print('  Deleting existing one')
                shutil.rmtree(path)
        else:
            print('Creating directory ', path)
            os.mkdir(path)
        return path + '/'

    except OSError:
        print('Ha fallado la creación de la carpeta %s' % path)


def inicia(texto, level=0):
    ahora = time.time()
    print('\n' + get_spaces(level) + '** Iniciando: {}'.format(texto))

    return [ahora, texto, level]


def get_spaces(level):
    if level == 0:
        spaces = ''
    else:
        ll = ['  ' for x in range(level)]
        # ll.append(str(level))
        spaces = ''.join(ll)
    return spaces


def tardado(lista: list):
    start, texto, level = lista

    elapsed = (time.time() - start)
    strftime = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(get_spaces(level) + '** Finalizado {}.  Ha tardado {}'.format(texto, strftime))
    return elapsed


def json_save(dic, path, datos_desc='', indent=None, overwrite=False, df_to_dict=False):
    """

    :param df_to_dict:
    :param overwrite:
    :param indent:
    :param dic:
    :param path:
    :param datos_desc: sólo para mostrar en un print
    """
    import json
    import os
    if path[-5::] == '.json':
        path2 = path
    else:
        path2 = path + '.json'

    existe = os.path.exists(path2)
    if existe:
        print('** File {} already exists.'.format(path2))
        if not overwrite:
            print('Skipping. Set overwrite=True to overwrite')
            return None

    print('** Guardado los datos ' + datos_desc + ' en {}'.format(path2))

    if df_to_dict:
        print('** transforming dataframe to dict. WARN: the Result will have only one level')
        dic = {k: dic[k].to_dict(orient='index') for k in dic}

    try:
        with open(path2, 'w', encoding="utf-8") as outfile:
            if indent is None:
                json.dump(dic, outfile, ensure_ascii=False)
            else:
                json.dump(dic, outfile, ensure_ascii=False, indent=indent)
    except Exception as e:
        print('** ERROR **** \n    ', e)
        example = dic[list(dic.keys())[0]]
        are_dataframes = isinstance(example, pd.DataFrame)
        if are_dataframes:
            print('** subelement are pandas dataframes: will be transformed to dict')
        json_save(dic, path, datos_desc, indent, overwrite, df_to_dict=True)


def json_read(json_file, keys_as_integer=False):
    import json
    print(f'** Leyendo json: {json_file}')
    if os.path.isfile(json_file):
        with open(json_file, encoding="utf-8") as in_file:
            data = json.load(in_file)

        if keys_as_integer:
            data = {int(x) if x.isdigit() else x: data[x] for x in data.keys()}
    else:
        print(f'** No existe el fichero{json_file}')
        data = {}

    return data


def json_from_string(s):
    import json
    return json.loads(s.replace("'", "\""))


def json_update_file(path, dic):
    import os
    existe = os.path.exists(path)
    if existe:
        j = json_read(path, keys_as_integer=True)
        j.update(dic)
    else:
        j = dic
    json_save(j, path, overwrite=True)


def get_now(utc=False):
    ct = now(utc)
    # ts = ct.timestamp()
    # print("timestamp:-", ts)

    return str(ct)  # podríamos quedarnos con el objeton (sin str)


def now(utc=False):
    from datetime import timezone, datetime
    if utc:
        tz = timezone.utc
    else:
        tz = None

    return datetime.now(tz)


def get_now_format(f=FORMAT_DATE, utc=False):
    """

    :param utc:
    :param f: ver https://pythonexamples.org/python-datetime-format/
    :return:
    """
    ct = now(utc)
    return ct.strftime(f)


def flatten(lista):
    """
transforma una lista anidada en una lista de componenetes únicos oredenados
OJO: SÓLO SI NO SE REPITEN ELEMENTOS
    :param lista:
    :return:
    """
    from itertools import chain

    # los que no están en anidados los metemos en lista, sino no funciona la iteración
    lista = [[x] if (type(x) != list) else x for x in lista]
    flat_list = list(chain(*lista))

    return sorted(list(set(flat_list)))


def log10p(x):
    """
    equivalente a log1p pero en base 10, que tiene más sentido en dinero
    :param x:
    :return:
    """
    import numpy as np
    return np.log10(x + 1)


def abslog(x):
    """
    función logaritmica que incluye el 0, es espejo en negativos, y es "aproximadamente" base 10
    :param x:
    :return:
    """
    if x < 0:
        y = -log10p(-x)
    else:
        y = log10p(x)
    return y


def df_save(df, path, name, save_index=False, append_size=True, overwrite=False, zip=False):
    import os

    name = name.replace('/', '_')

    if append_size:
        nrows = df.shape[0]
        if nrows > 1000:
            s_rows = str(round(nrows / 1000)) + 'k'
        else:
            s_rows = str(nrows)

        middle = '_' + s_rows + '_' + str(df.shape[1])
    else:
        middle = ''

    if not os.path.isdir(path):
        make_folder(path)

    ext = '.zip' if zip else '.csv'
    filename = path + '/' + name + middle + ext

    if os.path.isfile(filename):
        print('** File {} already exists **'.format(filename))
        if not overwrite:
            print(' Skipping. Set overwrite=True for overwrite')
            return filename

    print('Saving dataset: {}'.format(filename))
    if zip:
        df.to_csv(filename, index=save_index, sep=';', compression='zip')
    else:
        df.to_csv(filename, index=save_index, sep=';')

    return filename


def win_exe(cmd):
    import os
    from sys import platform
    print("**Executing in Windows shell:" + cmd)
    if platform == 'win32':
        cmd = cmd.replace('/', '\\')
    out = os.popen(cmd).read()
    print('**OUT:{}'.format(out))
    return out


def in_k(n, dec=0):
    return str(round(n / 1000, dec)) + 'k'


def time_from_str(s, formato, silent=False):
    # https://www.programiz.com/python-programming/datetime/strptime
    try:
        res = datetime.strptime(s, formato)
    except Exception as e:
        if not silent: print(e)
        res = 'no_date'
    return res


def time_to_str(t, formato=FORMAT_DATE):
    import numpy
    try:
        if type(t) is numpy.datetime64:
            import datetime
            t = pd.to_datetime(str(t))
        elif type(t) is str:
            print(f'ya es string {t}, puede que no tenga el formato apropiado')
            return t

        res = t.strftime(formato)
    except Exception as e:
        res = 'no'
    return res


def seq_len(ini, n, step):
    """
crea una secuencia de n enteros, a distancia step
    :param n:
    :param step:
    :param ini:
    :return:
    """
    end = ini + step * (n + 1)
    return list(np.arange(ini, end, step)[:n])


def nearest(x, lista):
    """
devuelve el número más cercano a x de la lista
    :param x:
    :param lista:
    :return:
    """
    deltas = [abs(s - x) for s in lista]
    pos = list_min_pos(deltas)

    return lista[pos]


def list_min_pos(lista, f=min):
    """
da la (primera) posición del elemento más pequeño
    :param lista:
    :return:
    """
    value = f(lista)
    position = lista.index(value)

    return position, value


def list_freqs(myList, as_df=False):
    frequencyDict = dict()
    visited = set()
    listLength = len(myList)
    for i in range(listLength):
        if myList[i] in visited:
            continue
        else:
            count = 0
            element = myList[i]
            visited.add(myList[i])
            for j in range(listLength - i):
                if myList[j + i] == element:
                    count += 1
            frequencyDict[element] = count
    #     print("Input list is:", myList)
    #     print("Frequency of elements is:")
    #     print(frequencyDict)

    if as_df:
        df = pd.DataFrame.from_dict(frequencyDict, 'index').rename(columns={0: 'n'}).sort_values('n', ascending=False)
        return df

    return frequencyDict


def json_update_file(path, dic):
    import os
    existe = os.path.exists(path)
    if existe:
        j = json_read(path)
        j.update(dic)
    else:
        j = dic
    json_save(j, path, overwrite=True)


def get_iso_week_from_date(date):
    """Return the ISO week from a date.

    Args:
        date (str): Date

    Returns:
        week (int): ISO week
    """
    date = pd.to_datetime(date)
    return date.isocalendar()


def get_start_date_of_isoweek(year, week, date_format='datetime'):
    """
    Get the start date of the given ISO week using isocalendar()

    Args:
        year (int): Year, i.e. 2022
        week (int): Week, i.e. 34
        date_format (string): Format of the returned date, default is datetime

    Returns:
        datetime.date: Start date of the given ISO week
    """
    from isoweek import Week

    if date_format == 'datetime':
        return Week(year, week).monday()
    else:
        return Week(year, week).monday().strftime(date_format)


def timeit(func):
    def my_wrap(*args, **kwargs):
        t = inicia(func.__name__)
        x = func(*args, **kwargs)
        tardado(t)
        return x

    return my_wrap


def make_folders(folder):
    folds = [x for x in folder.split('/') if ((x != '') and ('.' not in x))]

    for i in range(1, len(folds) + 1):
        acc = '/'.join(folds[:i])
        make_folder(acc)


def time_from_quarter(y, q):
    map_q = {1: 1, 2: 4, 3: 7, 4: 10}
    txt = str(y) + str(map_q[int(q)]).zfill(2) + '01'
    return time_from_str(txt, formato=FORMAT_DATE)


def add_times(preds):
    preds['year'] = preds['DATE'].dt.year
    preds['month'] = preds['DATE'].dt.month
    preds['quarter'] = preds['month'].map(D_MAP)
    preds['Y_Q'] = preds['year'].astype(str) + '_' + preds['quarter']


def log_xls(txt, lev, silent=True):
    """
logs for reading excels process
    :param lev:
    :param txt:
    """
    msg = get_spaces(lev) + txt
    if not silent:
        print(msg)
    logging.info(msg)


def setLogging(log2file, suffix, level=logging.DEBUG,
               formato='%(asctime)s - %(levelname)s - %(message)s', log_path='./log/'):
    """
    # To write to file or Jupyter [both are not compatible)
    :param log_path:
    :param log2file: True to set file log
    :param level: logging.DEBUG, INFO, WARN OR ERROR
    :param formato:
    """
    if log2file:
        # time_format = '%Y-%m-%d %H:%M:%S'
        time_format = '%d %H:%M:%S'
        make_folder(log_path, False)
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_file = log_path + date_time + '_' + suffix + '.log'
        print('Writing log to {}'.format(log_file))
        logging.basicConfig(level=level, filename=log_file, filemode='w',
                            format=formato, datefmt=time_format)
    else:  # To see output in Jupyter
        logger = logging.getLogger()
        logger.setLevel(level)


class SheetException(Exception):
    def __init__(self, message):
        super().__init__(message)


class FilesException(Exception):
    def __init__(self, message):
        super().__init__(message)


def get_filename_patt(folder, patt, ext):
    files = lista_files_recursiva(folder, ext=ext, recursivo=False)
    matched = [x for x in files if patt in x]
    # print(f'\n\n ****Buscando: ({patt})')
    # [print(x) for x in files]
    # print('matchec')
    # [print(x) for x in matched]
    if len(matched) == 1:
        print(f'Found: {get_filename(matched[0])}')
        return matched[0]
    elif len(matched) > 1:
        matched = [get_filename(x) for x in matched]
        raise FilesException(f'Too many files with pattern ({patt}) in path {folder}: \n{matched}')
    else:
        raise FilesException(f'No file with pattern ({patt}) in path {folder}')


def get_index_name(patt, df_indices=None):
    """

    :param patt:
    :param df_indices: si se da un df que tiene columna Index, se extrae de ahí los nombres, sino se usa
    los de prioridad 1
    :return:
    """
    if df_indices is None:
        listo = priority1_index
    else:
        listo = list(df_indices.Index.unique())

    lista = [x for x in listo if patt in x]
    r = 'Nada'
    if len(lista) == 1:
        r = lista[0]
        print('Seleccionado: ', r)
    elif len(lista) == 0:
        print('NO se encuentra, estos son los que hay:', listo)
    else:
        print('Ha varios, ponga un patrón más definido:')
        for x in sorted(lista):
            print(x)
    return r


def logret_inv(logret, y0):
    # y0 * (2 - np.exp(logret))
    return y0 * np.exp(logret)


def logret(y0, yf, sat=0.75):
    change = (yf - y0) / y0
    # saturado de manera que no pueda superar el 75%
    change = [-sat if value < -sat else value for value in change]
    change = [sat if value > sat else value for value in change]  # esto no es 75% hacia abajo
    y = np.log(1 + np.array(change))
    return y


def date_from_ym(y, m):
    return time_from_str(str(int(y)) + '-' + str(int(m)) + '-1', '%Y-%m-%d')


def get_date_range(df, col_date='date'):
    dates = df[col_date].unique()
    return (min(dates)), max(dates)


def fix_col_names(df):
    """
Remueve los caracteres peligrosos de los nombres de las columnas
lightgbm da problemas con algunos nombres, asi que quitamos caracteres
    :param df:
    :return:
    """
    import re

    return df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))


def shiftea_col(df, col, lag):
    df[col + '_fut'] = df[col].shift(-lag)


def shiftea_cols(df, cols, lag, silent=True):
    # ojo que tienen que ser intervalos regulares
    df = df.sort_values('yw_start')  # variable fecha, para ordenar
    for col in cols:
        shiftea_col(df, col, lag)
    df['lag'] = lag
    if not silent:
        display(df[['lag'] + cols + [x + '_fut' for x in cols]])
    return df
