import os


def getmtime(filename):
    """Return the last modification time of a file, reported by os.stat()."""
    return os.stat(filename).st_mtime


def getatime(filename):
    """Return the last access time of a file, reported by os.stat()."""
    return os.stat(filename).st_atime


def getctime(filename):
    """Return the metadata change time of a file, reported by os.stat()."""
    return os.stat(filename).st_ctime


def lista_files_recursiva(path, ext):
    """
devuelve la lista de archivos en la ruta, recursivamente, de la extensión especificada. la lista está ordenada por fecha
de modificación
    :param path:
    :param ext:
    :return:
    """
    import glob
    lista = glob.glob(path + '**/*.' + ext, recursive=True)
    lista = sorted(lista, key=getmtime, reverse=True)

    return lista


def fecha_mod(file):
    """
entrega la fecha de modificación de un archivo como un número yyyymmdd
    :param file:
    :return:
    """
    import datetime
    dt = datetime.datetime.fromtimestamp(getmtime(file))
    return int(dt.strftime('%Y%m%d'))


def get_filename(path):
    """
obtiene el nombre del fichero desde el path completo
    :param path:
    :return:
    """
    return os.path.basename(path)


def escribe_txt(txt, file_path):
    text_file = open(file_path, "wt")
    n = text_file.write(txt)
    text_file.close()


def lee_txt(file_path):
    """
lee fichero de texto
    :param file_path:
    :return:
    """
    import os
    if os.path.isfile(file_path):
        # open text file in read mode
        text_file = open(file_path, "r", encoding='utf-8')

        # read whole file to a string
        data = text_file.read()

        # close file
        text_file.close()
        return data


def lee_sheet(path, n_row_data, pre, fecha_min=None, show=False):
    """
Lee fichero excel de datos históricos. Requiere que la primera fila se de fechas
    :param show: Boolean. Muestra preview de cabecera y datos procesados
    :param pre: prefijo para las columnas
    :param path:
    :param n_row_data: fila en que comienzan los datos
    :param fecha_min: '2016-01-01'
    :return: dataframe la información de la cabecera
    """
    df = pd.read_excel(path, skiprows=n_row_data - 1)

    df = df[~pd.isna(df.iloc[:, 0])]  # quitamos filas sin fecha
    df = df.dropna(axis=1, how='all')  # quitamos columnas en blanco
    if fecha_min is not None:
        from datetime import datetime
        df = df[df.iloc[:, 0] > datetime.strptime(fecha_min, '%Y-%m-%d')]

    df = rename_cols(df, pre, 'fecha')

    cab = pd.read_excel(path, nrows=n_row_data - 2)
    if show:
        from IPython.core.display import display
        print('  ** Cabecera')
        display(cab)
        print('  ** Datos')
        display(df)

    return df, cab


def rename_cols(h, pre, col_fecha='fecha'):
    cols = h.columns
    h = h.rename(columns={cols[0]: col_fecha})
    h = h.rename(columns={x: pre + '_' + x for x in (cols[1:])})

    return h
