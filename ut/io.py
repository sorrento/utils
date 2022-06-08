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
