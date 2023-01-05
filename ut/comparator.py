import pandas as pd

from ut.base import df_save, json_update_file, json_read, make_folder
from ut.io import lista_files_recursiva, get_filename

PARAMS_JSON = 'params.json'


class Comparator:
    def __init__(self, path, reset_if_exists=False):
        self.path = make_folder(path, delete_if_exists=reset_if_exists)  # ruta general
        self._i = self.get_last_index() + 1
        print(f'el nuevo índice es {self._i}')
        self.path_exp = make_folder(self.path + str(self._i).zfill(3))  # ruta del experimento en particular

    def add_item(self, params):
        print(f'este item es {str(self._i)}')
        json_update_file(self.path + PARAMS_JSON, {self._i: params})

    def save_df(self, df, name, save_index=False):
        df_save(df, self.path_exp, name, save_index=save_index, append_size=False)

    def get_dfs(self):
        files = lista_files_recursiva(self.path_exp, 'csv')
        di = {}
        for file in files:
            name = get_filename(file, True)
            di[name] = pd.read_csv(file)

        return di

    def _read_json(self):
        return json_read(self.path + PARAMS_JSON, keys_as_integer=True)

    def get_last_index(self):
        j = self._read_json()
        if j == {}:
            res = 0
        else:
            res = max(list(j.keys()))
        return res

    def get_ranking_df(self, col_metric, asc=False):
        j = self._read_json()
        df = pd.DataFrame().from_dict(j, orient='index')
        df = df.sort_values(col_metric, ascending=asc)
        return df
