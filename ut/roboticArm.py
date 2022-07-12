import random
import time
import ipysheet
import pandas as pd
from IPython.core.display import display

import lss_const
import lss
from ut.base import now, json_save, json_read, time_to_str, FORMAT_DATETIME, save_df, json_update_file
from ut.io import escribe_txt
from lss_const import d_status

SEC_MARGIN = 100  # cuantas decimas de grado se dejan como maergen para no usar el límite del servo
RANGE_BASE = 1800


class Pattern:
    def __init__(self, di, name='name', desc=''):
        self.moves = {}
        self.di = di
        self.name = name
        self.desc = desc

    def add(self, servo, start, pos, vel):
        """
agrega un tramo de movimiento
        :param servo: ['base', 'hombro', 'codo', 'muneca', 'mano']
        :param start: tiempo de partida (en secs)
        :param pos: ángulo final (en décimas de grado
        :param vel: velocidad (max= 300)
        """
        lista = ['base', 'hombro', 'codo', 'muneca', 'mano']
        if servo in lista:
            aa = {start: {'o': servo, 'pos': pos, 'vel': vel}}
            self.moves.update(aa)
        else:
            print('part debe ser uno de estos {}')

    def get_dic_moves(self, random_perc=0):
        d_moves = varia_pattern(self.moves, random_perc)
        return d_moves

    def get_df_moves(self, random_perc=0):
        dic_moves = self.get_dic_moves(random_perc)
        df_move = pd.DataFrame.from_dict(dic_moves, orient='index').reset_index().sort_values('index')
        df_move = df_move.rename(columns={'index': 'time'})

        return df_move.copy().reset_index(drop=True)

    def get_editable_df(self):
        sheet = ipysheet.from_dataframe(self.get_df_moves())
        return sheet

    def set_sheet_moves(self, sheet, test_it=False):
        df = ipysheet.to_dataframe(sheet)
        dic = df.set_index('time').to_dict(orient='index')
        self.moves = dic
        if test_it:
            self.run(start_home=False)

    def stop(self, servo):
        print('poniendo todos en HOLD a causa de ', servo)
        for k in self.di:
            o = self.di[k]['o']
            o.hold()

    def _run(self, start_home=True, test_mode=False, silent=False, base_shift=0, random_perc=0):
        if start_home:
            home(self.di, shifted_base=base_shift)

        delta = 0  # tiempo antes del siguiente movimiento

        # APLICAMOS EL RANDOM
        df_moves = self.get_df_moves(random_perc)

        # APLICAMOS EL SHIFT
        if base_shift != 0:
            df_moves = apply_shift(df_moves, base_shift)

        # self.t = threading.Thread(target=monitoriza_servos, args=(self.di, self.stop))
        # self.t.start()
        n_moves = len(df_moves)

        for i in range(n_moves):
            row = df_moves.iloc[i, :]
            vel = row.vel
            servo = row.o
            x = row.pos
            if not silent:
                print('\n********** {} | {} (->{} vel:{}) | {}'.format(i, servo, row.pos, vel, now()))

            if test_mode:
                vel = 30
                delta = 1.5
            o = self.di[servo]['o']

            # MOVE
            o.moveTo(x, vel)

            # wait sleeping to start new instruction
            if i < (n_moves - 1) and not test_mode:
                r_next = df_moves.iloc[i + 1, :]
                delta = float(r_next['time']) - float(row['time'])

            if delta > 0:
                if not silent:
                    print('>>>Waiting ', round(delta, 2))
                time.sleep(delta)
                if not silent:
                    print('>>>Waited ', round(delta, 2))

            delta = 0

        # espera a estar quieto para siguiente instrucción
        # while is_moving(self.di):
        #     time.sleep(0.1)
        time.sleep(1)  # el is_moving depende de que respondan los servos y esto puede ocasionar tiempos muy variables

        if not silent:
            print('ya está quieto; ha terminado')

        executed_mov = {'shift_base':   base_shift,
                        'name':         self.name,
                        'random_perc':  random_perc,
                        'base_pattern': self.get_df_moves().to_dict('index'),
                        'real_pattern': df_moves.to_dict('index')}

        if not silent:
            display(executed_mov)

        return executed_mov

    def run(self, n=1, start_home=True, end_home=True, intercala_home=True, silent=False, random_perc=0,
            base_shift=0, test_mode=False):
        """

        :param test_mode:
        :param n:
        :param start_home:
        :param end_home:
        :param intercala_home:
        :param silent:
        :param random_perc: incorpora un porcentaje de aletoreidad al movimiento
        :param base_shift: los movimientos de la base están todos desplazados en esta cantidad
        """
        d = {}
        if start_home:
            home(self.di, shifted_base=base_shift)

        for i in range(n):
            if not silent:
                print('\n\n >>>>>>>>>>repeticion: {}/{}'.format(i + 1, n))
            d_moves = self._run(start_home=intercala_home, silent=silent, base_shift=base_shift,
                                random_perc=random_perc, test_mode=test_mode)
            d.update(d_moves)

        if end_home:
            home(self.di, shifted_base=base_shift, reset_if_error=False)

        return d

    def create_random(self, n_moves=4, t_max=4):
        """
creación de movimientos random
        :param n_moves:
        :param t_max:
        """
        import random
        di = self.di
        partes = random.choices(list(di.keys()), k=n_moves)
        move = {}

        for k in partes:
            pos = random.randint(di[k]['min'], di[k]['max'])
            vel = random.randint(30, 300)
            d = {round(random.random() * t_max, 2): {'o': k, 'pos': pos, 'vel': vel}}
            move.update(d)

        self.moves = move
        display(self.get_df_moves())

    def save(self, path='data_in/patrones/'):
        json_save(self.moves, path + 'move_' + self.name)

        # save description
        path2 = path + 'descriptions'
        json_update_file(path2, {self.name: self.desc})

    def load(self, path, verbatim=False):
        self.name = path.split('/')[-1].split('.')[0].split('_')[-1]
        self.moves = json_read(path)
        if verbatim:
            print('Movimiento llamado: {}'.format(self.name))
            display(self.get_df_moves())

    def set_moves(self, di_moves):
        self.moves = di_moves


class Experimento:
    def __init__(self, di, n, *files):
        self._reset_vars()

        self.di = di
        self.files = files
        self.base_shift = 0
        self.random_perc = 0
        self.n = n
        self._l_patterns = self._random_seq_of_pats(n)
        self.range_shifted = 0

        print(self.get_sequence())

    def _random_seq_of_pats(self, n):
        return random.choices(patterns_from_files(self.di, self.files), k=n)

    def get_sequence(self):
        return [x.name for x in self._l_patterns]

    def get_patterns(self):
        return self._l_patterns

    def get_patterns_moves_df(self):
        li = []
        for pat in self._l_patterns:
            moves = pat.get_df_moves()
            li.append(moves)
            display(moves)
        return li

    def get_patterns_used(self):
        return sorted(list(set(self.get_sequence())))

    def _reset_vars(self):
        self.df_moves_done = pd.DataFrame()
        self.di_moves_done = {}
        self.counter = 1

    def set_sequence(self, seq):
        moves = patterns_from_files(self.di, self.files)
        ll = []
        for s in seq:
            moves_red = [m for m in moves if m.name == s]
            if len(moves_red) != 1:
                print('problemas para enocontar el mov {}'.format(s))
            else:
                ll.append(moves_red[0])
        self._l_patterns = ll
        print('secuencia aceptada:', self.get_sequence())

    def set_shift(self, shift):
        self.base_shift = shift

    def set_random_perc(self, perc):
        self.random_perc = perc

    def regenerate_sequence(self, n):
        """
regenera la secuencia de movimientos, de largo n, usando los mismos patrones que definieron
el Experimento
        :param n:
        """
        pats = self._random_seq_of_pats(n)
        self._l_patterns = pats
        seq = self.get_sequence()
        print('nueva:', seq)

    def run(self, silent=True, test_mode=False, range_shifted=0):
        self._reset_vars()
        self.range_shifted = range_shifted
        bs = self.base_shift
        rp = self.random_perc

        for pat in self._l_patterns:
            c = self.counter
            t1 = now(True)

            # random shift
            if range_shifted != 0:
                df_moves = pat.get_df_moves()
                maxi_applicable, mini_applicable, _, _ = range_of_base(df_moves)
                bs = int(random.uniform(-range_shifted, range_shifted))

            if c == 1:
                home(self.di, shifted_base=bs)

            print('\n ', c, ' / ', self.n, ' doing ', pat.name,
                  ' | rand: ', rp, '% | shift_base: ', str(round(bs / 10, 2)),
                  'º | ', t1)

            d_move = pat.run(1, start_home=False, end_home=False, intercala_home=False, silent=silent,
                             random_perc=rp, base_shift=bs, test_mode=test_mode)
            d_move['start'] = time_to_str(t1, FORMAT_DATETIME)
            self.di_moves_done[c] = d_move

            #  Go Home
            time.sleep(1)
            t2 = now(True)
            home(self.di, shifted_base=bs, reset_if_error=False)
            df2 = pd.DataFrame({'time': [t1, t2], 'pat': [pat.name, 'GH'], 'i': [c, c], 'shift': [bs, bs],
                                'rand': [rp, rp]})

            self.df_moves_done = pd.concat([self.df_moves_done, df2])
            self.counter = c + 1

        # home(self.di)

    def save(self, name, desc, path):
        f = '%Y%m%d_%H%M%S'
        tx, n_moves = self.describe(desc, imprime=False)

        ini = self.df_moves_done.time.dt.strftime(f).iloc[0]
        end = self.df_moves_done.time.dt.strftime(f).iloc[-1]
        name2 = ini + '__' + end + '_' + name + '_n' + n_moves

        # df con los movimientos realizados (time - nombre)
        save_df(self.df_moves_done, path, name2, append_size=False)

        # descripción del experimento
        escribe_txt(tx, path + name2 + '.txt')

        # movimientos reales realizados (considerando la variación aleatoria y shift de la base
        json_save(dic=self.di_moves_done, path=path + name2 + '_real')

    def describe(self, desc='', imprime=True):
        moves_ = self.get_sequence()
        n_moves = str(len(moves_))
        tx = desc + \
             '\n\n' + 'n_moves: ' + n_moves + \
             '\n\n' + 'random_perc: ' + str(self.random_perc) + \
             '\n\n' + str(moves_) + \
             '\n\n' + 'range_base_shift (deg*10): ' + str(self.range_shifted)
        if imprime:
            print(tx)
            return None
        return tx, n_moves

    def apply_to_each_pat(self, fun):
        for i in range(len(self._l_patterns)):
            self._l_patterns[i] = fun(self._l_patterns[i])
        # x=[fun(x) for x in self._l_patterns]
        #  =x

    def set_list_patterns(self, list_patt):
        self._l_patterns = list_patt
        self.n = len(list_patt)


def get_status(servo, name, imprime=True):
    pos = servo.getPosition()
    rpm = servo.getSpeedRPM()
    curr = servo.getCurrent()
    volt = servo.getVoltage()
    temp = servo.getTemperature()

    if imprime:
        print("\nQuerying LSS... ", name)
        print("\r\n---- %s ----" % name)
        print("Position  (1/10 deg) = " + str(pos))
        print("Speed          (rpm) = " + str(rpm))
        print("Curent          (mA) = " + str(curr))
        print("Voltage         (mV) = " + str(volt))
        print("Temperature (1/10 C) = " + str(temp))

    df = pd.DataFrame({'name': [name], 'pos': [pos], 'rpm': [rpm],
                       'curr': [curr], 'volt': [volt], 'temp': [temp]}).set_index('name')
    dic = df.to_dict(orient='index')

    return df, dic


def get_stiff(servo, name):
    stf_hol = servo.getAngularHoldingStiffness()
    stf = servo.getAngularStiffness()

    df = pd.DataFrame({'name': [name], 'stiff': [stf], 'stf_hol': [stf_hol]}).set_index('name')
    dic = df.to_dict(orient='index')

    return df, dic


def update_position(di):
    pose = []
    for k in di:
        #         print(k)
        o = di[k]['o']
        try:
            pos = int(o.getPosition())
        except Exception as e:
            pos = -999
        di[k]['pos'] = pos
        pose.append(pos)

    print('Position (angles):', pose)


def home(di, reset_if_error=True, shifted_base=0):
    """
lleva a la posición de origen
    :param di:
    :param reset_if_error:
    :param shifted_base: ángulo por el cual dejaremos la base rotada. (900 = 90 grados clockwise)
    :return:
    """
    if is_at_home(di) and shifted_base == 0:
        print('ya está en casa')
        return
    else:
        print('\nGoing Home')

    for k in di:
        o = di[k]['o']
        status = o.getStatus()
        if status not in ['1', '6', '3', '4']:
            if status is None:
                kk = 'None'
            else:
                kk = d_status[int(status)]

            msg = '** WARNING status of <<{}>> is not normal, is {}:{}. Reseteamos el servo: {}'.format(k, status, kk,
                                                                                                        reset_if_error)
            print(msg)
            if reset_if_error:
                o.reset()
            # raise Exception(msg)
        if k == 'base':
            o.moveTo(shifted_base, vel=40)
            print('moviendo a home con base shiftada: ', str(round(shifted_base / 10, 1)), ' grados')
        else:
            o.moveTo(0, vel=40)  # si no limintamos la velocidad, a veces es muy brusco y
            # afloja los tornillos del brazo
    while is_moving(di):
        time.sleep(0.2)  # para garantizar que se detiene
    update_position(di)


def resetea_all(di):
    print('***+CUIDADO que el brazo se CAERÁ')
    time.sleep(2)
    for k in di:
        di[k]['o'].reset()


def monitoriza_servos(di, cb):
    l_codo = di['codo']['o']
    l_hombro = di['hombro']['o']

    n_not_moving = 0
    for i in range(1500):
        time.sleep(0.1)
        # get_speed = l_codo.getSpeed()
        # print(get_speed)
        # try:
        #     speed = int(get_speed)
        # except Exception as e:
        #     print(e)
        #     speed = -1

        txt = '>>{}  medida:{}, cu:{} , sp:{}, vol:{}'
        try:
            cc = int(l_codo.getCurrent())
            cv = int(l_codo.getSpeed())
            hc = int(l_hombro.getCurrent())
            hv = int(l_hombro.getSpeed())
            print(txt.format(i, 'codo', cc, cv, l_codo.getVoltage()))
            print(txt.format(i, 'hombro', hc, hv, l_hombro.getVoltage()))

            if (abs(cv) < 10) and cc > 1000:
                print('salimos CODO ')
                cb()
                break
            if (abs(hv) < 10) and hc > 1000:
                print('salimos HOMBRO ')
                cb()
                break

            if is_moving(di):
                n_not_moving = 0
            else:
                n_not_moving = n_not_moving + 1
                if n_not_moving > 4:
                    print('stp, ya no se mueve, detenemos la monitorización')
                    break
        except Exception as e:
            print(e)


def is_moving(di):
    li = []
    for k in di:
        o = di[k]['o']
        try:
            v = int(o.getSpeed())
        except Exception as e:
            print('No se puede medir velocidad de {}', k)
            v = 0
        # print(k, ' vel', v)
        li.append(abs(v))

    is_moving_ = sum(li) > 5
    # print('is_moving ', is_moving_)
    return is_moving_


def is_at_home(di):
    li = []
    try:
        for k in di:
            o = di[k]['o']
            p = int(o.getPosition())
            li.append(abs(p))
        at_home = sum(li) < 15
    except Exception as e:
        print('no puedo medir posicion ', e)
        at_home = False

    return at_home


def get_variables(di):
    status = pd.DataFrame()
    for k in di:
        df, j = get_status(di[k]['o'], k, False)
        status = pd.concat([status, df])
        di[k]['status'] = j[k]

    return status


def get_variables_st(di):
    """
https://wiki.lynxmotion.com/info/wiki/lynxmotion/view/lynxmotion-smart-servo/lss-communication-protocol/#HAngularHoldingStiffness28AH29
    :param di:
    :return:
    """
    status = pd.DataFrame()
    for k in di:
        df, j = get_stiff(di[k]['o'], k)
        status = pd.concat([status, df])
        di[k]['stiff'] = j[k]

    return status


def init(CST_LSS_Port="COM5", go_home=True):
    # Use the app LSS Flowarm that makes automatic scanning
    CST_LSS_Baud = lss_const.LSS_DefaultBaud
    lss.initBus(CST_LSS_Port, CST_LSS_Baud)

    print('Asignando las variables de servos')
    l_base = lss.LSS(1)
    l_hombro = lss.LSS(2)
    l_codo = lss.LSS(3)
    l_muneca = lss.LSS(4)
    l_mano = lss.LSS(5)

    print('Encencidendo las luces')
    l_base.setColorLED(lss_const.LSS_LED_Red)
    l_hombro.setColorLED(lss_const.LSS_LED_Blue)
    l_codo.setColorLED(lss_const.LSS_LED_Green)
    l_muneca.setColorLED(lss_const.LSS_LED_White)
    l_mano.setColorLED(lss_const.LSS_LED_Cyan)

    di = {'base':   {'o': l_base},
          'hombro': {'o': l_hombro},
          'codo':   {'o': l_codo},
          'muneca': {'o': l_muneca},
          'mano':   {'o': l_mano},
          }

    # fijamos los límites de movimiento
    for k in di:
        di[k]['min'] = -900
        di[k]['max'] = 900

    di['mano']['max'] = 0
    di['base']['min'] = -1800
    di['base']['max'] = 1800

    if go_home:
        home(di)
    return di, l_base, l_hombro, l_codo, l_muneca, l_mano


def pattern_from_file(di, file):
    a = Pattern(di)
    a.load(file)
    return a


def patterns_from_files(di, files):
    return [pattern_from_file(di, file) for file in files]


def varia(x, p, n=0, mini=None, maxi=None):
    """
varia el valor de x en un porcentaje p, con n decimales. mini y maxi son valores límites de saturación
    :param x:
    :param p:
    :param n:
    :param mini:
    :param maxi:
    :return:
    """
    if p == 0:
        r = x
    else:
        lim = x * p / 100
        # print(lim)
        r = round(x + lim * random.uniform(-1, 1), n)
        if mini is not None and r < mini:
            r = mini
        if maxi is not None and r > maxi:
            r = maxi

    return r


def varia_pattern(d_mov, p, di=None):
    if p == 0:
        d = d_mov
    else:
        d = {}
        for k in d_mov:
            t = str(varia(float(k), p, 2))
            dd = d_mov[k].copy()
            # limitamos por si hubiera limites (di)
            if di is not None:
                mini, maxi = di[k]['min'], di[k]['max']
            else:
                mini, maxi = None, None
            dd['pos'] = int(varia(dd['pos'], p, mini=mini, maxi=maxi))
            dd['vel'] = int(varia(dd['vel'], p, mini=30))
            d[t] = dd
    return d


def varia_pattern2(d_mov, p, di=None):
    if p == 0:
        d = d_mov
    else:
        d = {}
        for k in d_mov:
            t = str(varia(float(k), p, 2))
            dd = d_mov[k].copy()
            # limitamos por si hubiera limites (di)
            if di is not None:
                mini, maxi = di[k]['min'], di[k]['max']
            else:
                mini, maxi = None, None
            dd['pos'] = int(varia(dd['pos'], p, mini=mini, maxi=maxi))
            dd['vel'] = int(varia(dd['vel'], p, mini=30))
            d[t] = dd
    return d


def test_move(move, files, di):
    p = move_from_files(di, files, move)
    p.run(end_home=False)
    return p


def move_from_files(di, files, move):
    pats = patterns_from_files(di, files)
    return [x for x in pats if x.name == move][0]


def test_all_moves(files, di):
    pats = patterns_from_files(di, sorted(files))
    for mo in pats:
        print('\n\n>>>>>>>>>>>>>>>>>>> ', mo.name)
        mo.run(start_home=True, end_home=True, silent=True)


def home_definition(di, setType):
    """
usar la posición actual como configuración home (utilizar por ejemplo la app para ponerlo centrado)
https://wiki.lynxmotion.com/info/wiki/lynxmotion/view/lynxmotion-smart-servo/lss-communication-protocol/#HOriginOffset28O29
setType = LSS_SetConfig  # para siempre
setType = LSS_SetSession  # para la sesión

    :param di:
    :param setType:LSS_SetConfig  # para siempre, LSS_SetSession  # para la sesión
    """
    print('Ojo que se se aplica dos veces se vuelve a la posión central de fábrica, que este caso hace petar la tenaza')
    for k in di:
        o = di[k]['o']
        current_pos = o.getPosition()

        o.setOriginOffset(current_pos, setType)
        current_pos2 = o.getPosition()
        print(k, ' ', current_pos, ' -> ', current_pos2)


def apply_shift(df_moves, shift):
    """
Modifica el df_moves de manerea que los movimientos de la base están shifteados por shift
Contiene una seguridad para que no se salga se la escala
    :param df_moves:
    :param shift:
    :return:
    """
    maxi_applicable, mini_applicable, buf, mask = range_of_base(df_moves)

    msg = 'Demasiado shift, se saldría de la escala con este movimiento. ' \
          'Aplicaremos el máximo shift%s para este caso:'

    if shift > maxi_applicable:
        maxi = maxi_applicable - SEC_MARGIN
        print(msg % ' POSITIVO', maxi)
        shift = maxi

    if shift < mini_applicable:
        mini = mini_applicable + SEC_MARGIN
        print(msg % ' NEGATIVO', mini)
        shift = mini

    if buf is not None:
        df_moves.loc[mask, 'pos'] = buf['pos'].map(lambda x: x + shift)

    return df_moves


def range_of_base(df_moves):
    """
devuelve el rango de los máximo y mínimos shifts que se le pueden hacer a la base en este patrón
    :param df_moves:
    :return:
    """
    mask = df_moves['o'] == 'base'
    buf = df_moves[mask].copy()

    if len(buf) == 0:
        maxi_applicable = RANGE_BASE - SEC_MARGIN
        mini_applicable = -RANGE_BASE - SEC_MARGIN
        buf = None
    else:
        maxi_applicable = RANGE_BASE - max(buf.pos)
        mini_applicable = -RANGE_BASE - min(buf.pos)

    return maxi_applicable, mini_applicable, buf, mask
