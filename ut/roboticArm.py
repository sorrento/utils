import random
import time

import pandas as pd
from IPython.core.display import display

from ut import lss_const, lss
from ut.base import now, save_json, read_json, time_to_str, FORMAT_DATETIME, save_df
from ut.io import escribe_txt
from ut.lss_const import d_status


class Pattern:
    def __init__(self, di, name='name'):
        self.moves = {}
        self.di = di
        self.name = name

    def add(self, part, start, pos, vel):
        lista = ['base', 'hombro', 'codo', 'muneca', 'mano']
        if part in lista:
            aa = {start: {'o': part, 'pos': pos, 'vel': vel}}
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

    def stop(self, servo):
        print('poniendo todos en HOLD a causa de ', servo)
        for k in self.di:
            o = self.di[k]['o']
            o.hold()

    def _run(self, start_home=True, test_mode=False, silent=False, base_shift=0, random_perc=0):
        if start_home:
            home(self.di, shifted_base=base_shift)

        delta = 0  # tiempo antes del siguiente movimiento
        df_moves = self.get_df_moves(random_perc)
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
                time.sleep(delta)
                if not silent:
                    print('>>>Waiting ', round(delta, 2))

            delta = 0

        # espera a estar quieto para situiente instrucción
        while is_moving(self.di):
            time.sleep(0.1)

        if not silent:
            print('ya está quieto; ha terminado')

        executed_mov = {'shift_base':   base_shift,
                        'name':         self.name,
                        'random_perc':  random_perc,
                        'base_pattern': self.get_df_moves().to_dict('index'),
                        'real_pattern': df_moves.to_dict('index')}

        return executed_mov

    def run(self, n=1, start_home=True, end_home=True, intercala_home=True, silent=False, random_perc=0,
            base_shift=0):
        """

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
                                random_perc=random_perc)
            d.update(d_moves)

        if end_home:
            home(self.di, shifted_base=base_shift)

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
        save_json(self.moves, path + 'move_' + self.name)

    def load(self, path, verbatim=False):
        self.name = path.split('/')[-1].split('.')[0].split('_')[-1]
        self.moves = read_json(path)
        if verbatim:
            print('Movimiento llamado: {}'.format(self.name))
            display(self.get_df_moves())


class Experimento:
    def __init__(self, di, n, *files):
        self.di = di
        moves = patterns_from_files(di, files)
        self.r_moves = random.choices(moves, k=n)
        seq = [x.name for x in self.r_moves]
        print(seq)
        self.df = pd.DataFrame()
        self.d_moves_done = {}
        self.base_shift = 0
        self.random_perc = 0
        self.counter = 1
        self.n = n

    def set_shift(self, shift):
        self.base_shift = shift

    def set_random_perc(self, perc):
        self.random_perc = perc

    def run(self):
        home(self.di, shifted_base=self.base_shift)
        for m in self.r_moves:
            c = self.counter
            t1 = now(True)
            print('\n ', c, ' / ', self.n, ' doing ', m.name, ' | ', t1)
            d_move = m.run(1, start_home=False, end_home=False, intercala_home=False, silent=True,
                           random_perc=self.random_perc, base_shift=self.base_shift)
            d_move['start'] = time_to_str(t1, FORMAT_DATETIME)
            self.d_moves_done[c] = d_move

            #  Go Home
            time.sleep(1)
            t2 = now(True)
            home(self.di, shifted_base=self.base_shift)

            df2 = pd.DataFrame({'time': [t1, t2], 'move': [m.name, 'GH'], 'i': [c, c]})

            self.df = pd.concat([self.df, df2])
            self.counter = c + 1

        home(self.di)

    def run_shifted(self, list_shifts):
        for s in list_shifts:
            time.sleep(1)
            self.set_shift(s)
            self.run()

    def save(self, name, desc, path):
        f = '%Y%m%d_%H%M%S'
        moves_ = [x.name for x in self.r_moves]
        le = str(len(moves_))
        tx = desc + '\n\n' + 'n_moves: ' + le + '\n\n' + str(moves_)

        ini = self.df.time.dt.strftime(f).iloc[0]
        end = self.df.time.dt.strftime(f).iloc[-1]
        name2 = ini + '__' + end + '_' + name + '_n' + le

        # df con los movimientos realizados (time - nombre)
        save_df(self.df, path, name2, append_size=False)

        # descripción del experimento
        escribe_txt(tx, path + name2 + '.txt')

        # movimientos reales realizados (considerando la variación aleatoria y shift de la base
        save_json(dic=self.d_moves_done, path=path + name2 + '_real')


def get_status(myLSS, name="Telemetry", imprime=True):
    pos = myLSS.getPosition()
    rpm = myLSS.getSpeedRPM()
    curr = myLSS.getCurrent()
    volt = myLSS.getVoltage()
    temp = myLSS.getTemperature()

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

            msg = '** WARNING status of <<{}>> is not normal, is {}:{}. Reseteamos el servo '.format(k, status, kk)
            # print(msg)
            if reset_if_error:
                o.resetea_all()
            raise Exception(msg)
        if k == 'base':
            o.moveTo(shifted_base)
            print('moviendo a home con base shiftada: ', str(round(shifted_base / 10, 1)), ' grados')
        else:
            o.moveTo(0)
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

    return sum(li) > 5


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


def init(CST_LSS_Port="COM5", go_home=True):
    # Use the app LSS Flowarm that makes automatic scanning
    CST_LSS_Baud = lss_const.LSS_DefaultBaud
    lss.initBus(CST_LSS_Port, CST_LSS_Baud)

    l_base = lss.LSS(1)
    l_hombro = lss.LSS(2)
    l_codo = lss.LSS(3)
    l_muneca = lss.LSS(4)
    l_mano = lss.LSS(5)

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


def test_move(move, files, di):
    p = move_from_files(di, files, move)
    p.run(end_home=False)
    return p


def move_from_files(di, files, move):
    pats = patterns_from_files(di, files)
    return [x for x in pats if x.name == move][0]


def test_all_moves(files, di):
    pats = patterns_from_files(di, files)
    for mo in pats:
        print('\n\n>>>>>>>>>>>>>>>>>>> ', mo.name)
        mo.run(start_home=True, end_home=True)


def home_definition(di, setType):
    """
usar la posición actual como configuración home (utilizar por ejemplo la app para ponerlo centrado)
https://wiki.lynxmotion.com/info/wiki/lynxmotion/view/lynxmotion-smart-servo/lss-communication-protocol/#HOriginOffset28O29
setType = LSS_SetConfig  # para siempre
setType = LSS_SetSession  # para la sesión

    :param di:
    :param setType:
    """
    print('Ojo que se se aplica dos veces se vuelve a la posión central de fábrica, que este caso hace petar la tenaza')
    for k in di:
        o = di[k]['o']
        current_pos = o.getPosition()

        o.setOriginOffset(current_pos, setType)
        current_pos2 = o.getPosition()
        print(k, ' ', current_pos, ' -> ', current_pos2)


def shiftea(x, base_shift):
    x = x + base_shift
    if x < -1800:
        x = -1800
    if x > 1800:
        x = 1800
    return x


def apply_shift(df_moves, shift):
    mask = df_moves['o'] == 'base'
    buf = df_moves[mask].copy()
    df_moves.loc[mask, 'pos'] = buf['pos'].map(lambda x: shiftea(x, shift))

    return df_moves
