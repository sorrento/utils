from ut.base import seq_len, nearest
from ut.plots import plot_hist


def to_ticked_time(x, tic):
    q = x // tic
    seq = seq_len(ini=tic * q, n=2, step=tic)
    return nearest(x, seq)


def get_tick(times, plotea=True, verbose=True):
    import numpy as np
    m = []
    for i in range(len(times) - 1):
        k = (times[i + 1] - times[i])

        mili = k.microseconds / 1e3
        m.append(mili)

    if plotea:
        plot_hist(m, 30)
    tic = int(np.median(m))
    if verbose:
        print('La mediana de tick es ', tic)

    return tic