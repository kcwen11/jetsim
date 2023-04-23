import matplotlib.pyplot as plt
import numpy as np
from main import get_data2
from main3 import *

path = r'C:\Users\kcwen\Documents\Heinzen Lab\JetSim\T_n_data'
mHe = 6.6422e-27
kB = 1.380649e-23
dN = 0.0002


def cross_section(temp):
    d0 = 2.33 * 10 ** -10
    temp0 = 273
    omega = 0.66
    return np.pi * d0 ** 2 * (temp0 / temp) ** (omega - 1/2)  # / np.sqrt(2)


def continuum_temp(x1, T01):
    return 0.287 * T01 * (dN / x1) ** (4/3)


def v0(T01):
    return np.sqrt(5 * kB * T01 / mHe)


def vbar(T1):
    return np.sqrt(8 * kB * T1 / (np.pi * mHe))


def density(x1, n01):
    return 0.154 * n01 * (dN / x1) ** 2


def mfp_moving(n1, sigma1):
    return 1 / (n1 * sigma1) / np.sqrt(2)


def col_rate(n1, sigma1, vbar1):
    return n1 * sigma1 * vbar1 * np.sqrt(2)


def continuum_zeta(mfp1, T1, T01):
    return C * mfp1 / dN * (T1 / T01) ** (1/4)


def full_mfp(x1, T01, n01):
    T1 = continuum_temp(x1, T01)
    n1 = density(x1, n01)
    sigma1 = cross_section(T1)
    return mfp_moving(n1, sigma1)


def full_zeta(x1, T01, n01):
    T1 = continuum_temp(x1, T01)
    mfp1 = full_mfp(x1, T01, n01)
    return continuum_zeta(mfp1, T1, T01)


def full_col_rate(x1, T01, n01):
    T1 = continuum_temp(x1, T01)
    n1 = density(x1, n01)
    sigma1 = cross_section(T1)
    vbar1 = vbar(T1)
    return col_rate(n1, sigma1, vbar1)


if __name__ == '__main__':
    T0 = '200'
    n0 = '2e25'
    data = t_ratio_vs_mfp_fit(T0, n0, plot=False, xcut=0.1, t_ratio_cut=0.8, color='k', return_all=True)
    uncut_data = t_ratio_vs_mfp(T0, n0)
    T0_actual = uncut_data[-2][0]
    n0_actual = uncut_data[-1][0]
    sigma = cross_section(data[7])
    x = data[1] * 0.0046
    x_range = np.linspace(0.0005, 0.0046, 200)
    x_range_long = np.linspace(0.0005, 0.50, 5000)
    inverse_mfp_range_long = full_mfp(x_range_long, float(T0), float(n0)) ** (-1)
    coll_remaining = []
    for i in range(200):
        coll = np.trapz(inverse_mfp_range_long[i:], x=x_range_long[i:])
        coll_remaining.append(coll)

    zeta = data[0]

    print(T0_actual, n0_actual)
    plt.plot(x * 100, zeta, 'o', label='Zeta from DSMV')
    plt.plot(x_range * 100, full_zeta(x_range, float(T0), float(n0)), '-', label='Zeta from Continuum Equations')

    # plt.plot(x[0:-1], mct * speed[0:-1], '-', label='mct * speed')
    # plt.plot(x, (sigma * n) ** -1, '-', label=r'$(\sigma n)^{-1}$')

    plt.title("zeta vs position")
    plt.xlabel("Distance after nozzle (cm)")
    plt.ylabel("zeta (unitless)")

    plt.legend()
    plt.tick_params(axis='both', direction='in', top='true', right='true')
    plt.show()

    plt.plot(x_range * 100, np.array(coll_remaining), '-', label='Collisions Remaining')
    plt.legend()
    plt.xlabel("Distance after nozzle (cm)")
    plt.ylabel("collisions remaining")

    plt.tick_params(axis='both', direction='in', top='true', right='true')
    plt.show()

    x_range = np.linspace(0.0005, 0.50, 5000)
    x_range_long = np.linspace(0.0005, 2.0, 20000)
    T_range = continuum_temp(x_range, 4)
    inverse_mfp_range_long = full_mfp(x_range_long, 4, 2e25) ** (-1)
    coll_remaining = []
    for i in range(5000):
        coll = np.trapz(inverse_mfp_range_long[i:], x=x_range_long[i:])
        coll_remaining.append(coll)

    plt.plot(x_range * 100, full_zeta(x_range, 4, 2e25), '-', label='zeta')
    plt.plot(x_range * 100, v0(4) / x_range / full_col_rate(x_range, 4, 2e25), '-', label='also zeta')
    plt.legend()
    plt.xlabel("Distance after nozzle (cm)")
    plt.ylabel("zeta")

    plt.tick_params(axis='both', direction='in', top='true', right='true')
    plt.show()

    # plt.plot(x_range * 100, full_col_rate(x_range, 4, 2e25), '-')
    plt.legend()
    plt.xlabel("Distance after nozzle (cm)")

    plt.tick_params(axis='both', direction='in', top='true', right='true')
    plt.show()
