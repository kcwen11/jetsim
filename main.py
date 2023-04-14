import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from uncertainties import ufloat, unumpy
from scipy.optimize import curve_fit

mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.25
mpl.rcParams['ytick.major.width'] = 1.25
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.size'] = 5

path = r'C:\Users\kcwen\Documents\Heinzen Lab\JetSim\T_n_data'


def get_data(temp, n, slice, datatype, plot=False):
    df0 = pd.read_csv(f'{path}\\{temp}K\\{n}\\data\\{slice}\\{datatype}.txt', sep='   ', header=None, engine='python')
    x0 = df0.iloc[:, 0].to_numpy()
    y0 = df0.iloc[:, 1].to_numpy()
    if plot:
        plt.plot(x0, y0, '-', color='black')
    return [x0, y0]


def get_data2(temp, n, datatype, plot=False):
    df0 = pd.read_csv(f'{path}\\{temp}K\\{n}\\{datatype}.txt', sep='   ', header=None, engine='python')
    x0 = df0.iloc[:, 0].to_numpy()
    y0 = df0.iloc[:, 1].to_numpy()
    if plot:
        plt.plot(x0, y0, '-', color='black', label=f'{temp}K, {n}, {datatype}')
        # plt.legend()
        # plt.show()
    return [x0, y0]


def sort_mfp(mfp, var_array):
    return np.array([x for _, x in sorted(zip(mfp, var_array))])


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def mfp_fit(x, *p):
    n, a, c = p
    return a * x ** n + c


def t_ratio_fit(mfp, *p):
    a, n, mfp0 = p
    return (10 ** 3 * a * mfp + mfp0) ** (-n)


def t_ratio_fit_inverse(t_ratio, *p):
    a, n, mfp0 = p
    return 1 / (10 ** 3 * a) * (t_ratio ** (-1 / n) - mfp0)


def plot_default(labelsize=18, legendsize=12, axis=None, fig=None, save=False):
    if axis is None:
        plt.xlabel("Mean free path ($10^{-4}$ m)", fontsize=labelsize)
        plt.ylabel(r"$T_{\perp}/T_{\parallel}$", fontsize=labelsize)
        plt.legend(fontsize=legendsize)
        plt.tick_params(axis='both', direction='in', top='true', right='true')
        plt.tight_layout()
        plt.show()
    else:
        plt.xlabel(r"$\lambda_{\rm {mf}}$ ($10^{-4}$ m)", fontsize=labelsize)
        for ax in axis:
            ax.set_ylabel(r"$T_{\perp}/T_{\parallel}$", fontsize=labelsize)
            ax.legend(fontsize=legendsize)
            ax.tick_params(axis='both', direction='in', top='true', right='true')
            ax.set_xlim(0, 4)
        fig.tight_layout()
        if save:
            plt.savefig('t_ratio_vs_mfp.png', dpi=250)
        plt.show()


def t_ratio_vs_mfp(temp, n, a_guess=2 * 10 ** -5, plot=False, cut=0.0, xcut=0.0, mfp_up_cut=None):
    x0, mfp0 = get_data2(temp, n, 'mfp')
    x0 = x0 * 4.9
    _, Tx0 = get_data2(temp, n, 'Tx')
    _, Ty0 = get_data2(temp, n, 'Ty')
    p_guess = [2, a_guess, 0]
    coeff0, var_matrix0 = curve_fit(mfp_fit, x0, mfp0, p0=p_guess)
    n_val, a_val, c_val = coeff0
    n_err, a_err, c_err = np.sqrt(np.diag(var_matrix0))
    n0, a0, c0 = ufloat(n_val, n_err), ufloat(a_val, a_err), ufloat(c_val, c_err)
    mfp_fitted0 = mfp_fit(x0, n0, a0, c0)

    if plot:
        mfp_fitted_vals = unumpy.nominal_values(mfp_fitted0)
        mfp_fitted_err = unumpy.std_devs(mfp_fitted0)
        plt.errorbar(mfp_fitted_vals, Ty0 / Tx0, xerr=mfp_fitted_err, fmt='-', label=f'{temp}K, {n}')
    i = 0

    if cut > 0:
        i = (np.abs(mfp_fitted0 - cut * 10 ** (-4))).argmin()
    if xcut > 0:
        i = (np.abs(x0 - xcut)).argmin()
        # print(i)

    if mfp_up_cut is None:
        return x0[i:], mfp0[i:], mfp_fitted0[i:], Tx0[i:], Ty0[i:], coeff0
    else:
        j = (np.abs(mfp_fitted0 - mfp_up_cut * 10 ** (-4))).argmin()
        return x0[i:j], mfp0[i:j], mfp_fitted0[i:j], Tx0[i:j], Ty0[i:j], coeff0


def t_ratio_vs_mfp_fit(temp, n, a_guess=2 * 10 ** -5, plot=False, color='k', axis=None, legend=None):
    x0, mfp0, mfp_fitted0, Tx0, Ty0, coeff0 = t_ratio_vs_mfp(temp, n, a_guess=a_guess, xcut=0.6, mfp_up_cut=None)
    p_guess = [60, 0.5, 1]
    # mfp_fitted0 = mfp_fitted0 / np.sqrt(9 + x0 ** 2)
    mfp_fitted_vals = unumpy.nominal_values(mfp_fitted0)
    mfp_fitted_err = unumpy.std_devs(mfp_fitted0)
    coeff1, var_matrix0 = curve_fit(t_ratio_fit_inverse, Ty0 / Tx0, mfp_fitted_vals,
                                    sigma=mfp_fitted_err, absolute_sigma=True, p0=p_guess)
    a_val, n_val, mfp0_val = coeff1
    a_err, n_err, mfp0_err = np.sqrt(np.diag(var_matrix0))
    a0, n0, mfp1 = ufloat(a_val, a_err), ufloat(n_val, n_err), ufloat(mfp0_val, mfp0_err)
    t_ratio_fitted0 = t_ratio_fit(mfp_fitted0, a0, n0, mfp1)
    mfp_50 = (0.5 ** (-1/n0) - mfp1) / (10 ** 3 * a0)
    if plot:
        if legend is None:
            legend = f'{temp}K, {n}, fit'
        t_ratio_fitted_vals = unumpy.nominal_values(t_ratio_fitted0)
        t_ratio_fitted_err = unumpy.std_devs(t_ratio_fitted0)
        # plt.errorbar(mfp_fitted_vals, t_ratio_fitted_vals, fmt='-', yerr=t_ratio_fitted_err, label=f'{temp}K, {n}, fit')
        if axis is None:
            plt.plot(mfp_fitted_vals * 10 ** 4, t_ratio_fitted_vals, '-', color=color, label=legend)
        else:
            axis.plot(mfp_fitted_vals * 10 ** 4, t_ratio_fitted_vals, '-', color=color, label=legend)
            t_ratio = Ty0 / Tx0
            # t_ratio_sorted = sort_mfp(mfp0, t_ratio)
            axis.plot(mfp0 * 10 ** 4, t_ratio, '.', color=color, ms=2.5)
            # mfp_sorted = sort_mfp(mfp0, mfp0)
            #
            # axis.plot(moving_average(mfp_sorted * 10 ** 4, n=14), moving_average(t_ratio_sorted, n=14), '-', color=color,
            #           label=legend, markersize=2)

    return x0, mfp0, mfp_fitted0, Tx0, Ty0, coeff0, t_ratio_fitted0, coeff1, mfp_50


if __name__ == '__main__':
    t_ratio_vs_mfp('200', '2e25', plot=True)
    plot_default()

    t_list = ['100', '150', '200', '250', '300']
    n_list = ['1e25', '1.5e25', '2e25', '2.5e25', '3e25']

    data_200K_cut = []
    for n in n_list:
        data_200K_cut.append(t_ratio_vs_mfp_fit('200', n, plot=True))
    plot_default()

    mfp_50_200K_cut = []
    for data in data_200K_cut:
        mfp_50_200K_cut.append(data[-1])
    plt.errorbar([1, 1.5, 2, 2.5, 3], unumpy.nominal_values(mfp_50_200K_cut), fmt='k.',
                 yerr=unumpy.std_devs(mfp_50_200K_cut))
    plt.axhline(y=np.average(unumpy.nominal_values(mfp_50_200K_cut)), color='r', linestyle='-')
    plt.show()
    print(np.average(mfp_50_200K_cut))



    sys.exit()

    t_ratio_vs_mfp('300', '3e25', plot=True, new=True)
    t_ratio_vs_mfp('300', '2.5e25', plot=True)
    t_ratio_vs_mfp('300', '2e25', plot=True)
    # plt.title("Ty/Tx vs mfp")
    # plt.xlabel("mfp (m)")
    # plt.ylabel("Ty/Tx")
    # plt.legend()
    # plt.tick_params(axis='both', direction='in', top='true', right='true')
    # plt.show()

    t_ratio_vs_mfp('200', '4e25', plot=True)
    t_ratio_vs_mfp('200', '3e25', plot=True)
    t_ratio_vs_mfp('200', '2e25', plot=True)
    # plt.title("Ty/Tx vs mfp")
    # plt.xlabel("mfp (m)")
    # plt.ylabel("Ty/Tx")
    # plt.legend()
    # plt.tick_params(axis='both', direction='in', top='true', right='true')
    # plt.show()

    t_ratio_vs_mfp('100', '3e25', plot=True)
    t_ratio_vs_mfp('100', '2e25', plot=True)
    t_ratio_vs_mfp('100', '1e25', plot=True)
    plt.title("Ty/Tx vs mfp")
    plt.xlabel("mfp (m)")
    plt.ylabel("Ty/Tx")
    # plt.yscale('log')
    plt.legend()
    plt.tick_params(axis='both', direction='in', top='true', right='true')
    plt.show()

    x, mfp, mfp_fitted, Tx, Ty, coeff = t_ratio_vs_mfp('300', '2.5e25')

    plt.plot(x, mfp, 'o', label='Given mfp')
    plt.plot(x, mfp_fitted, '-', label='mfp fit to $x^n$')

    plt.title("mfp vs position")
    plt.xlabel("Distance after nozzle (mm)")
    plt.ylabel("Mean free path (m)")

    plt.legend()
    plt.tick_params(axis='both', direction='in', top='true', right='true')
    plt.show()

    plt.plot(x, Ty / Tx, '-', label='T perp / T parallel')

    plt.title("Ty/Tx vs position")
    plt.xlabel("distance from nozzle (mm)")
    plt.ylabel("Ty/Tx")

    plt.legend()
    plt.tick_params(axis='both', direction='in', top='true', right='true')
    plt.show()

    plt.plot(mfp_fitted, Ty / Tx, '-', label='T perp / T parallel')


