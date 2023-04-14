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


def get_data(temp, n, datatype, plot=False):
    df0 = pd.read_csv(f'{path}\\{temp}K\\{n}\\{datatype}.txt', sep='   ', header=None, engine='python')
    x0 = df0.iloc[:, 0].to_numpy()
    y0 = df0.iloc[:, 1].to_numpy()
    if plot:
        plt.plot(x0, y0, '-', color='black', label=f'{temp}K, {n}, {datatype}')
        # plt.legend()
        # plt.show()
    return [x0, y0]


def t_ratio_fit(x, *p):
    ex, a, c = p
    return a * x ** ex + c


def t_ratio_fit_errors(x, param, err):
    ex, a, c = param
    ex_err, a_err, c_err = err
    # return a_err * np.abs(x ** ex) + ex_err * np.abs(a * np.log(x) * x ** ex) + c_err
    return t_ratio_fit(x, ex + ex_err, a + a_err, c + c_err) - t_ratio_fit(x, ex - ex_err, a - a_err, c - c_err)


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
            # ax.set_xlim(0, 0.0002)
        fig.tight_layout()
        if save:
            plt.savefig('t_ratio_vs_mfp.png', dpi=250)
        plt.show()


def t_ratio_vs_mfp(temp, n, plot=False, xcut=0.0, color='k'):
    x0, t_perp0 = get_data(temp, n, 'Ty')
    _, t_parallel0 = get_data(temp, n, 'Tx')
    t_ratio0 = t_perp0 / t_parallel0
    _, mfp0 = get_data(temp, n, 'mfp')
    _, mct0 = get_data(temp, n, 'mct')
    _, speed0 = get_data(temp, n, 'speed')
    _, T0 = get_data(temp, n, 'T')
    mfp_lab0 = mct0 * speed0
    i = 0
    if xcut > 0:
        i = (np.abs(x0 - xcut)).argmin()
    if plot:
        plt.plot(mfp_lab0[i:] / x0[i:], t_ratio0[i:], color=color)
    return x0[i:], t_perp0[i:], t_parallel0[i:], t_ratio0[i:], mfp0[i:], mct0[i:], speed0[i:], mfp_lab0[i:], T0[i:]


def trim_data(old_data, reference, cut, lowerbound=False):
    if lowerbound:
        return np.array([point for point, ref in zip(old_data, reference) if ref > cut])
    else:
        return np.array([point for point, ref in zip(old_data, reference) if ref < cut])


def t_ratio_vs_mfp_fit(temp, n, plot=False, t_ratio_cut=None, xcut=0.0, color='k', axis=None, legend=None):
    x0, t_perp0, t_parallel0, t_ratio0, mfp0, mct0, speed0, mfp_lab0, T0 = t_ratio_vs_mfp(temp, n, xcut=xcut)
    if t_ratio_cut is not None:
        x0 = trim_data(x0, t_ratio0, t_ratio_cut)
        mfp0 = trim_data(mfp0, t_ratio0, t_ratio_cut)
        mct0 = trim_data(mct0, t_ratio0, t_ratio_cut)
        t_ratio0 = trim_data(t_ratio0, t_ratio0, t_ratio_cut)
        mfp_lab0 = trim_data(mfp_lab0, t_ratio0, t_ratio_cut)
        speed0 = trim_data(speed0, t_ratio0, t_ratio_cut)
        T0 = trim_data(T0, t_ratio0, t_ratio_cut)

    ind_var = mfp0 * (float(temp) / T0) ** (-1/4)
    p_guess = [-0.332, 0.021, -0.131]
    mfp_list = np.linspace(min(ind_var), max(ind_var), 200)
    # plt.plot(mfp0, t_ratio0, '.', ms=2.5, color=color)
    # plt.plot(mfp_list, t_ratio_fit(mfp_list, *p_guess), '-', color=color)
    # plt.show()

    # mfp_fitted0 = mfp_fitted0 / np.sqrt(9 + x0 ** 2)
    coeff0, var_matrix0 = curve_fit(t_ratio_fit, ind_var, t_ratio0, p0=p_guess)
    ex_val, a_val, c_val = coeff0
    ex_err, a_err, c_err = np.sqrt(np.diag(var_matrix0))
    ex0, a0, c0 = ufloat(ex_val, ex_err), ufloat(a_val, a_err), ufloat(c_val, c_err)
    t_ratio_fitted0 = t_ratio_fit(mfp_list, ex0, a0, c0)
    direct_errors = t_ratio_fit_errors(mfp_list, coeff0, np.sqrt(np.diag(var_matrix0)))
    print(coeff0)
    print(ex0, a0, c0)

    intercept = ((0.5 - c0) / a0) ** (1/ex0)
    if legend is None:
        legend = f'{temp}K, {n}, fit'
    if plot:
        t_ratio_fitted_vals = unumpy.nominal_values(t_ratio_fitted0)
        t_ratio_fitted_err = unumpy.std_devs(t_ratio_fitted0)
        if axis is None:
            plt.errorbar(mfp_list, t_ratio_fitted_vals, color=color, label=legend, yerr=t_ratio_fitted_err)
            plt.plot(ind_var, t_ratio0, '.', ms=5, color='r')
        else:
            # axis.plot(mfp_list, t_ratio_fitted_vals, '-', color=color, label=legend)
            axis.errorbar(mfp_list, t_ratio_fitted_vals, color=color, label=legend, yerr=t_ratio_fitted_err)
            axis.plot(ind_var, t_ratio0, '.', ms=2.5, color=color)

    return intercept


if __name__ == '__main__':
    t_list = ['100', '150', '200', '250', '300']
    # n_list = ['1e25', '1.5e25', '2e25', '2.5e25', '3e25']
    # color_list = ['k', 'r', 'orange', 'cyan', 'b']
    n_list = ['3e25']
    color_list = ['k']

    # t_ratio_vs_mfp('200', '2e25', plot=True)
    # plt.show()

    data_200K_cut = []
    for n, color in zip(n_list, color_list):
        data_200K_cut.append(t_ratio_vs_mfp_fit('200', n, plot=True, xcut=0.1, t_ratio_cut=0.8, color=color))
    plt.show()

    sys.exit()

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


