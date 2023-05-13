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

mHe = 6.65e-27
kB = 1.380649e-23
dN = 0.0002
C = np.sqrt(5 * np.pi / 8) / 0.287 ** (3 / 4)
R = 8.3


def get_data(temp, n, datatype, plot=False):
    df0 = pd.read_csv(f'{path}\\{temp}K\\{n}\\{datatype}.txt', sep='   ', header=None, engine='python')
    x0 = df0.iloc[:, 0].to_numpy()
    y0 = df0.iloc[:, 1].to_numpy()
    if plot:
        plt.plot(x0, y0, '-', color='black', label=f'{temp}K, {n}, {datatype}')
        # plt.legend()
        # plt.show()
    return [x0, y0]


#  these are for the old power law fits - I found exponential fits worked better
# def t_ratio_fit(x, *p):
#     a, b, c = p
#     return a * x ** b + c
#
#
# def t_ratio_fit_inverse(t_ratio, params):
#     a, b, c = params
#     return ((t_ratio - c) / a) ** (1 / b)
#
#
# def t_ratio_fit_errors(x, param, covar):
#     a, b, c = param
#     a_s, b_s, c_s = np.diag(covar)
#     diag_covar = np.array([[a_s, 0, 0], [0, b_s, 0], [0, 0, c_s]])
#     errors = []
#     for zeta in x:
#         j_matrix = np.array([zeta ** b, a * np.log(zeta) * zeta ** b, 1])
#         errors.append(np.dot(np.matmul(j_matrix, covar), j_matrix))
#     return np.sqrt(np.array(errors))
#
#
# def t_ratio_fit_inverse_error(t_ratio, params, covar):
#     a, b, c = params
#     a_s, b_s, c_s = np.diag(covar)
#     diag_covar = np.array([[a_s, 0, 0], [0, b_s, 0], [0, 0, c_s]])
#     j_matrix = np.array([- (1 / b) * ((t_ratio - c) / a) ** (1 / b - 1) * (t_ratio - c) / a ** 2,
#                          - ((t_ratio - c) / a) ** (1 / b) * np.log((t_ratio - c) / a) / b ** 2,
#                          - (1 / b) * ((t_ratio - c) / a) ** (1 / b - 1) * (1 / a)])
#     return np.sqrt(np.dot(np.matmul(j_matrix, covar), j_matrix))


def t_ratio_fit(x, *p):
    a, b, c = p
    return a * np.exp(x * b) + c


def t_ratio_fit_inverse(t_ratio, params):
    a, b, c = params
    return np.log((t_ratio - c) / a) / b


def t_ratio_fit_errors(x, param, covar):
    a, b, c = param
    a_s, b_s, c_s = np.diag(covar)
    diag_covar = np.array([[a_s, 0, 0], [0, b_s, 0], [0, 0, c_s]])
    errors = []
    for zeta in x:
        j_matrix = np.array([np.exp(zeta * b), zeta * a * np.exp(zeta * b), 1])
        errors.append(np.dot(np.matmul(j_matrix, covar), j_matrix))
    return np.sqrt(np.array(errors))


def t_ratio_fit_inverse_error(t_ratio, params, covar):
    a, b, c = params
    a_s, b_s, c_s = np.diag(covar)
    diag_covar = np.array([[a_s, 0, 0], [0, b_s, 0], [0, 0, c_s]])
    j_matrix = np.array([- 1 / (a * b),
                         - np.log((t_ratio - c) / a) / b ** 2,
                         - 1 / (b * np.log(t_ratio - c))])
    return np.sqrt(np.dot(np.matmul(j_matrix, covar), j_matrix))


def plot_default(labelsize=18, legendsize=12, axis=None, fig=None, save=False):
    if axis is None:
        plt.xlabel("Mean free path ($10^{-4}$ m)", fontsize=labelsize)
        plt.ylabel(r"$T_{\perp}/T_{\parallel}$", fontsize=labelsize)
        plt.legend(fontsize=legendsize)
        plt.tick_params(axis='both', direction='in', top='true', right='true')
        plt.tight_layout()
        plt.show()
    else:
        plt.xlabel(r"$P$ (unitless)", fontsize=labelsize)
        for ax in axis:
            ax.set_ylabel(r"$T_{\perp}/T_{\parallel}$", fontsize=labelsize)
            ax.legend(fontsize=legendsize)
            ax.tick_params(axis='both', direction='in', top='true', right='true')
            ax.set_ylim(0.1, 1)
            ax.set_xlim(0.12, 1.85)
            ax.yaxis.set_label_coords(-0.12, .5)
        fig.tight_layout()
        if save:
            plt.savefig('t_ratio_vs_P.png', dpi=250)
        plt.show()


def t_ratio_vs_mfp(temp, n, plot=False, xcut=0.0, color='k'):
    x0, t_perp0 = get_data(temp, n, 'Ty')
    _, t_parallel0 = get_data(temp, n, 'Tx')
    t_ratio0 = t_perp0 / t_parallel0
    _, mfp0 = get_data(temp, n, 'mfp')
    _, mct0 = get_data(temp, n, 'mct')
    _, speed0 = get_data(temp, n, 'speed')
    _, T0 = get_data(temp, n, 'T')
    _, n0 = get_data(temp, n, 'number_density')
    _, mach0 = get_data(temp, n, 'mach_number')
    mfp_lab0 = mct0 * speed0
    i = 0
    if xcut > 0:
        i = (np.abs(x0 - xcut)).argmin()
    if plot:
        plt.plot(mfp_lab0[i:] / x0[i:], t_ratio0[i:], color=color)
    return x0[i:], t_perp0[i:], t_parallel0[i:], t_ratio0[i:], mfp0[i:], mct0[i:], speed0[i:], mfp_lab0[i:], T0[i:], \
           n0[i:], mach0[i:]


def trim_data(old_data, reference, cut, lowerbound=False):
    if lowerbound:
        return np.array([point for point, ref in zip(old_data, reference) if ref > cut])
    else:
        return np.array([point for point, ref in zip(old_data, reference) if ref < cut])


def hard_sphere_mfp(visc, T, n):
    return 16 / 5 * visc / n * 1 / (2 * np.pi * R * T) ** (1/2)


def hard_sphere_visc(T, sigma):
    return 5 / (16 * np.pi ** (1/2)) * (mHe * kB * T) ** (1/2) / sigma ** 2


def vhs_sigma(T, omega):
    d_ref = 2.33e-10
    T_ref = 273
    return np.pi * d_ref ** 2 * (T_ref / T) ** (omega - 1/2)


def vhs_mfp(omega, T, n):
    return 1 / (np.sqrt(2) * n * vhs_sigma(T, omega))


def viscosity_mfp(omega, T, n):
    sigma = vhs_sigma(T, omega)
    visc = hard_sphere_visc(T, sigma)
    return hard_sphere_mfp(visc, T, n)


def ind_var_kevin(mfp, mach):
    gamma = 5/3
    C5 = np.pi * np.sqrt(gamma) * (np.pi * dN ** 2) ** (-1 / 2) * (
            (gamma + 1) / 2) ** ((gamma + 1) / (4 * (gamma - 1)))
    term = 1 + (gamma - 1) / 2 * mach ** 2
    return C5 * mfp * mach ** (7/2) * term ** (-(gamma + 1)/(4 * (gamma - 1)) - 1) * (
            (2 / (gamma + 1) * term) ** (-1) * mach ** 2 - 1) ** (-1)


def ind_var_bird(mfp_init, mach, omega):
    gamma = 5/3
    nflow = 2
    return mfp_init / dN * 1 / 2 * (np.pi * gamma / 2) ** (1 / 2) * nflow * mach ** (3 + 1 / nflow) / (
            mach ** 2 - 1) * ((gamma + 1) / 2) ** ((gamma + 1) / (2 * nflow * (gamma - 1))) * (
            1 + (gamma - 1) / 2 * mach ** 2) ** ((gamma + 1) / (2 * (gamma - 1)) * (1 - 1 / nflow) - omega)


def ind_var_power_law(x, mfp):
    cT = 0.287
    const = 2 * np.sqrt(5 * np.pi / (8 * cT))
    return const * mfp / (dN ** (2/3) * x ** (1/3))


def ind_var_mixed(x, mfp, mach):
    gamma = 5/3
    const = 2 * np.sqrt(gamma * np.pi / 8)
    return const * mfp * mach / x


def ind_var_temp_n(mfp, T, n, T_init, n_init):
    cn = 0.154
    const = np.sqrt(5 * np.pi / 8) * cn ** (-1/2) * 2
    return const * mfp / dN * (T_init / T) ** (1/2) * (n / n_init) ** (1/2)


def ind_var_mach_n(mfp, mach, n, n_init):
    gamma = 5/3
    cn = 0.154
    const = np.sqrt(gamma * np.pi / 8) * cn ** (-1/2) * 2
    return const * mfp / dN * mach * (n / n_init) ** (1/2)


def t_ratio_vs_mfp_fit(temp, n, plot=False, t_ratio_cut=None, xcut=0.0, color='k', axis=None, legend=None,
                       t_ratio_cut_lower=None, return_all=False):
    x0, t_perp0, t_parallel0, t_ratio0, mfp0, mct0, speed0, mfp_lab0, T, n0, mach0 = t_ratio_vs_mfp(temp, n, xcut=xcut)

    if t_ratio_cut is not None:
        mach0 = trim_data(mach0, t_ratio0, t_ratio_cut)
        T = trim_data(T, t_ratio0, t_ratio_cut)
        n0 = trim_data(n0, t_ratio0, t_ratio_cut)
        x0 = trim_data(x0, t_ratio0, t_ratio_cut)
        t_perp0 = trim_data(t_perp0, t_ratio0, t_ratio_cut)
        t_ratio0 = trim_data(t_ratio0, t_ratio0, t_ratio_cut)
        mfp0 = trim_data(mfp0, t_ratio0, t_ratio_cut)

    if t_ratio_cut_lower is not None:
        mach0 = trim_data(mach0, t_ratio0, t_ratio_cut_lower, lowerbound=True)
        T = trim_data(T, t_ratio0, t_ratio_cut_lower, lowerbound=True)
        n0 = trim_data(n0, t_ratio0, t_ratio_cut_lower, lowerbound=True)
        x0 = trim_data(x0, t_ratio0, t_ratio_cut_lower, lowerbound=True)
        t_perp0 = trim_data(t_perp0, t_ratio0, t_ratio_cut_lower, lowerbound=True)
        t_ratio0 = trim_data(t_ratio0, t_ratio0, t_ratio_cut_lower, lowerbound=True)
        mfp0 = trim_data(mfp0, t_ratio0, t_ratio_cut_lower, lowerbound=True)
    x0 = x0 * 0.0049 + 0.0001

    gamma = 5 / 3  # 5/3 for monatomic gasses
    omega = 0.66  # 0.5 for hard sphere, 0.66 for vhs helium, 1 for maxwell molecule

    T_init = float(temp)
    n_init = float(n[:6])

    mfp0 = vhs_mfp(omega, t_perp0, n0)
    ind_var = ind_var_mixed(x0, mfp0, mach0)

    # ind_var = ind_var_bird(get_data(temp, n, 'mfp')[1][0], mach0, 0.66)

    dep_var = t_ratio0
    p_guess = [0.5, -0.6, 2]
    mfp_list = np.linspace(min(ind_var), max(ind_var), 200)
    # plt.plot(ind_var, dep_var, '.', ms=2.5, color=color)
    # plt.plot(mfp_list, t_ratio_fit(mfp_list, *p_guess), '-', color=color)
    # plt.show()

    # mfp_fitted0 = mfp_fitted0 / np.sqrt(9 + x0 ** 2)
    coeff0, var_matrix0 = curve_fit(t_ratio_fit, ind_var, dep_var, p0=p_guess)
    # print(var_matrix0)
    # print(np.diag(var_matrix0))
    # print(np.sqrt(np.diag(var_matrix0)))
    a_val, b_val, c_val = coeff0
    a_err, b_err, c_err = np.sqrt(np.diag(var_matrix0))
    a0, b0, c0 = ufloat(a_val, a_err), ufloat(b_val, b_err), ufloat(c_val, c_err)
    t_ratio_fitted0 = t_ratio_fit(mfp_list, a_val, b_val, c_val)
    t_ratio_error0 = t_ratio_fit_errors(mfp_list, coeff0, var_matrix0)
    # direct_errors = t_ratio_fit_errors(mfp_list, coeff0, np.sqrt(np.diag(var_matrix0)))
    # print('Coefficients', coeff0, np.sqrt(np.diag(var_matrix0)))
    print('Coefficients with uncertainty', a0, b0, c0)

    t_ratio_section = 0.5

    intercept = t_ratio_fit_inverse(t_ratio_section, coeff0)
    intercept_error = t_ratio_fit_inverse_error(t_ratio_section, coeff0, var_matrix0)
    if legend is None:
        legend = f'{temp}K, {n}, fit'
    if plot:
        t_ratio_fitted_vals = t_ratio_fitted0
        t_ratio_fitted_err = t_ratio_error0
        if axis is None:
            plt.errorbar(mfp_list, t_ratio_fitted_vals, color=color, label=legend, yerr=t_ratio_fitted_err)
            plt.plot(ind_var, dep_var, '.', ms=5, color=color)
            # plt.plot(T0, dep_var, '.', ms=5, color='r')
        else:
            axis.plot(mfp_list, t_ratio_fitted_vals, '-', color=color, label=legend)
            # axis.errorbar(mfp_list, t_ratio_fitted_vals, color=color, label=legend, yerr=t_ratio_fitted_err)
            axis.plot(ind_var, dep_var, '.', ms=2.5, color=color)
    if return_all:
        return ind_var, x0, t_ratio0, mfp0, mct0, speed0, mfp_lab0, T, ufloat(intercept, intercept_error)
    else:
        return ufloat(intercept, intercept_error)  # t_ratio_fit_inverse(0.5, a0, b0, c0)


if __name__ == '__main__':
    print(C)
    t_list = ['100', '150', '200', '250', '300']
    n_list = ['1.0e25', '1.5e25', '2.0e25', '2.5e25', '3.0e25']
    color_list = ['k', 'r', 'orange', 'cyan', 'b']


    # t_ratio_vs_mfp('200', '2e25', plot=True)
    # plt.show()

    data_200K_cut = []
    for n, color in zip(n_list, color_list):
        data_200K_cut.append(t_ratio_vs_mfp_fit('200', n, plot=True, xcut=0.1, t_ratio_cut=0.8, t_ratio_cut_lower=0.25,
                                                color=color))
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
