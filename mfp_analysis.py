import matplotlib.pyplot as plt

from main5 import *
import math


def vhs_mfp(n, sigma, omega):
    return 1 / (n * sigma * 2 ** (1/2))  # / (2 - (omega - 1/2)) ** (omega - 1/2) * math.gamma(2 - (omega - 1/2)))


def vhs_sigma(temp, omega):
    d0 = 2.33 * 10 ** -10
    temp0 = 273
    return np.pi * d0 ** 2 * (temp0 / temp) ** (omega - 1/2)


def mach_num(temp, speed):
    gamma = 5/3
    return speed / np.sqrt(gamma * kB * temp / mHe)


def density_power_law(x, n_init):
    return 0.154 * n_init * (dN / x) ** 2


def temp_power_law(x, T_init):
    return 0.287 * T_init * (dN / x) ** (4/3)


def speed_ratio_power_law(x):
    return np.sqrt(5 * np.pi / (8 * 0.287)) * (x / dN) ** (2/3)


def speed_ratio_temp(T, T_init):
    return np.sqrt(5 * np.pi / 8) * (T / T_init) ** (-1/2)


def speed_mach(mach, T_init):
    gamma = 5/3
    return mach * np.sqrt(gamma * kB * T_init / mHe) * (1 + (gamma - 1) / 2 * mach ** 2) ** (-1/2)


def speed_terminal(x, T_init):
    return x + np.sqrt(5 * kB * T_init / mHe) - x


def density_mach(mach, n_init):
    gamma = 5/3
    return n_init * (1 + (gamma - 1) / 2 * mach ** 2) ** (-1 / (gamma - 1))


def temp_mach(mach, T_init):
    gamma = 5/3
    return T_init * (1 + (gamma - 1) / 2 * mach ** 2) ** (-1)


def x_mach(mach):
    gamma = 5/3
    A_star = np.pi * dN ** 2
    return dN * np.sqrt(1 / (2 * mach)) * (2 / (gamma + 1) * (1 + (gamma - 1) / 2 * mach ** 2)) ** (
        (gamma + 1) / (4 * (gamma - 1)))


def ind_var_power_law(x, mfp):
    cT = 0.287
    const = 2 * np.sqrt(5 * np.pi / (8 * cT))
    return const * mfp / (dN ** (2/3) * x ** (1/3))


def ind_var_mixed(x, mfp, mach):
    gamma = 5/3
    const = 2 * np.sqrt(gamma * np.pi / 8)
    return const * mfp * mach / x


def analysis(temp, n):
    x0, t_perp0, t_parallel0, t_ratio0, mfp0, mct0, speed0, mfp_lab0, T, n0, mach0 = t_ratio_vs_mfp(temp, n)
    x0 = x0 * 0.0049
    omega = 0.66
    plt.title(f'{temp}, {n}')

    plt.plot(x0, mfp0, label='DSMC mean free path')
    plt.plot(x0, vhs_mfp(n0, vhs_sigma(T, omega), omega), label='analytic mean free path')
    plt.ylabel('mean free path (m)')

    # plt.plot(x0, T, label='DMSC T')
    # plt.plot(x0, temp_power_law(x0, float(temp)), label='power law T')
    # plt.plot(x0, temp_mach(mach0, float(temp)), label='miller equation T')
    # plt.ylabel('temperature (K)')

    # plt.plot(x0, n0, label='DSMC n')
    # plt.plot(x0, density_power_law(x0, float(n[:6])), label='power law n')
    # plt.plot(x0, density_mach(mach0, float(n[:6])), label='miller equation n')
    # plt.ylabel('number density')

    # plt.plot(x0, speed0, label='DSMC flow speed')
    # plt.plot(x0, speed_terminal(x0, float(temp)), label='terminal speed')
    # plt.plot(x0, speed_mach(mach0, float(temp)), label='miller equation speed')
    # plt.ylabel('flow speed')

    # plt.plot(x0, speed0 / np.sqrt(8 * kB * T / (np.pi * mHe)), label='DSMC speed/v_bar')
    # plt.plot(x0, speed_ratio_temp(T, float(temp)), label='in terms of temperature speed/v_bar')
    # plt.plot(x0, speed_ratio_power_law(x0), label='power law speed/v_bar')
    # plt.plot(x0, mach0 * np.sqrt(5 / 3 * np.pi / 8), label='miller equation speed/v_bar')
    # plt.ylabel('speed/v_bar')

    # plt.plot(x0, x0, label='position x')
    # plt.plot(x0, x_mach(mach0), label='position as a function of mach')
    # plt.ylabel('position')

    # plt.plot(x0, t_ratio0)

    # plt.plot(x0, ind_var_power_law(x0, mfp0), label='P power law')
    # plt.plot(x0, ind_var_mixed(x0, mfp0, mach0), label='P mixed')

    # plt.plot(ind_var_mixed(x0, mfp0, mach0), x_mach(mach0) / x0, label=f'{temp}, {n}')


if __name__ == '__main__':
    t_list = ['100', '150', '200', '250', '300']
    n_list = ['1.0e25', '1.5e25', '2.0e25', '2.5e25', '3.0e25']
    n_legend = [r'$1.0$', r'$1.5$', r'$2.0$', r'$2.5$', r'$3.0$']

    color_list = ['k', 'r', 'orange', 'cyan', 'b']

    # analysis('300', '1.0e24_hard_sphere')
    # plt.legend()
    # plt.show()

    # mfp_angled = get_data('200', '2.0e25', 'mfp_angled')
    # plt.plot(mfp_angled[0] * 0.004898, mfp_angled[1], label='DSMC mfp, slightly angled off axis')
    # analysis('200', '2.0e25')
    # plt.xlabel('distance from nozzle (m)')
    # # plt.yscale('log')
    # plt.legend()
    # plt.show()

    for n, color, legend in zip(n_list, color_list, n_legend):
        analysis('200', n)
        # plt.yscale('log')
        plt.xlabel('distance from nozzle (m)')
        plt.legend()
        plt.show()

    for t, color in zip(t_list, color_list):
        analysis(t, '2.0e25')
        plt.xlabel('distance from nozzle (m)')
        plt.legend()
        plt.show()

    # plt.title('position as a function of mach divided by actual position')
    # plt.xlabel('parameter P')
    # plt.legend()
    # plt.show()


