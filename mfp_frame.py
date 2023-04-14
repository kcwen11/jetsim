import matplotlib.pyplot as plt
import numpy as np
from main import get_data2

path = r'C:\Users\kcwen\Documents\Heinzen Lab\JetSim\T_n_data'


def cross_section(temp):
    d0 = 2.33 * 10 ** -10
    temp0 = 273
    omega = 0.66
    return np.pi * d0 ** 2 * (temp0 / temp) ** (2 * omega - 1)  # / np.sqrt(2)


if __name__ == '__main__':
    x, mfp = get_data2('100', '2e25', 'mfp')
    # _, mct = get_data2('300', '2e25', 'mct')
    _, speed = get_data2('100', '2e25', 'speed')
    _, n = get_data2('100', '2e25', 'number_density')
    _, T = get_data2('100', '2e25', 'Ty')

    sigma = cross_section(T)

    plt.plot(x, mfp, 'o', label='Given mfp')
    # plt.plot(x[0:-1], mct * speed[0:-1], '-', label='mct * speed')
    plt.plot(x, (sigma * n) ** -1, '-', label=r'$(\sigma n)^{-1}$')

    plt.title("mfp vs position")
    plt.xlabel("Distance after nozzle (mm)")
    plt.ylabel("Mean free path (m)")

    plt.legend()
    plt.tick_params(axis='both', direction='in', top='true', right='true')
    plt.show()

