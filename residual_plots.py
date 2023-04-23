import numpy as np

from main3 import *
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 1.75
mpl.rcParams['ytick.major.width'] = 1.75
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams["errorbar.capsize"] = 4


def plot_font_settings(tick_size=16):
    plt.rc('xtick', labelsize=tick_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=tick_size)  # fontsize of the tick labels


def plot_residuals(xlabel=r'${}^4{\rm {He}}$ Number density ($10^{25}$ m$^{-3}$)', labelsize=18, axis=None):
    if axis is None:
        plt.xlabel(xlabel, fontsize=labelsize)
        plt.ylabel(r"$\zeta$ (unitless)", fontsize=labelsize)
        plt.tick_params(axis='both', direction='in', top='true', right='true')
        plt.tight_layout()
        plt.show()
    else:
        axis.set_xlabel(xlabel, fontsize=labelsize)
        axis.set_ylabel(r"$\zeta$ (unitless)", fontsize=labelsize)
        axis.tick_params(axis='both', direction='in', top='true', right='true')
        # axis.set_ylim(0.068, 0.084)


plot_font_settings()


t_list = ['100', '150', '200', '250', '300']
n_list = ['1e25', '1.5e25', '2e25', '2.5e25', '3e25']
n_legend = [r'$1.0$', r'$1.5$', r'$2.0$', r'$2.5$',
            r'$3.0$']

color_list = ['k', 'r', 'orange', 'cyan', 'b']

fig, (ax1, ax2) = plt.subplots(2, sharex='all')


data_200K_cut = []
for n, color, legend in zip(n_list, color_list, n_legend):
    data_200K_cut.append(t_ratio_vs_mfp_fit('200', n, plot=True, color=color, axis=ax1, legend=legend,
                                            xcut=0.1, t_ratio_cut=0.8, t_ratio_cut_lower=0.25))

data_2e25_cut = []
for t, color in zip(t_list, color_list):
    data_2e25_cut.append(t_ratio_vs_mfp_fit(t, '2e25', plot=True, color=color, axis=ax2, legend=f'{t} K',
                                            xcut=0.1, t_ratio_cut=0.8, t_ratio_cut_lower=0.25))
ax1.yaxis.set_label_coords(-.12, .1)
ax2.yaxis.set_label_coords(-.12, .1)
plot_default(axis=(ax1, ax2), fig=fig, save=True)

mfp_50_200K_cut = []
for data in data_200K_cut:
    mfp_50_200K_cut.append(data)

mfp_50_2e25_cut = []
plt.figure(figsize=(6, 3))
for data in data_2e25_cut:
    mfp_50_2e25_cut.append(data)


plt.close()
fig, (ax1, ax2) = plt.subplots(2)
ax1.errorbar([1, 1.5, 2, 2.5, 3], unumpy.nominal_values(mfp_50_200K_cut), fmt='k.',
             yerr=unumpy.std_devs(mfp_50_200K_cut), elinewidth=1.5, ms=8)
ax1.axhline(y=unumpy.nominal_values(np.average(mfp_50_200K_cut)), color='r', linestyle='-', linewidth=1.5)
# ax1.set_ylim(2.9, 5.2)
plot_residuals(axis=ax1)


ax2.errorbar([100, 150, 200, 250, 300], unumpy.nominal_values(mfp_50_2e25_cut), fmt='k.',
             yerr=unumpy.std_devs(mfp_50_2e25_cut), elinewidth=1.5, ms=8)
ax2.axhline(y=unumpy.nominal_values(np.average(mfp_50_2e25_cut)), color='r', linestyle='-', linewidth=1.5)
# ax2.set_ylim(2.9, 4.9)
plot_residuals(xlabel=r'Nozzle temperature (K)', axis=ax2)

fig.tight_layout()
plt.savefig('residuals_zeta.png', dpi=250)
plt.show()


#  FOR TESTING THE NP.AVERAGE FUNCTION
# nom = 0
# err = 0
# for mfp in mfp_50_2e25_cut:
#     nom += unumpy.nominal_values(mfp)
#     err += (unumpy.std_devs(mfp) / 5) ** 2
# print(nom / 5, np.sqrt(err))

n_sweep_average0 = np.average(mfp_50_200K_cut)
n_sweep_average = ufloat(unumpy.nominal_values(n_sweep_average0),
                         np.sqrt(unumpy.std_devs(n_sweep_average0) ** 2 +
                         np.std(unumpy.nominal_values(mfp_50_200K_cut)) ** 2 / 5))

temp_sweep_average0 = np.average(mfp_50_2e25_cut)
temp_sweep_average = ufloat(unumpy.nominal_values(temp_sweep_average0),
                            np.sqrt(unumpy.std_devs(temp_sweep_average0) ** 2 +
                                    np.std(unumpy.nominal_values(mfp_50_2e25_cut)) ** 2 / 5))

print(n_sweep_average)
print(temp_sweep_average)



