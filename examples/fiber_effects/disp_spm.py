from optcom import *

domain = Domain(samples_per_bit=1024, bit_width=20.0)

lt = Layout(domain)
power = 5.0
width = 1.0
pulse = Gaussian(channels=1, peak_power=[power], width=[width],
                 center_lambda=[1550.0])
beta_2 = 20.0
beta = [[.0, .0, beta_2], [.0, .0, -beta_2]]
steps = int(1e4)

#N^2 = L_D / L_{NL} where L_{NL} = 1 / (\gamma * power)
L_D = calc_dispersion_length(width, beta_2)
N_2 = 0.03
gamma = N_2 / (L_D * power)

x_datas = []
y_datas = []

for i in range(len(beta)):
    fiber = Fiber(length=0.1, method="ssfm_symmetric", beta=beta[i],
                  gamma=gamma, nl_approx=True, ATT=False, DISP=True, SPM=True,
                  SS=False, RS=False, steps=steps, medium='sio2')

    lt.link((pulse[0], fiber[0]))
    lt.run(pulse)
    lt.reset()
    if (not i):
        x_datas.append(pulse[0][0].time)
        y_datas.append(temporal_power(pulse[0][0].channels))
        x_datas.append(pulse[0][0].nu)
        y_datas.append(spectral_power(pulse[0][0].channels))
        x_datas.append(pulse[0][0].time)
        y_datas.append(phase(pulse[0][0].channels))
    x_datas.append(fiber[1][0].time)
    y_datas.append(temporal_power(fiber[1][0].channels))
    x_datas.append(fiber[1][0].nu)
    y_datas.append(spectral_power(fiber[1][0].channels))
    x_datas.append(fiber[1][0].time)
    y_datas.append(phase(fiber[1][0].channels))


x_labels = ['t', 'nu', 't']
y_labels = ['P_t', 'P_nu', 'phi']
plot_titles = ["Temporal power", "Spectral power", "Phase"]
fig_title = "Effect of dispersion and self-phase modulation on Gaussian Pulse"

plot_groups = 3 * [0,1,2]
plot_labels = 3 * ['original pulse']
plot_labels.extend(3* [r'w/ $\beta_2 > 0$'])
plot_labels.extend(3 *[r'w/ $\beta_2 < 0$'])
x_ranges = [None, (191.5, 195.5), None]


plot2d(x_datas, y_datas, x_labels=x_labels, y_labels=y_labels,
       x_ranges=x_ranges, plot_titles=plot_titles, plot_groups=plot_groups,
       plot_labels=plot_labels, fig_title=fig_title, opacity=0.1,
       triangle_layout=True,
       filename="./examples/fiber_effects/images/disp_spm_effect_n2_0_03.png")
