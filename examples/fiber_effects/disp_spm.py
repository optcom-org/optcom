from optcom import *

domain = Domain(samples_per_bit=1024, bit_width=40.0)

lt = Layout(domain)
power = 5.0
width = 1.0
pulse = Gaussian(channels=1, peak_power=[power], width=[width],
                 center_lambda=[1550.0])
beta_2 = 20.0
beta = [[.0, .0, beta_2], [.0, .0, -beta_2]]
steps = int(1e3)

#N^2 = L_D / L_{NL} where L_{NL} = 1 / (\gamma * power)
L_D = calc_dispersion_length(width, beta_2)
N_2 = 3.0
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
        x_datas.append(pulse.fields[0].time)
        y_datas.append(temporal_power(pulse.fields[0].channels))
        x_datas.append(pulse.fields[0].nu)
        y_datas.append(spectral_power(pulse.fields[0].channels))
    x_datas.append(fiber.fields[1].time)
    y_datas.append(temporal_power(fiber.fields[1].channels))
    x_datas.append(fiber.fields[1].nu)
    y_datas.append(spectral_power(fiber.fields[1].channels))


x_labels = ['t', 'nu']
y_labels = ['P_t', 'P_nu']
plot_titles = ["Effect of dispersion and self-phase modulation on Gaussian "
               "pulse - Temporal power", "Effect of dispersion and self-phase "
               "modulation on Gaussian pulse - Spectral power"]

plot_groups = [0,1,0,1,0,1]
plot_labels = ['original pulse', 'original pulse']
plot_labels.extend([r'w/ $\beta_2 > 0$', r'w/ $\beta_2 > 0$'])
plot_labels.extend([r'w/ $\beta_2 < 0$', r'w/ $\beta_2 < 0$'])


plot(x_datas, y_datas, x_labels=x_labels, y_labels=y_labels,
     plot_titles=plot_titles, plot_groups=plot_groups,
     plot_labels=plot_labels, opacity=0.1)
