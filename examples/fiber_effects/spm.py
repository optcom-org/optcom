from optcom import *

domain = Domain(samples_per_bit=1024, bit_width=15.0)

lt = Layout(domain)
chirp = 0.0
pulse = Gaussian(channels=1, peak_power=[5.0], width=[0.2],
                 center_lambda=[1550.0], chirp=[chirp])

gamma = 0.5
steps = int(1e4)

x_datas = []
y_datas = []
length = [0.5, 2.5, 5.0]

for i in range(len(length)):

    fiber = Fiber(length=length[i], method="ssfm_symmetric",
                  gamma=gamma, nl_approx=True, ATT=False, DISP=False,
                  SPM=True, SS=False, RS=False, steps=steps, medium='sio2',
                  save=True)

    lt.link((pulse[0], fiber[0]))
    lt.run(pulse)
    lt.reset()
    if (not i):
        x_datas.append(pulse.fields[0].nu)
        y_datas.append(spectral_power(pulse.fields[0].channels))
        x_datas.append(pulse.fields[0].time)
        y_datas.append(temporal_power(pulse.fields[0].channels))
        x_datas.append(pulse.fields[0].time)
        y_datas.append(phase(pulse.fields[0].channels))
    x_datas.append(fiber.fields[0].nu)
    y_datas.append(spectral_power(fiber.fields[1].channels))
    x_datas.append(fiber.fields[0].time)
    y_datas.append(temporal_power(fiber.fields[1].channels))
    x_datas.append(fiber.fields[0].time)
    y_datas.append(phase(fiber.fields[1].channels))

x_labels = ['nu', 't', 't']
y_labels = ['P_nu', 'P_t', 'phi']
plot_titles = ["Spectral power", "Temporal power", "Phase"]
fig_title = "Effect of self-phase modulation on Gaussian pulse"

plot_groups = [0,1,2]
plot_labels = 3 * ['original pulse']
for i in range(len(length)):
    plot_groups.extend([0,1,2])
    for j in range(3):
        plot_labels.append('w/ SPM - {} km'.format(length[i]))


plot(x_datas, y_datas, x_labels=x_labels, y_labels=y_labels,
     plot_titles=plot_titles, plot_groups=plot_groups,
     plot_labels=plot_labels, fig_title=fig_title, opacity=0.1)
