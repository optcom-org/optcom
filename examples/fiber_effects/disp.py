from optcom import *

domain = Domain(samples_per_bit=1024, bit_width=30.0)

lt = Layout(domain)
pulse = Gaussian(channels=1, peak_power=[10.0], width=[1.0],
                 center_lambda=[1550.0])

beta  = [[.0, 1.0, .0, .0], [.0, .0, 1.0, .0], [.0, .0, .0, 1.0]]
steps = int(1e3)

x_datas = []
y_datas = []

for i in range(len(beta)):
    fiber = Fiber(length=5.0, method="ssfm_symmetric", beta=beta[i],
                  nl_approx=True, ATT=False, DISP=True, SPM=False, SS=False,
                  RS=False, steps=steps, medium='sio2', save=True)

    lt.link((pulse[0], fiber[0]))
    lt.run(pulse)
    lt.reset()
    if (not i):
        x_datas.append(pulse.fields[0].time)
        y_datas.append(temporal_power(pulse.fields[0].channels))
        x_datas.append(pulse.fields[0].nu)
        y_datas.append(spectral_power(pulse.fields[0].channels))
        x_datas.append(pulse.fields[0].time)
        y_datas.append(phase(pulse.fields[0].channels))
    x_datas.append(fiber.fields[1].time)
    y_datas.append(temporal_power(fiber.fields[1].channels))
    x_datas.append(fiber.fields[0].nu)
    y_datas.append(spectral_power(fiber.fields[1].channels))
    x_datas.append(fiber.fields[0].time)
    y_datas.append(phase(fiber.fields[1].channels))


x_labels = ['t', 'nu', 't']
y_labels = ['P_t', 'P_nu', 'phi']
plot_titles = ["Temporal power", "Spectral power", "Phase"]
fig_title = "Effect of dispersion on Gaussian pulse"

plot_groups = [0,1,2]
plot_labels = 3 * ['original pulse']
for i in range(len(beta)):
    plot_groups.extend([0,1,2])
    for j in range(3):
        plot_labels.append(r'w/ $\beta_{}$'.format(i+1))

plot(x_datas, y_datas, x_labels=x_labels, y_labels=y_labels,
     plot_titles=plot_titles, plot_groups=plot_groups,
     plot_labels=plot_labels, fig_title=fig_title, opacity=0.1)
