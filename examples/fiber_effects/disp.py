from optcom import *

domain = Domain(samples_per_bit=2048, bit_width=14.0)

lt = Layout(domain)
pulse = Gaussian(channels=1, peak_power=[5.0], width=[1.0],
                 center_lambda=[1550.0])

beta  = [[.0, 0.5, .0, .0], [.0, .0, 0.5, .0], [.0, .0, .0, 0.75]]
steps = int(1e5)

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
    y_datas.append(phase(fiber.fields[1].channels, False))


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
x_ranges = [None, (191.5, 195.5), None]

plot(x_datas, y_datas, x_labels=x_labels, y_labels=y_labels,
     x_ranges=x_ranges, plot_titles=plot_titles, plot_groups=plot_groups,
     plot_labels=plot_labels, fig_title=fig_title, opacity=0.1)
#     filename="./examples/fiber_effects/images/disp_effect.png")
