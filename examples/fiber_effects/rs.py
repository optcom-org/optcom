from optcom import *

domain = Domain(samples_per_bit=1024, bit_width=40.0)

lt = Layout(domain)
pulse = Gaussian(channels=1, peak_power=[5.0], width=[0.2],
                 center_lambda=[1550.0])
gamma = 0.5
steps = int(1e4)

x_datas = []
y_datas = []

fiber_init = Fiber(length=2.5, method="ssfm_symmetric",
                   gamma=gamma, nl_approx=True, ATT=False, DISP=False,
                   SPM=True, SS=False, RS=False, steps=steps,
                   medium='sio2', save=True)

fiber = Fiber(length=10.0, method="ssfm_symmetric",
                gamma=gamma, nl_approx=True, ATT=False, DISP=False,
                SPM=False, SS=False, RS=True, steps=steps,
                medium='sio2', save=True)

lt.link((pulse[0], fiber_init[0]), (fiber_init[1], fiber[0]))
lt.run(pulse)
x_datas.append(fiber_init[1][0].nu)
x_datas.append(fiber[0][0].nu)
y_datas.append(spectral_power(fiber_init[1][0].channels))
y_datas.append(spectral_power(fiber[1][0].channels))
x_datas.append(fiber_init[1][0].time)
x_datas.append(fiber[0][0].time)
y_datas.append(temporal_power(fiber_init[1][0].channels))
y_datas.append(temporal_power(fiber[1][0].channels))
x_datas.append(fiber_init[1][0].time)
x_datas.append(fiber[0][0].time)
y_datas.append(phase(fiber_init[1][0].channels))
y_datas.append(phase(fiber[1][0].channels))

x_labels = ['nu', 't', 't']
y_labels = ['P_nu', 'P_t', 'phi']
plot_titles = ["Spectral power", "Temporal power", "Phase"]
fig_title = "Effect of Raman scattering on Gaussian pulse"

plot_groups = [0,0,1,1,2,2]
plot_labels = 3 * ['original pulse', 'w/ Raman scattering']
x_ranges = [None, (18.5, 21.5), (18.5, 21.5)]

plot2d(x_datas, y_datas, x_labels=x_labels, y_labels=y_labels,
       x_ranges=x_ranges, plot_titles=plot_titles, plot_groups=plot_groups,
       plot_labels=plot_labels, fig_title=fig_title, opacity=0.1,
       triangle_layout=True,
       filename="./examples/fiber_effects/images/rs_effect.png")
