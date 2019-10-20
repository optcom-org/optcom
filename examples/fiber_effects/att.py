from optcom import *

domain = Domain(samples_per_bit=2**(15), bit_width=3.0)

lt = Layout(domain)
pulse = Gaussian(channels=1, peak_power=[5.0], width=[0.2],
                 center_lambda=[1550.0])

alpha = 0.046
steps = int(1e4)

x_datas = []
y_datas = []

fiber = Fiber(length=5.0, method="ssfm_symmetric", alpha=alpha,
              nl_approx=True, ATT=True, DISP=False,
              SPM=False, SS=False, RS=False, steps=steps, medium='sio2',
              save=True)

lt.link((pulse[0], fiber[0]))
lt.run(pulse)
x_datas.append(pulse.fields[0].time)
x_datas.append(fiber.fields[0].time)
y_datas.append(temporal_power(pulse.fields[0].channels))
y_datas.append(temporal_power(fiber.fields[1].channels))
x_datas.append(pulse.fields[0].nu)
x_datas.append(fiber.fields[0].nu)
y_datas.append(spectral_power(pulse.fields[0].channels))
y_datas.append(spectral_power(fiber.fields[1].channels))
x_datas.append(pulse.fields[0].time)
x_datas.append(fiber.fields[1].time)
y_datas.append(phase(pulse.fields[0].channels))
y_datas.append(phase(fiber.fields[1].channels))

x_labels = ['t', 'nu', 't']
y_labels = ['P_t', 'P_nu', 'phi']
plot_titles = ["Temporal power",
               "Spectral power",
               "Phase"]
fig_title = "Effect of attenuation on Gaussian pulse"

plot_groups = [0,0,1,1,2,2]
plot_labels = 3 * ['original pulse', 'w/ attenuation']


plot(x_datas, y_datas, x_labels=x_labels, y_labels=y_labels,
     plot_titles=plot_titles, plot_groups=plot_groups,
     plot_labels=plot_labels, fig_title=fig_title, opacity=0.1)
