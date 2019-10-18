from optcom import *

domain = Domain(samples_per_bit=1024, bit_width=3.0)

lt = Layout(domain)
pulse = Gaussian(channels=1, peak_power=[5.0], width=[0.2],
                 center_lambda=[1550.0])

gamma = 0.5
steps = int(1e4)

x_datas = []
y_datas = []

fiber = Fiber(length=5.0, method="ssfm_symmetric",
              gamma=gamma, nl_approx=True, ATT=False, DISP=False,
              SPM=False, SS=True, RS=False, steps=steps, medium='sio2',
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

x_labels = ['t', 'nu']
y_labels = ['P_t', 'P_nu']
plot_titles = ["Effect of self-steepening on Gaussian pulse - Temporal power",
               "Effect of self-steepening on Gaussian pulse - Spectral power"]

plot_groups = [0,0,1,1]
plot_labels = ['original pulse', 'w/ self-steepening', 'original pulse',
               'w/ self-steepening']


plot(x_datas, y_datas, x_labels=x_labels, y_labels=y_labels,
     plot_titles=plot_titles, plot_groups=plot_groups,
     plot_labels=plot_labels, opacity=0.1)
