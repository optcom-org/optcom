import copy

from optcom import *


lt_1 = Layout(Domain(samples_per_bit=512,bit_width=500.0))
lt_2 = Layout(Domain(samples_per_bit=512,bit_width=500.0))
combiner = IdealCombiner(arms=2, combine=False)
pulse_1 = Gaussian(channels=1, peak_power=[25e-3], width=[7.0],
                   center_lambda=1552.0)
pulse_2 = copy.deepcopy(pulse_1)

cw = CW(channels=1, peak_power=15e-3, center_lambda=980.0)

fiber_1 = Fiber(length=20, method="ssfm", alpha=0.046,
              beta=[0.0,0.0,10.0,-19.83,0.031],
              gamma=4.3, nl_approx=False, ATT=True, DISP=True,
              SPM=True, XPM=True, SS=False, RS=True, approx_type=1,
              steps=1000, save=True)

fiber_2 = copy.deepcopy(fiber_1)

lt_1.link((pulse_1[0], fiber_1[0]))
lt_1.run(pulse_1)
lt_2.link((pulse_2[0], combiner[0]))
lt_2.link((cw[0], combiner[1]))
lt_2.link((combiner[2], fiber_2[0]))
lt_2.run(pulse_2, cw)
'''
x_datas = [pulse.fields[0].nu, cw.fields[0].nu, fiber.fields[1].nu,
           pulse.fields[0].time, cw.fields[0].time, fiber.fields[1].time]
y_datas = [spectral_power(pulse.fields[0].channels),
           spectral_power(cw.fields[0].channels),
           spectral_power(fiber.fields[1].channels),
           temporal_power(pulse.fields[0].channels),
           temporal_power(cw.fields[0].channels),
           temporal_power(fiber.fields[1].channels)]
x_labels = ['nu', 'nu', 'nu', 't', 't', 't']
y_labels = ['P_nu', 'P_nu', 'P_nu', 'P_t', 'P_t', 'P_t']
plot_titles = ['a', 'b', 'c', 'd', 'e', 'f']
plot_groups=[0,1,2,3,4,5]
'''
x_datas = [fiber_1.fields[1].time, fiber_2.fields[1].time]
y_datas = [temporal_power(fiber_1.fields[1].channels),
           temporal_power(fiber_2.fields[1].channels)]
x_labels = ['t', 't']
y_labels = ['P_t', 'P_t']
plot_titles = ['a', 'b']
plot_groups=[0,1]


plot(x_datas, y_datas, x_labels = x_labels, y_labels = y_labels,
     plot_titles=plot_titles, plot_groups=plot_groups, opacity=0.3)
