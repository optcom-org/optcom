from optcom import *

lt = Layout()

peak_power = 10.0
pulse = Gaussian(peak_power=peak_power)

method = "ssfm_super_sym"
steps = 10000
fields = []
XPM = [False, True]

plot_groups = [0]
plot_titles = ["Original pulse"]

for i, xpm in enumerate(XPM):
    # Propagation
    fiber = Fiber(length=4.0, method=method, alpha=0.3,
                  betas=[.0,-.3,.0,.0], gamma=0.2, nl_approx=True,
                  SPM=True, SS=True, RS=True, XPM=xpm, approx_type=3,
                  steps=steps)
    lt.link((pulse[0], fiber[0]))
    lt.run(pulse)
    lt.reset()
    # Plot parameters and get waves
    fields.append(temporal_power(fiber.fields[1]))
    plot_groups += [i+1]
plot_titles += (["Pulse at the fiber end without XPM, {} spatial "
                 "steps and {} solver".format(str(steps), method)]
                + ["Pulse at the fiber end with XPM, {} spatial "
                  "steps and {} solver".format(str(steps), method)])
plot([lt.domain.time], [temporal_power(pulse.fields[0])]
          +fields, plot_groups=plot_groups, plot_titles=plot_titles,
          x_labels=['t'], y_labels=['P_t'], opacity=0.3)
