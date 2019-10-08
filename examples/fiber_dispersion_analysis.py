from optcom import *

lt = Layout()

peak_power = 10.0
pulse = Gaussian(peak_power=peak_power)

method = "ssfm_symmetric"
steps = 100
fields = []
approx_types = [1,2,3]
disp = [[1.0, 0.0, 0.0, 0.0],
        [0.0, 3.8, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]]

plot_groups = [0]
plot_labels = [None]
plot_titles = ["Original pulse"]

for i, d in enumerate(disp):
    for approx_type in approx_types:
        # Propagation
        fiber = Fiber(length=4.0, method=method, alpha=0.3,
                      betas=d, gamma=0.9, nl_approx=True,
                      SPM=True, SS=True, RS=True,
                      approx_type=approx_type, steps=steps)
        lt.link((pulse[0], fiber[0]))
        lt.run(pulse)
        lt.reset()
        # Plot parameters and get waves
        fields.append(temporal_power(fiber.fields[1]))
        plot_groups += [i+1]
        plot_labels += ["NLSE approx type {}".format(approx_type)]

    plot_titles += ["Pulse at the fiber end, {} spatial steps "
                    "and dispersion coeff {}"
                    .format(str(steps), d)]
plot([lt.domain.time], [temporal_power(pulse.fields[0])]
     +fields, plot_groups=plot_groups, plot_titles=plot_titles,
     x_labels=['t'], y_labels=['P_t'], plot_labels=plot_labels, opacity=0.3)
