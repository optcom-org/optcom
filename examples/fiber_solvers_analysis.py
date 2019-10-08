from optcom import *

lt = Layout(Domain(bit_width=500.0))

peak_power = 0.1
width = 7.0
pulse = Gaussian(peak_power=peak_power, width=width)

length = 20
alpha = 0.046
betas = [0.0, 0.0, -19.83, 0.031]
gamma = 4.3

methods = ["ssfm", "ssfm_reduced", "ssfm_symmetric", "rk4ip"]
#methods = ["rk4ip"]
steps = [1e3, 10e3, 100e3]
fields = []

plot_groups = [0]
plot_labels = [None]
plot_titles = ["Original pulse with peak power {}W and width {}ps"
               .format(str(peak_power), str(width))]

for i, step in enumerate(steps):
    for j, method in enumerate(methods):
        # Propagation
        fiber = Fiber(length=length, method=method, alpha=alpha,
                      betas=betas, gamma=gamma, nl_approx=True,
                      SPM=True, SS=False, RS=True,
                      approx_type=1, steps=step)
        lt.link((pulse[0], fiber[0]))
        lt.run(pulse)
        lt.reset()
        # Plot parameters and get waves
        fields.append(temporal_power(fiber.fields[1]))
        plot_groups += [i+1]
    plot_labels += methods
    plot_titles += ["Pulse coming out of the Fiber with n={}"
                    .format(str(step))]
    fig_title = r"$\alpha={}$, $\beta={}$, $\gamma={}$, $L={}km$".format(
                str(alpha), str(betas), str(gamma), str(length))

plot([lt.domain.time], [temporal_power(pulse.fields[0])]
     +fields, plot_groups=plot_groups, plot_titles=plot_titles,
     x_labels=['t'], y_labels=['P_t'], plot_labels=plot_labels,
     opacity=0.3, fig_title=fig_title)
