from optcom import *

lt = Layout()

peak_power = 10
pulse = Gaussian(peak_power=peak_power)

method = "rk4ip"
steps = 10000
length = 4.0
alpha = 0.3
betas = [.0,-.3,.0,.0]
gamma = 1.3
fields = []
approx_types = [1,2,3]
SS = [False, True]

plot_groups = [0]
plot_labels = [None]
plot_titles = ["Original pulse"]

for i, ss in enumerate(SS):
    for approx_type in approx_types:
        # Propagation
        fiber = Fiber(length=length, method=method, alpha=alpha,
                      betas=betas, gamma=gamma, nl_approx=True,
                      SPM=True, SS=ss, RS=True,
                      approx_type=approx_type, steps=steps)
        lt.link((pulse[0], fiber[0]))
        lt.run(pulse)
        lt.reset()
        # Plot parameters and get waves
        fields.append(temporal_power(fiber.fields[1]))
        plot_groups += [i+1]
        plot_labels += ["NLSE approx type {}".format(approx_type)]
    # Propagation
    fiber = Fiber(length=length, method=method, alpha=alpha,
                  betas=betas, gamma=gamma, nl_approx=False,
                  SPM=True, SS=ss, RS=True, approx_type=1,
                  steps=steps)
    lt.link((pulse[0], fiber[0]))
    lt.run(pulse)
    lt.reset()
    # Plot parameters and get waves
    fields.append(temporal_power(fiber.fields[1]))
    plot_groups += [i+1]
    plot_labels += ["(G)NLSE"]

plot_titles += (["Pulse at the fiber end without SS, {} spatial "
                 "steps and {} solver"
                 .format(str(steps), method)]
               + ["Pulse at the fiber end with SS, {} spatial steps "
                  "and {} solver"
                  .format(str(steps), method)])
plot([lt.domain.time], [temporal_power(pulse.fields[0])]
     +fields, plot_groups=plot_groups, plot_titles=plot_titles,
     x_labels=['t'], y_labels=['P_t'], plot_labels=plot_labels, opacity=0.3)
