import sys

import math

from optcom import *


# -------------------------- Coupler -----------------------------------
# set nbr of round trip: nbr round trip = max_nbr_input - 1
max_nbr_input = [5]    # Can set for each port, if 1 nbr, for every ports
# Default value of max recursion in python is 999, need to be increase
if (max_nbr_input[0] > 998):
    sys.setrecursionlimit(10e6)

steps = 1e3
alphas = [0.0,0.0]
betas = [[1.0,1.0,-19.83],[0.5,1.0,-19.83]]
gammas = [0.1,0.1]
kappas = [1, 1]
kappa = kappas[0]
# if L = Lc / 2 -> 50:50 coupler
delta_a = 0.5*(betas[0][0] - betas[1][0])
length_c = math.pi/(2*math.sqrt(delta_a**2 + kappa**2))
length = length_c / 2
coupler = FiberCoupler(max_nbr_input=max_nbr_input,
                       length=length, alphas=alphas, betas=betas,
                       gammas=gammas, kappas=kappas,
                       nl_approx=True, SPM=False, SS=False, RS=False,
                       XPM=False, approx_type=1, method='ssfm_super_sym',
                       steps=steps, save=True, wait=[False, True])


# -------------------------- Fiber -------------------------------------
length = 10 # km
alpha = 0.046   # dB km^-1
betas = [1.0,0.36,0.019]  # ps^n km^-1
gamma = 1.48    # km^-1 W^-1
SPM = False  # self-phase mod
SS = False # self-steepening
RS = False   # Raman scattering
steps = 1e3
method = 'rk4ip' # 'ssfm'
fiber = Fiber(length=length, alpha=alpha, betas=betas, gamma=gamma,
              nl_approx=True, SPM=SPM, SS=SS, RS=RS, method=method,
              steps=steps, save=True)


# -------------------------- Pulse -------------------------------------
width = 10.0     # Ps
peak_power = 10  # W
pulse = Gaussian(peak_power=peak_power, width=width)


# ------------------------ Layout and experiment -----------------------
bit_width = 100.0
lt = Layout(Domain(bit_width=bit_width))

lt.link((pulse[0],coupler[0]), (coupler[2], fiber[0]), (fiber[1], coupler[3]))


# -------------------------- Run ---------------------------------------
lt.run(pulse)


# -------------------------- Plotting results --------------------------
x_datas = [pulse.times[0], coupler.times[0], coupler.times[1]]
y_datas = [temporal_power(pulse.fields[0]),
           temporal_power(coupler.fields[0]),
           temporal_power(coupler.fields[1])]
plot_titles = ["Original pulses", "Pulse exiting the port 0 of the coupler",
               "Pulse exiting the port 0 of the coupler"]

plot(x_datas, y_datas, x_labels=['t'], y_labels=['P_t'], plot_groups=[0,1,2],
     plot_titles=plot_titles, opacity=0.3)
