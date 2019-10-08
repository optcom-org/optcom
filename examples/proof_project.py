import sys

import math

from optcom import *

# -------------------------- Mach Zehnder mod --------------------------
PS_MZ1 = 0.0  # rad
loss_MZ1 = 0.0  # dB
PS_MZ2 = 0.0
loss_MZ2 = 0.0
PS_MZ3 = 0.0
loss_MZ3 = 0.0

MZ1 = IdealMZ(phase_shift=PS_MZ1, loss=loss_MZ1)
MZ2 = IdealMZ(phase_shift=PS_MZ2, loss=loss_MZ2)
MZ3 = IdealMZ(phase_shift=PS_MZ3, loss=loss_MZ3)


# -------------------------- Coupler -----------------------------------
# set nbr of round trip: nbr round trip = max_nbr_input - 1
max_nbr_input = [5]    # Can set for each port, if 1 nbr, for every ports
# Default value of max recursion in python is 999, need to be increase
if (max_nbr_input[0] > 998):
    sys.setrecursionlimit(10e6)

coupler = IdealFiberCoupler(max_nbr_input=max_nbr_input, save=True)


# -------------------------- Fiber -------------------------------------
length = 10 # km
alpha = 0.046   # dB km^-1
beta = [1.0,0.36,0.019]  # ps^n km^-1
gamma = 1.48    # km^-1 W^-1
SPM = False  # self-phase mod
SS = False # self-steepening
RS = False   # Raman scattering
steps = 1e3
method = 'rk4ip' # 'ssfm'
fiber = Fiber(length=length, alpha=alpha, beta=beta, gamma=gamma,
              nl_approx=True, SPM=SPM, SS=SS, RS=RS, method=method,
              steps=steps, save=True)


# -------------------------- Amplifier ---------------------------------
gain_coupler = 10*math.log10(2)    # dB , from the coupler loss
gain_fiber = length*alpha  # dB , from the fiber loss
gain = gain_coupler + gain_fiber

amp = IdealAmplifier(gain=gain, save=True)


# -------------------------- Pulse -------------------------------------
bit_rate = 0.0    # GHz
width = 0.4     # Ps
peak_power = 1  # W
pulse = Gaussian(bit_rate=bit_rate, width=width, peak_power=peak_power)


# -------------------------- Domain ------------------------------------
center_lambda = 1030    # nm
center_nu = Domain.lambda_to_nu(center_lambda)
samples_per_bit = 1024   # take power of 2 to make fft calc efficient
bits = 4
bit_width = 10  # ps

domain = Domain(bits=bits, samples_per_bit=samples_per_bit,
                bit_width=bit_width, center_nu=center_nu)


# ------------------------ Layout and experiment -----------------------
lt = Layout(domain)

lt.link((pulse[0],MZ1[0]), (MZ1[1], coupler[0]), (coupler[2], MZ2[0]),
        (coupler[3], fiber[0]), (fiber[1], amp[0]), (amp[1], MZ3[0]),
        (MZ3[1], coupler[1]))


# -------------------------- Run ---------------------------------------
lt.run(pulse)


# -------------------------- Plotting results --------------------------
x_datas = [pulse.times[0], coupler.times[1]]
y_datas = [temporal_power(pulse.fields[0]),
           temporal_power(coupler.fields[1])]
plot_titles = ["Original pulses", "Pulse exiting the port 2 of the coupler "
                "after {} round trips".format(max_nbr_input[0]-1)]

plot(x_datas, y_datas, x_labels=['t'], y_labels=['P_t'], plot_groups=[0,1],
     plot_titles=plot_titles, opacity=0.3)
