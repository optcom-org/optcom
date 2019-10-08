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
# set nbr of round trip: nbr round trip = max_nbr_pass
max_nbr_pass = [1]    # Can set for each port, if 1 nbr, for every ports
# Default value of max recursion in python is 999, need to be increase
if (max_nbr_pass[0] > 998):
    sys.setrecursionlimit(10e6)

steps = int(1e3)
method = 'rk4ip'
alpha = [0.0,0.0]
beta = [[1.0,10.0],[1.0,10.0]]
gamma = [4.3, 4.3]
# k = 1cm^-1
kappa = [1.0e5, 1.0e5]  # cm^-1 -> km^-1
delta_a = 0.5*(beta[0][0] - beta[1][0])
length_c = math.pi/(2*math.sqrt(delta_a**2 + kappa[0]**2))
length = length_c / 2

coupler = FiberCoupler(length=length, alpha=alpha, beta=beta, gamma=gamma,
                       kappa=kappa, ATT=True, DISP=True,
                       nl_approx=False, SPM=True, SS=False, RS=True,
                       XPM=True, approx_type=1, method='ssfm_super_sym',
                       steps=steps, save=True, wait=False,
                       max_nbr_pass=max_nbr_pass)

# -------------------------- Fiber -------------------------------------
length = 1.0 # km
alpha = [0.046]   # dB km^-1
beta = [50.0,10.0,19.31]  # ps^n km^-1
gamma = 1.48    # km^-1 W^-1
SPM = True  # self-phase mod
SS = False # self-steepening
RS = True   # Raman scattering
steps = 1e3
method = 'rk4ip' # 'ssfm'
#method = 'ssfm_super_sym'
fiber = Fiber(length=length, method=method, alpha=alpha,
              #beta=beta,
              #gamma=4.3,
              beta_order=3,
              nl_approx=False, ATT=True, DISP=True,
              SPM=True, XPM=True, SS=True, RS=True, approx_type=1,
              steps=1000, medium='SiO2')


# -------------------------- Amplifier ---------------------------------
steps = 1000
length = 0.0008   # km
#sigma_a = [2e-8, 2e-6]
#sigma_e = [2e-7, 1e-6]
file_sigma_a = ('./data/fiber_amp/cross_section/absorption/yb.txt')
file_sigma_e = ('./data/fiber_amp/cross_section/emission/yb.txt')
sigma_a = CSVFit(file_sigma_a, conv_factor=[1e9, 1e18])
sigma_e = CSVFit(file_sigma_e, conv_factor=[1e9, 1e18])
n_core = None
n_clad = 0.0
NA = 0.2
r_core = 5.0    # um
r_cladding = 62.5   # um
temperature = 293.15    # K
tau_meta = 840.0  # us
N_T = 6.3e-2    # nm^{-3}
eta_s = 1.26    # km^-1
eta_p = 1.41    # km^-1
R_0 = 8e-4
R_L = R_0
medium = 'sio2'
dopant = 'yb'
error = 0.1


amp = FiberAmplifier(length=length, method="ssfm_super_sym", alpha=[0.046],
                      beta_order = 2,
                      #beta=[0.0,0.0,10.0,-19.83,0.031],
                      #gamma=0.43,
                      gain_order=0, GS=True,
                      nl_approx=False, ATT=True, DISP=True,
                      SPM=True, XPM=True, SS=True, RS=True,
                      approx_type=1, sigma_a=sigma_a, sigma_e=sigma_e,
                      n_core=n_core, n_clad=n_clad, NA=NA,
                      core_radius=r_core, clad_radius=r_cladding,
                      temperature=temperature, tau_meta=tau_meta,
                      N_T=N_T, eta_s=eta_s, eta_p=eta_p, R_0=R_0, R_L=R_L,
                      medium=medium, dopant=dopant, steps=steps, error=error,
                      para_update=True,
                      solver_order='following', save=True, save_all=False)

# -------------------------- Pulse -------------------------------------
center_lambda = 1030.0    # nm
bit_rate = 0.0    # GHz
width = 3.0    # Ps
peak_power = 1.0  # W
pulse = Gaussian(bit_rate=bit_rate, width=width, peak_power=peak_power,
                 center_lambda=center_lambda)
# -------------------------- Pump --------------------------------------
center_lambda = 976.0    # nm
bit_rate = 0.0    # GHz
width = 50.0    # Ps
peak_power = 0.2  # W
pump = Gaussian(bit_rate=bit_rate, width=width, peak_power=peak_power,
                center_lambda=center_lambda)
pump = CW(peak_power=peak_power, center_lambda=center_lambda)


# -------------------------- Domain ------------------------------------
samples_per_bit = 1024   # take power of 2 to make fft calc efficient
bits = 1
bit_width = 200.0  # ps

domain = Domain(bits=bits, samples_per_bit=samples_per_bit,
                bit_width=bit_width)


# ------------------------ Layout and experiment -----------------------
lt = Layout(domain)

lt.link((pulse[0],MZ1[0]), (MZ1[1], coupler[0]), (coupler[2], MZ2[0]),
        (coupler[3], fiber[0]), (fiber[1], amp[0]), (pump[0], amp[2]),
        (amp[1], MZ3[0]), (MZ3[1], coupler[1]))


# -------------------------- Run ---------------------------------------
lt.run(pulse, pump)#, pump, pump, pump, pump)#, pump)#, pump, pump, pump, pump)#, pump)


# -------------------------- Plotting results --------------------------
'''
x_datas = [pulse.fields[0].time, pulse.fields[0].nu, coupler.fields[1].time,
           coupler.fields[1].nu]
y_datas = [temporal_power(pulse.fields[0].channels),
           spectral_power(pulse.fields[0].channels),
           temporal_power(coupler.fields[1].channels),
           spectral_power(coupler.fields[1].channels)]
plot_titles = ["Original pulses", "Pulse exiting the port 2 of the coupler "
                "after {} round trips".format(max_nbr_pass[0]),
               "Original pulses", "Pulse exiting the port 2 of the coupler "
               "after {} round trips".format(max_nbr_pass[0])]
plot(x_datas, y_datas, x_labels=['t', 'nu', 't', 'nu'],
     y_labels=['P_t', 'P_nu', 'P_t', 'P_nu'], plot_groups=[0,1,2,3],
     plot_titles=plot_titles, opacity=0.3)

'''
x_datas = [pulse.fields[0].time, coupler.fields[2].time]
y_datas = [temporal_power(pulse.fields[0].channels),
           temporal_power(coupler.fields[2].channels)]
plot_titles = ["Original pulse", "Pulse exiting the port [3] of the coupler "
               "after {} round trips".format(max_nbr_pass[0])]

plot(x_datas, y_datas, x_labels=['t'],
     y_labels=['P_t'], plot_groups=[0,1],
     plot_titles=plot_titles, opacity=0.3)
