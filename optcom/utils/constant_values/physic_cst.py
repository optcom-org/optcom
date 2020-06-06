import scipy.constants as scipy_cst

# physic constants
C: float = 1.0e-3 * scipy_cst.c   # m / s -> nm / ps
PI: float = scipy_cst.pi
KB: float = scipy_cst.Boltzmann * 1e-6 # J / K = m^2 kg s^-2 K^-1 -> nm^2 kg ps^-2 K^-1
H: float = scipy_cst.Planck * 1e6    # m^2 kg s^-1 -> nm^2 kg ps^-1
HBAR: float = H / (2*scipy_cst.pi)
M_E: float = scipy_cst.electron_mass # kg
C_E: float = scipy_cst.elementary_charge * 1e12     # A s -> A ps
EPS_0: float = scipy_cst.epsilon_0 * 1e21 # m^-3 kg^-1 s^4 A^2 -> nm^-3 kg^-1 ps^4 A^2
TEMPERATURE: float = 293.1  # K
