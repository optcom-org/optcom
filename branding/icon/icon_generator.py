from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np


wavelengths: Dict[str, float] = {'red': 700., 'orange': 620., 'yellow': 580.,
                                 'greenyellow': 530., 'cyan': 500.,
                                 'blue': 470., 'violet': 420.}

# Choose colors
chosen_colors: List[str] = ['red', 'greenyellow', 'blue']
nbr_colors: int = len(chosen_colors)
period_ratios: List[float] = [wavelengths[color] for color in chosen_colors]
max_wavelength: float = max(period_ratios)
# Normalization
period_ratios: List[float] = [ratio/max_wavelength for ratio in period_ratios]

period_scale: float = 3.
amplitude_scale: float = 1.5

def sin_func(x_data: np.ndarray, wavelength: float) -> np.ndarray:

    return amplitude_scale * np.sin(np.pi*x_data*period_scale/wavelength)

fig, ax = plt.subplots()
x_data = np.linspace(0.0, 1.5, 10000, False)
total_angle: float = 160.
angles = [(-(total_angle/2)+(i*(total_angle/(nbr_colors-1))))
          for i in range(nbr_colors)]

for i in range(nbr_colors):
    y_data = sin_func(x_data, period_ratios[i])
    #y_data += (10/4) * amplitude_scale * i
    base = plt.gca().transData
    rot = transforms.Affine2D().skew_deg(0.,angles[i])
    ax.plot(x_data, y_data, color=chosen_colors[i], transform=rot+base,
            linewidth=20., zorder=0)

ax.scatter(0.,0., 5000., color='black', zorder=1)

plt.axis('off')

fig.set_size_inches(((12,12)))
dpi = 400
title_str = 'icon'
plt.savefig('./branding/icon/'+title_str+'.png', dpi=dpi)
