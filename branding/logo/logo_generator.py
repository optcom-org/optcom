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

period_scale: float = 2.0
amplitude_scale: float = 0.47

def sin_func(x_data: np.ndarray, wavelength: float) -> np.ndarray:

    return amplitude_scale * np.sin(np.pi*x_data*period_scale/wavelength)

fig, ax = plt.subplots()
x_data = np.linspace(0.0, 2.0, 10000, False)
for i in range(nbr_colors):
    y_data = sin_func(x_data, period_ratios[i])
    y_data += (10/4) * amplitude_scale * i
    ax.plot(x_data, y_data, color=chosen_colors[i])

total_offset = (nbr_colors-1) * (10/4) * amplitude_scale
total_y_length = 12.
y_margin = 2 * amplitude_scale
y_margin = (total_y_length - total_offset) / 2.
ax.set_ylim((-y_margin, total_offset+y_margin))

text_x_pos = (abs(ax.get_xlim()[1]+ax.get_xlim()[0])) / 2.
text_y_pos = (abs(ax.get_ylim()[1]+ax.get_ylim()[0])) / 2.

co_name = 'OPTCOM'
font = {'family': 'serif',
        'color': 'black',
        'style': 'italic',
        'weight': 'ultralight',
        'variant': 'small-caps'}
a = ax.text(text_x_pos, text_y_pos, co_name, ha='center', va='center',
        fontdict=font, size= 160)
plt.axis('off')
#plt.show()

fig.set_size_inches(((16,9)))
dpi = 600
title_str = 'logo'
plt.savefig('./branding/logo/'+title_str+'.png', dpi=dpi)
