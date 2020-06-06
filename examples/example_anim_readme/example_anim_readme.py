# Copy of the readme code for the animation 2D example
from optcom import *

# Create 2 Gaussian channels
pulse = Gaussian(channels=2, center_lambda=[1030., 1550.],
                 peak_power=[0.5, 1.0])
# Create fiber with a user-defined attenuation coefficient
fiber = Fiber(length=1.0, alpha=[0.4], ATT=True, DISP=True, SPM=True,
              save_all=True)
# Create an optical layout and link the 2 first ports of pulse and fiber
layout = Layout()
layout.link((pulse[0], fiber[0]))
layout.run_all()
# Extract outputs and plot
time = fiber.storage.time
power = temporal_power(fiber.storage.channels)
space = fiber.storage.space
animation2d(time, power, space, x_label=['t'], y_label=['P_t'],
            plot_title='My first Optcom example',
            plot_labels=['1030. nm channel', '1550. nm channel'],
            filename='example_anim_readme.mp4')
