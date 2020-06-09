import optcom as oc

# Create 2 Gaussian channels
pulse = oc.Gaussian(channels=2, center_lambda=[1030., 1550.], peak_power=[0.5, 1.0])
# Create fiber with a user-defined attenuation coefficient
fiber = oc.Fiber(length=1.0, alpha=[0.4], ATT=True, DISP=True, SPM=True, save_all=True)
# Create an optical layout and link the first port of 'pulse' to the first port of 'fiber'
layout = oc.Layout()
layout.add_link(pulse.get_port(0), fiber.get_port(0))
layout.run_all()
# Extract outputs and plot
time = fiber.storage.time
power = oc.temporal_power(fiber.storage.channels)
space = fiber.storage.space
oc.animation2d(time, power, space, x_label=['t'], y_label=['P_t'],
               plot_title='My first Optcom example',
               plot_labels=['1030. nm channel', '1550. nm channel'])
               #filename='example_anim_readme.mp4')
