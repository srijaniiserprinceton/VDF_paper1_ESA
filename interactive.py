from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

def signal(amp, freq):
    return amp * sin(2 * pi * freq * t)

def gauss_2D(mu1, mu2):
    return np.exp(-0.5 * (((xx-mu1)/sig)**2 + ((yy-mu2)/sig)**2))

axis_color = 'lightgoldenrodyellow'

fig = plt.figure()
ax = fig.add_subplot(111)

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.25)

x = np.linspace(-100, 100, 100)
y = np.linspace(-100, 100, 100)
xx, yy = np.meshgrid(x, y, indexing='ij')

# Draw the initial plot
mu10, mu20, sig = 0,0,5
im = ax.pcolormesh(xx, yy, gauss_2D(mu10, mu20))

# Add two sliders for tweaking the parameters

# Define an axes area and draw a slider in it
mu1_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
mu1_slider = Slider(mu1_slider_ax, 'mu1', -100, 100.0, valinit=0.0)

# Draw another slider
mu2_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axis_color)
mu2_slider = Slider(mu2_slider_ax, 'mu2', -100, 100.0, valinit=0.0)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    im.set_array(gauss_2D(mu1_slider.val, mu2_slider.val))
    fig.canvas.draw_idle()
mu1_slider.on_changed(sliders_on_changed)
mu2_slider.on_changed(sliders_on_changed)

'''
# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    freq_slider.reset()
    amp_slider.reset()
reset_button.on_clicked(reset_button_on_clicked)

# Add a set of radio buttons for changing color
color_radios_ax = fig.add_axes([0.025, 0.5, 0.15, 0.15], facecolor=axis_color)
color_radios = RadioButtons(color_radios_ax, ('red', 'blue', 'green'), active=0)
def color_radios_on_clicked(label):
    line.set_color(label)
    fig.canvas.draw_idle()
color_radios.on_clicked(color_radios_on_clicked)
'''

plt.show()
