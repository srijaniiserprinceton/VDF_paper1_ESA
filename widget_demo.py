import numpy as np
from scipy.interpolate import LinearNDInterpolator as interp
import pickle, sys
import matplotlib.pyplot as plt; plt.ion(); plt.style.use('dark_background')
from matplotlib.widgets import Slider, Button, RadioButtons

t = np.linspace(-np.pi, np.pi, 100)

def makewave(freq):
    return np.sin(freq * t)

#----------------- interactive plotting --------------------#
fill_color = 'black'
fig = plt.figure(figsize=(8,10))
ax1 = fig.add_subplot(111)
fig.subplots_adjust(bottom=0.25)

[spcmin] = ax1.plot(t, makewave(1), '-y')

# Define an axes area and draw a slider in it
axis_color = 'white'
freq_slider_ax  = fig.add_axes([0.4, 0.12, 0.2, 0.03], facecolor=axis_color)
freq_slider = Slider(freq_slider_ax, r'$\nu$', 0, 50, valinit=1)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    spcmin.set_ydata(makewave(freq_slider.val))
    fig.canvas.draw_idle()

freq_slider.on_changed(sliders_on_changed)

# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color='black', hovercolor='0.1')
def reset_button_on_clicked(mouse_event):
    freq_slider.reset()
reset_button.on_clicked(reset_button_on_clicked)