import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

def get_the_slice(x,y,z, surfacecolor):
    return go.Surface(x=x,
                      y=y,
                      z=z,
                      surfacecolor=surfacecolor,
                      coloraxis='coloraxis',
                      opacity=1.0)

def get_lims_colors(surfacecolor):# color limits for a slice
    return np.min(surfacecolor), np.max(surfacecolor)

def get_surf_color(surface='z', idx=50, Ngrids=200):
    if(surface=='z'):
        xr, yr = np.meshgrid(np.linspace(x[:,idx].min(), x[:,idx].max(), Ngrids), np.linspace(y[:,idx].min(), y[:,idx].max(), Ngrids))
        zr = np.zeros_like(xr)
        surf_color = griddata((x[:,idx].flatten(), y[:,idx].flatten()), VDF[:,idx].flatten(), (xr, yr) , method='cubic', fill_value=VDF[:,idx].min())
    elif(surface=='y'):
        xr, zr = np.meshgrid(np.linspace(x[:,:,idx].min(), x[:,:,idx].max(), Ngrids), np.linspace(z[:,:,idx].min(), z[:,:,idx].max(), Ngrids))
        yr = np.zeros_like(xr)
        surf_color = griddata((x[:,:,idx].flatten(), z[:,:,idx].flatten()), VDF[:,:,idx].flatten(), (xr, zr) , method='cubic', fill_value=VDF[:,:,idx].min())
    
    return xr, yr, zr, surf_color


# loading the data
x = np.load('./output_data_files/VX.npy')
y = np.load('./output_data_files/VY.npy')
z = np.load('./output_data_files/VZ.npy')
VDF = np.load('./output_data_files/VDF_3D_rec.npy')



xx, yy, zz, surf_color_z = get_surf_color(surface='z', idx=50)
slice_z = get_the_slice(xx, yy, zz, surf_color_z)

xx, yy, zz, surf_color_y1 = get_surf_color(surface='y', idx=100)
slice_y1 = get_the_slice(xx, yy, zz, surf_color_y1)

xx, yy, zz, surf_color_y2 = get_surf_color(surface='y', idx=-1)
slice_y2 = get_the_slice(xx, yy, zz, surf_color_y2)

def colorax(vmin, vmax):
    return dict(cmin=vmin,
                cmax=vmax)

vmax = VDF.max()
vmin = vmax-7

fig1 = go.Figure(data=[slice_z, slice_y1, slice_y2])
fig1.update_layout(
         title_text='Slices in reconstructed VDF', 
         font=dict(color="white"
         ),
         title_x=0.5,
         paper_bgcolor='black',
         width=700,
         height=700, 
         coloraxis=dict(colorscale='inferno',
                        colorbar_thickness=25,
                        colorbar_len=0.75,
                        **colorax(vmin, vmax)))
fig1.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
fig1.show()