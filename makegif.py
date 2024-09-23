import imageio
VDF_dir = '/Users/srijanbharatidas/Documents/Research/SpaceWeather/VDF_projects/VDF_Paper1_ESA/VDF_paper1_plots'

'''
images_dir = f'{VDF_dir}/fourway_plot'
with imageio.get_writer(f'{VDF_dir}/fourplot.gif', mode='I') as writer:
    for i in range(1329):
        try:
            image = imageio.imread(f'{images_dir}/{i}.png')
            writer.append_data(image)
        except: continue

images_dir = f'{VDF_dir}/VDF_solo'
with imageio.get_writer(f'{VDF_dir}/soloplot.gif', mode='I') as writer:
    for i in range(1329):
        try:
            image = imageio.imread(f'{images_dir}/{i}.png')
            writer.append_data(image)
        except: continue

images_dir = f'{VDF_dir}/VDF_SPAN_polar_plot'
with imageio.get_writer(f'{VDF_dir}/VDF_SPAN_polar_plot.gif', mode='I') as writer:
    for i in range(878):
        try:
            image = imageio.imread(f'{images_dir}/{i}.png')
            writer.append_data(image)
        except: continue

images_dir = f'{VDF_dir}/VDF_rec_polar_plot'
with imageio.get_writer(f'{VDF_dir}/soloplot.gif', mode='I') as writer:
    for i in range(1329):
        try:
            image = imageio.imread(f'{images_dir}/{i}.png')
            writer.append_data(image)
        except: continue
'''

images_dir = f'{VDF_dir}/2D_MMS'
with imageio.get_writer(f'{VDF_dir}/2D_MMS.gif', mode='I') as writer:
    for i in range(1399):
        try:
            image = imageio.imread(f'{images_dir}/2D_MMS_{i}.png')
            writer.append_data(image)
        except: continue

images_dir = f'{VDF_dir}/3D_MMS'
with imageio.get_writer(f'{VDF_dir}/3D_MMS.gif', mode='I') as writer:
    for i in range(1399):
        try:
            image = imageio.imread(f'{images_dir}/3D_{i}.png')
            writer.append_data(image)
        except: continue

# bash line command I used to generate .mp4 using ffmpeg
# ffmpeg -f image2 -r 10 -i VDF_paper1_plots/3D_MMS/3D_%d.png -vcodec mpeg4 -y -b 1000k 3D_MMS.mp4