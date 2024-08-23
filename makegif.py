import imageio
VDF_dir = '/Users/srijanbharatidas/Documents/Research/SpaceWeather/VDF_projects/VDF_Paper1_ESA/VDF_paper1_plots'


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

'''
images_dir = f'{VDF_dir}/VDF_rec_polar_plot'
with imageio.get_writer(f'{VDF_dir}/soloplot.gif', mode='I') as writer:
    for i in range(1329):
        try:
            image = imageio.imread(f'{images_dir}/{i}.png')
            writer.append_data(image)
        except: continue
'''