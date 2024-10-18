def import_instrument_scripts(instrument):
    '''
    Function to import relevant functions based on the instrument we are using.
    '''
    if(instrument == 'PSP-SPAN'): 
        # imports from our custom package
        from source_scripts.PSP import sph2slep, extract_data, locate_axis, VDF_rec_polarcaps, VDF_rec_cartesian
        return sph2slep, extract_data, locate_axis, VDF_rec_polarcaps, VDF_rec_cartesian, None
    
    elif(instrument == 'PSP-SPAN-MT'):
        # imports from our custom package
        from source_scripts.PSP import sph2slep, extract_data, locate_axis
        return sph2slep, extract_data, locate_axis, None, None
        

    elif(instrument == 'MMS'):
        # imports from our custom package
        from source_scripts.MMS import sph2slep, extract_data, locate_axis, VDF_rec_polarcaps, build_3D_VDF
        return sph2slep, extract_data, locate_axis, VDF_rec_polarcaps, build_3D_VDF, None

    elif(instrument == 'SolO'):
        # imports from our custom package
        from source_scripts.SolO import sph2slep, extract_data, locate_axis, VDF_rec_polarcaps, build_3D_VDF
        from Plotter.SolO import plot_VDF
        return sph2slep, extract_data, locate_axis, VDF_rec_polarcaps, build_3D_VDF, plot_VDF