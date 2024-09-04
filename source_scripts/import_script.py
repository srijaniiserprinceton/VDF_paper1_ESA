def import_instrument_scripts(instrument):
    '''
    Function to import relevant functions based on the instrument we are using.
    '''
    if(instrument == 'PSP-SPAN'): 
        # imports from our custom package
        from source_scripts.PSP import sph2slep, extract_data, locate_axis, VDF_rec_polarcaps, VDF_rec_cartesian
        return sph2slep, extract_data, locate_axis, VDF_rec_polarcaps, VDF_rec_cartesian

    elif(instrument == 'MMS'):
        # imports from our custom package
        from source_scripts.MMS import sph2slep, extract_data, locate_axis, VDF_rec_polarcaps
        return sph2slep, extract_data, locate_axis, VDF_rec_polarcaps, None