import numpy as np

# Merge dist data intervals
def merge_vdf_data(time, vdf, energy, velocity, theta, phi):
    """
    This function takes in the ion VDF and reduces time resolution from 150 ms to 300 ms
    to gain increased energy resolution. 32 ---> 64 energy channels.
    
    Inputs: 
        time  - is the distribution time stamps (N)
        vdf   - is the original 3D velocity distribution (N x 32 x 16 x 32)
        vel   - is the velocity channel at each time. (N x 32) array
        theta - is the corresponding elevation angle. (16) array
                Note: this value is not timestamped as this angle is the size of the 
                anodes
        phi   - is the azimuthal angle. (N x 32) array
    
    Outputs:
        time_merged  - new time stamp
        vel_merged   - new velocity array
        theta_merged - returned as theta. included for completeness
        phi_merged   - Currently returns the average phi for both bins. NEEDS CORRECTIONS
        vdf_merged   - The adjusted VDF 
    """
    # Check if the time array is even or odd
    time_temp = time - time[0]
    if len(time) % 2 == 0:
        time_merged = np.sum(time_temp.reshape(len(time)//2, 2), axis=1)/2. + time[0]
    else:
        time_merged = np.sum(time_temp[:-1].reshape(len(time)//2, 2), axis=1)/2. + time[0]

    # Merge the velocity array and reduce time
    vel_merged = np.append(velocity[:-1:2,:], velocity[1::2,:], axis=1)
    energy_merged = np.append(energy[:-1:2,:], energy[1::2,:], axis=1)

    vdf_merged = np.append(vdf[:-1:2,:,:,:], vdf[1::2,:,:,:], axis=1)

    # Sort the velocity array
    ind = np.argsort(vel_merged[:,:], axis=1)
    rows, cols = np.meshgrid(np.arange(ind.shape[0]), np.arange(ind.shape[1]) , indexing='ij')

    vel_merged = vel_merged[rows, ind]
    energy_merged = energy_merged[rows, ind]
    vdf_merged = vdf_merged[rows, ind, :, :]

    # Average the two phi bins together (There is a 2 degree offest from each measurement)
    phi_merged = (phi[:-1:2,:] + phi[1::2,:])/2.

    theta_merged = theta[0:-1:2, :]

    return [time_merged, vdf_merged, energy_merged, vel_merged, theta_merged, phi_merged]
