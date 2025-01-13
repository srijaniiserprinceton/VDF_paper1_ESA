import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.ion()

with open('/home/michael/Research/VDF_paper1_ESA/rec_moments.pkl', 'rb') as file:
    rec_moments = pickle.load(file)

with open('/home/michael/Research/VDF_paper1_ESA/data_moments.pkl', 'rb') as file:
    data_moments = pickle.load(file)


rec_den, data_den = [], []
rec_vel, data_vel = [], []
rec_pten, data_pten = [], []

for i in rec_moments.keys():
    rec_den.append(rec_moments[i][0])
    rec_vel.append(rec_moments[i][1])
    rec_pten.append(rec_moments[i][2])

    data_den.append(data_moments[i][0])
    data_vel.append(data_moments[i][1])
    data_pten.append(data_moments[i][2])

plt.figure(layout='constrained')
plt.scatter(np.array(data_den)/100**3, np.array(rec_den)/100**3, marker='.', color='k')
plt.ylabel('reconstructed density')
plt.xlabel('data density')
xx = np.linspace(min(np.array(data_den)/100**3), max(np.array(data_den)/100**3))
plt.plot(xx, xx, color='r')
plt.xlim([10, 170])
plt.ylim([10, 170])


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4), layout='constrained')

ax[0].scatter(np.array(data_vel)[:,0]/1000, np.array(rec_vel)[:,0]/1000, marker='.', color='k')
ax[0].set_ylabel('reconstructed vx')
ax[0].set_xlabel('data vx')
xx1 = np.linspace(min(np.array(rec_vel)[:,0]/1000), max(np.array(rec_vel)[:,0]/1000))
ax[0].plot(xx1, xx1, color='r')

ax[1].scatter(np.array(data_vel)[:,1]/1000, np.array(rec_vel)[:,1]/1000, marker='.', color='k')
ax[1].set_ylabel('reconstructed vy')
ax[1].set_xlabel('data vy')
xx2 = np.linspace(min(np.array(rec_vel)[:,1]/1000), max(np.array(rec_vel)[:,1]/1000))
ax[1].plot(xx2, xx2, color='r')

ax[2].scatter(np.array(data_vel)[:,2]/1000, -np.array(rec_vel)[:,2]/1000, marker='.', color='k')
ax[2].set_ylabel('reconstructed vz')
ax[2].set_xlabel('data vz')
xx3 = np.linspace(min(-np.array(rec_vel)[:,2]/1000), max(-np.array(rec_vel)[:,2]/1000))
ax[2].plot(xx3, xx3, color='r')