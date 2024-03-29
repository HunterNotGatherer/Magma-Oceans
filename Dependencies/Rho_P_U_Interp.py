import numpy as np
from scipy.interpolate import interp1d

n = 50 # of points for smoothed data

# ------------------------ # Reading S0=1100 data # ------------------------ #
rho_P = [line.split() for line in open('rho_u_S1100.dat')]
rho_input = P_input = U_input = np.zeros(shape=(0,0))  # density, pressure, internal energy model
for m in range(1, len(rho_P)):  # data from rho_u_S1100.dat
    rho_input = np.append(rho_input, float(rho_P[m][0]))
    P_input = np.append(P_input, float(rho_P[m][1]) * 1e9)  # converting from GPa to Pa.
    U_input = np.append(U_input, float(rho_P[m][2]))   
    
# ------------------------ # Reading S0=3160 data # ------------------------ #
rho_P2 = [line.split() for line in open('rho_u_S3160.dat')]
rho_input2 = P_input2 = U_input2 = np.zeros(shape=(0,0))  # density, pressure, internal energy model
for m2 in range(1, len(rho_P2)):  # data from rho_u_S3160.dat
    rho_input2 = np.append(rho_input2, float(rho_P2[m2][0]))
    P_input2 = np.append(P_input2, float(rho_P2[m2][1]) * 1e9)  # converting from GPa to Pa.
    U_input2 = np.append(U_input2, float(rho_P2[m2][2]))

# ------------------------ # Interpolating U # ------------------------ #
rho_new=10**np.linspace(np.log10(rho_input[0]),np.log10(rho_input[-1]),n) # S0=1100 data 
yrP = interp1d(rho_input,P_input); P_new = yrP(rho_new)
yPU = interp1d(P_input,U_input); U_new = yPU(P_new)

rho_new2=10**np.linspace(np.log10(rho_input2[0]),np.log10(rho_input2[-1]),n) # S0=3160 data
yrP2 = interp1d(rho_input2,P_input2); P_new2 = yrP2(rho_new2)
yPU2 = interp1d(P_input2,U_input2); U_new2 = yPU2(P_new2)

TempInterp = 11 # number of points to interpolate including boundaries
U_all = np.array([np.linspace((U_new[i]),(U_new2[i]),TempInterp) for i in np.arange(len(U_new))]).T #linear spacing between data

# ------------------------ # Adjusting Surface from 88K-1187K to 300K-1700K # ------------------------ #
U_fixed = U_all + np.linspace(0,np.linspace(3e5,17e5,TempInterp)-U_all[:,0],50)[::-1].T

rho_u_data = np.append(rho_new,P_new/1e9)
rho_u_data = np.append(rho_u_data,U_fixed)

with open('rho_u_S9999.dat', 'w') as f:
    print('# rho (kg/m^3) P (GPa)     u: S=1100 J/kg/K  S= +206 J/kg/K', file=f)
    dat = rho_u_data.reshape(2+TempInterp,-1).T
    for s in np.arange(len(dat)):
        strings = ['%.8e' % number for number in dat[s]]
        print(' '.join(strings), file=f)