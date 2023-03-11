from MeltModelJupyter import Model
import numpy as np

# Calculating initial du for different mass/entropy combinations
Init_du = np.zeros(shape=(11,11))
massT = np.zeros(shape=(11))
Init_Mass = 3.1023287

for mass in np.arange(11):
    print('Mass:', Init_Mass*(1+mass/5), '\nEntropy:',end=' ')
    for ent in np.arange(11):
        print(ent*206+1100,end=' ',flush=True)
        initM = Model(Mtotal=Init_Mass*(1+mass/5), gamma=1e-9, vel=1.0, entropy0=(1100+206*ent), impact_angle=90); initRun=initM.run_model()
        mantM = np.sum(initM.radiusPlanet**3*initM.dv.T*initM.rho) # mass of mantle from density * volume *
        Init_du[ent][mass] = np.sum(initM.radiusPlanet**3*initM.dv*(initM.du.T*initM.rho).T)*1e5/mantM
    print('\n')

with open ('Init_du.dat','w') as f:
    print('Init_Mass\n',str(Init_Mass),'\nInit_du # [Ent][Mass]',file=f)
    [[print(round(y,2), end=' ',file=f) for y in x] and print('',file=f) for x in Init_du]