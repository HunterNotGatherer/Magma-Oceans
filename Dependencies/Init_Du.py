from MeltModelJupyter import Model
import numpy as np

# Finding initial du for different mass/entropy combinations
init_du = np.zeros(shape=(11,11))
massT = np.zeros(shape=(11))
init_Mass = 3.1023287

for mass in np.arange(11):
    print('Mass: ', init_Mass*(1+mass/5), '\nEntropy: ',end='')
    for ent in np.arange(11):
        print(ent*206+1100,end=' ',flush=True)
        initM = Model(Mtotal=init_Mass*(1+mass/5), gamma=1e-9, vel=1.0, entropy0=(1100+206*ent), impact_angle=90); initRun=initM.run_model()
        mantM = np.sum(initM.radiusPlanet**3*initM.dv.T*initM.rho) # mass of mantle from density * volume *
        init_du[ent][mass] = np.sum(initM.radiusPlanet**3*initM.dv*(initM.du.T*initM.rho).T)*1e5/mantM
    print('\n')

with open ('init_du.dat','w') as f:
    print('# Initial Mass\n',init_Mass,'\n# Initial du [entropy,mass] for 1-3 init mass and 1100-3160 entropy',file = f)
    for x in init_du:
        [print(np.round(y,2), end=" ", file = f) for y in x]; print('', file=f)