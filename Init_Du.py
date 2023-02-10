from MeltModelJupyter import Model
import numpy as np

# Finding initial du for different mass/entropy combinations
init_du = np.zeros(shape=(11,11))
massT = np.zeros(shape=(11))
init_Mass = 3.1023287

for mass in np.arange(11):
    for ent in np.arange(11):
        initM = Model(Mtotal=init_Mass*(1+mass/5), gamma=1e-9, vel=1.0, entropy0=(1100+206*ent), impact_angle=90, outputfigurename="init.png", use_tex=False); initRun=initM.run_model()
        mantM = np.sum(initM.radiusPlanet**3*initM.dv.T*initM.rho) # mass of mantle from density * volume *
        init_du[ent][mass] = np.sum(initM.radiusPlanet**3*initM.dv*(initM.du.T*initM.rho).T)*1e5/mantM

with open ('init_du.dat','w') as f:
    print('init_Mass = ', init_Mass, file=f)
    print('init_du = ',[[y for y in x] for x in init_du],end='',file=f); 