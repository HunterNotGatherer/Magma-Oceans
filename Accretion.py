from MeltModelJupyter import Model
import os
import numpy as np
from scipy.interpolate import interp1d

# General Settings
SavingData = True
Compressed = True # compress raw data from txt, raw takes much longer and generates much more data, not recommended
SimInput = len(os.listdir("Nakajima_Inputs")) # array of inputs to run, change to load files from folder automatically

# Model Parameters
RunSteamModel = True # Set to 1 for steam model or 0 for constant atm model
InitMolten = False 
initH2O = 205 # earth case 410ppm
initEnt = 1100 # 1100 for 300k to 3160 for 1800k
Insolation = True # young sun (75% of todays energy output) with 27.5% and 76% albedo for steam and const atm respectively

# Model Constants
init_Mass=float([line for line in open('init_du.dat')][1])
init_du=[[float(y) for y in x.split(' ')[:11]] for x in [line for line in open('init_du.dat')][3:]]
F_const = 160 - Insolation*(1-.76)*0.75*340 # 160 w/m2 venus like atm, 282 w/m2 for extreme steam atm
F_Solar=Insolation*0.75*(1-.275)*340 # 75% solar outout, 15-40% albedo
F_bar = [float(x) for x in [line for line in open('F_bar.dat')][1].split(',')[:-1]] 
barF=interp1d(np.arange(len(F_bar)), F_bar) #turning log fit into a continuous function -> useful for pressure < 25 bar
Koliv = 0.007 #partitioning coeff olivine used as whole mantle average -> retains more water -> less water in atm
Lm = 7.2e5 # J/Kg
Cv = 1000 # J/(Kg*K) 
dt = (24*3600) # Epoch in days, F in seconds; F*dt*stepSize = 1 step
stepSize = 365*25 # 25 years step size
UpCoolLim = 365*100000000 # years - 100my upperbound, change appending to not add so much whitespace

for sim in np.arange(1,SimInput+1):
    print("sim",sim,"\nImpact: ",end='')
# ------------ # data parsing and init - JimaInputs -> ['Name', ' Epoch', ' Angle', ' Mtotal', ' Impact_velocity', ' Gamma']
    data = np.array([[float(q) for q in x] for x in [(z[0].split(', ')[1:6]) for z in [x.split('\n') for x in open("Nakajima_Inputs/AccSim"+str(sim))][1:]]])
    JIsorted=data[data[:,0].argsort()] #sorted by impact date
    if(InitMolten): JIsorted = np.insert(JIsorted,0,[1,0,init_Mass,1,.2]).reshape(-1,5); initEnt = 3160
        
# ------------ # Initialize arrays
    E_rem = np.zeros(2) # energy remaining in melt
    atmBuildup = np.zeros(int((JIsorted[-1][0]+UpCoolLim)/stepSize)) # atm only tracked in F_steam cooling model
    meltDSim = np.zeros([2,int((JIsorted[-1][0]+UpCoolLim)/stepSize)]) # depth for melt - Steam/Const cooling model
    mantleXw = np.ones(100)*initH2O/1e6   # mantle water resovior vs depth - init water %wt 
    atmW = np.zeros(2) # Current atm, P saturation above magma (bar), mass Kg
    meltDStep = np.zeros(1) # Depth for current step - steam model

# ------------ # Running accretion model    
    entropy=initEnt #start each simulation at ~300K
    for k in np.arange(0,len(JIsorted)):
        print(k,end=' ',flush=True)

# ------------ # Melt model (Nakajima 2021, doi:10.1016/j.epsl.2021.116983)            
        m = Model(Mtotal=JIsorted[k][2], gamma=JIsorted[k][4], vel=max(1,JIsorted[k][3]), entropy0=entropy, impact_angle=JIsorted[k][1]); resp = m.run_model()

# ------------ # Melt Model variables are from core to surface, reveresed here for ease of use, other useful variables
        rr = m.rr[::-1]; dr = rr[0] - rr[1] # normed radius to 1 and incremental radius    
        Vplanet = 4/3*np.pi*m.radiusPlanet**3 # volume planet in m^3
        surfaceArea = 4*np.pi*m.radiusPlanet**2 # meters square
        V = [4/3*np.pi*(m.radiusPlanet**3*(rr[vol]**3-(rr[vol]-dr)**3)) for vol in np.arange(len(rr))] # volume of each radial section from surface to core
        MmeltTotal = (V*m.rho[::-1])  # mass melt at each depth surface to core
        
        rho = interp1d(np.linspace(0,1,30), m.rho[::-1]) ; r = (1-rr[-1])/100 # interpolation of density, finer resolution dr
        mv = np.zeros([100,2]) # mass, volume for different depths - finer resolution for mantle water
        for i in np.arange(100): 
            mv[i,1] = 4*np.pi/3*((m.radiusPlanet*(1-r*i))**3 - (m.radiusPlanet*(1-r*(i+1)))**3) # volume
            mv[i,0] = mv[i,1]*rho(i/100) # mass = V*density 
            
# ------------ # Finding Magma Ocean Depth using isostatic adjustment of melt Pool model
        Udiff = m.tmelt.T[0][::-1]/1e5 - (m.du-m.du_gain)[::-1,0] #energy diff from init_du to melt T vs depth (*1e-5 J/Kg)
        warmingMag = 0; heatMelt = (Udiff*1e5+Lm)*MmeltTotal # energy to melt each layer U(J/kg) * V * density = joules       
        Ulayer = np.zeros(len(rr))
        for i in np.arange(len(Ulayer)): # all energy deposited from surface down
            Ulayer[i] = heatMelt[i]+warmingMag 
            if (i < 29): warmingMag = ((m.tmelt.T[0][::-1][i+1]-m.tmelt.T[0][::-1][i])/1000 * np.sum(MmeltTotal[0:i]) * Cv) # Energy to warm all magma to new melt T - should use Cp? -> *1.3
        sumUlayer = np.array([sum(Ulayer[0:i]) for i in np.arange(len(Ulayer)+1)]) # summed from 0 to depth

        meltDStep = (m.DP_PoolOcnConv[1,0]*1000)/(dr*m.radiusPlanet) # initial depth
        isoMeltE = (meltDStep%1)*Ulayer[int(meltDStep)] + np.sum(Ulayer[0:int(meltDStep)]) # E calc from latent heat and melt T
        if(m.DP_PoolOcnConv[1,0] == 0):
            meltDStep = (m.DP_PoolOcnConv[2,0]*1000)/(dr*m.radiusPlanet)*0.44 # empirical factor averaging isoMelt/convMelt
            isoMeltE = (meltDStep%1)*Ulayer[int(meltDStep)] + np.sum(Ulayer[0:int(meltDStep)])
        
# ------------ # Cooling Times - F funciton of steam, fixed F
        epoch = int(JIsorted[k][0]/stepSize)  #epoch in data is days (86400s) -> stepsize 
        if k < len(JIsorted)-1:
            coolingTime = int(JIsorted[k+1][0]/stepSize) - epoch 
        else:
            coolingTime = int(UpCoolLim/stepSize)

# ------------ # Cooling model with constant atm
        if(1-RunSteamModel):
            constMeltE = isoMeltE + E_rem[1]; constMeltStep = epoch # initial energy and tracking total steps
            while(constMeltE > 0):
                meltIndx = min(29,np.where(constMeltE > sumUlayer)[0][-1])
                layerCoolStep = int((constMeltE - sumUlayer[meltIndx])/(F_const*surfaceArea*dt*stepSize)) #steps to cool this layer of melt
                if(layerCoolStep+constMeltStep-epoch < coolingTime): #only use if enough time before next impact
                    layerCoolE = constMeltE-np.arange(0,layerCoolStep+1)*(F_const*surfaceArea*dt*stepSize) # array - energy after cooling for steps
                    meltDSim[1,constMeltStep:layerCoolStep+constMeltStep+1] = (1 - rr[meltIndx] + (layerCoolE - sum(Ulayer[0:meltIndx]))/Ulayer[meltIndx]*dr) * m.radiusPlanet # convert energy to depth
                    constMeltStep += layerCoolStep # track total steps 
                    constMeltE = layerCoolE[-1] # update energy
                    if (constMeltE < F_const*surfaceArea*dt*stepSize+sumUlayer[meltIndx]): constMeltE = max(0,constMeltE-F_const*surfaceArea*dt*stepSize) # prevents small number error
                    if (constMeltE == 0): E_rem[1] = 0
                else:
                    layerCoolStep = int(coolingTime - (constMeltStep - epoch))
                    layerCoolE = constMeltE-np.arange(0,layerCoolStep+1)*(F_const*surfaceArea*dt*stepSize) # array - energy after cooling for steps
                    meltDSim[1,constMeltStep:layerCoolStep+constMeltStep+1] = (1 - rr[meltIndx] + (layerCoolE - sum(Ulayer[0:meltIndx]))/Ulayer[meltIndx]*dr) * m.radiusPlanet # convert energy to depth
                    E_rem[1] = layerCoolE[-1]; constMeltE = 0   # remaining energy carries into next impact
            meltDSim[1,epoch:] = m.radiusPlanet*(1-rr[-1])*(meltDSim[1,epoch:]>m.radiusPlanet*(1-rr[-1]))+(meltDSim[1,epoch:]<m.radiusPlanet*(1-rr[-1]))*meltDSim[1,epoch:]

# ------------ # Initializing depth, water conc of melt, data arrays    
        if(RunSteamModel):
            d0 = min(29/30,meltDStep*dr/(1-rr[-1])) # fractional melt depth of mantle
            XwRoot = sum([mantleXw[i]*mv[i,0] for i in np.arange(int(d0*100))],mantleXw[int(d0*100)]*mv[int(d0*100),0]*(100*d0%1))/sum(mv[0:int(d0*100),0],mv[int(d0*100),0]*(100*d0%1)) # total water in melt (Xi*Mi/Mtotal) 

            isoMeltEStep = np.float64(isoMeltE + E_rem[0]) # Energy -> steam model + remaining energy from previous MO

            for i in np.arange(len(Ulayer)): # initialize depth
                if (isoMeltEStep >= sum(Ulayer[0:i])): 
                    meltDSim[0,epoch] = min((1 - rr[i] + (isoMeltEStep-sum(Ulayer[0:i]))/Ulayer[min(i,29)]*dr),1-rr[-1]) * m.radiusPlanet  
            atmBuildup[epoch] = atmW[0]

# ------------ # cooling step, E loss, and depth change
            for step in np.arange(1,max(2,coolingTime+1)):
                if(np.sum(isoMeltEStep) == 0): 
                    meltDSim[0,epoch+step:] = 0; atmBuildup[epoch+step:] = atmW[0]
                    break # break and fill depth with 0's if all energy is lost to cooling, atm stays constant

                isoMeltEStep = max(0, isoMeltEStep - (barF(atmW[0]) - F_Solar)*surfaceArea*dt*stepSize) # net radiation out  

                for i in np.arange(len(Ulayer)): # Heat magma, melt layer, convert energy to depth
                    if (isoMeltEStep >= sum(Ulayer[0:i])): 
                        meltDStep = min((1 - rr[i] + (isoMeltEStep-sum(Ulayer[0:i]))/Ulayer[min(i,29)]*dr),1-rr[-1]) * m.radiusPlanet  

# ------------ # calculations for water content and outgassing of steam
                #depth calcs
                d0 = min(29/30,meltDSim[0,epoch+step-1]/m.radiusPlanet/(1-rr[-1])) # fractional melt depth of mantle
                d1 = min(29/30,meltDStep/m.radiusPlanet/(1-rr[-1])) 

                # mass calcs 
                solidM = rho((d0+d1)/2)*4/3*np.pi*((m.radiusPlanet - meltDStep)**3 - (m.radiusPlanet - meltDSim[0,epoch+step-1])**3) # density * volume of solidified melt
                solidW = solidM * XwRoot*(0.01 + Koliv) # 1% of melt trapped in solid -> mass water in newly solidified chunk
                newMeltM = sum(mv[0:int(d1*100),0]) + (100*d1%1)*mv[int(d1*100),0]
                unMelted = max(0, (mv[int(d0*100),0] - (newMeltM - sum(mv[0:int(d0*100),0]))) - solidM) # total solid mass - cooled solid = never melted

                # Xwater calcs # Most of this comes from (Nikolaou et al., 2019)
                labileW = atmW[1] + sum([mantleXw[i]*mv[i,0] for i in np.arange(int(d0*100))]) + XwRoot*mv[int(d0*100),0]*(100*d0%1) #integrate water w/ depth + atm
                XwSolvr = np.roots(np.array([(surfaceArea / ((6.67430*10**-11)*(m.Mmar*m.Mtotal)/m.radiusPlanet**2))/(6.8e-8)**(1/.7), 0, 0, newMeltM, 0,0,0,0,0,0,solidW-labileW]))
                XwRoot = (XwSolvr.real[(abs(XwSolvr.imag)<1e-5)*(XwSolvr>0)][0])**7      # post cooling Xh20      

                # atm calcs
                atmW[0] = ((min(max(XwRoot,0), 1)/6.8e-8)**(1/.7))/1e5 # update psat and mass atm # (Caroll & Holloway 1994)
                atmW[1] = atmW[0]*1e5 * surfaceArea/((6.67430*10**-11)*(m.Mmar*m.Mtotal)/m.radiusPlanet**2) # Kg h2o in atmosphere if saturated

                # mantle water content
                mantleXw[0:np.int(d1*100)] = XwRoot # only fully melted chunks assigned melt Xw value
                mantleXw[np.int(d0*100)] = (unMelted*mantleXw[np.int(d0*100)] + min(1,(mv[int(d0),0] - unMelted)/(solidM+1))*solidW)/(1+unMelted + min(solidM,mv[int(d0),0] - unMelted))  # this mantile chunk has fractional Xw of unmelted + cooled melt
                if(int(d1*100) < int(d0*100)): 
                    mantleXw[np.int(d1*100)] = solidW/solidM # chunk only contains newly cooled solid and melt
                
                #saving resultant atm and melt depth
                atmBuildup[epoch+step] = atmW[0]
                meltDSim[0,epoch+step] = meltDStep       

# ------------ # updating remaining energy and entropy for mantle                
            E_rem[0] = isoMeltEStep #Energy for existing magma ocean (successive impacts only)
            if(np.sum(1-m.du_melt) > 0):
                EntSolid = np.sum(Vplanet*m.dv*(1-m.du_melt)*(m.du_gain.T*m.rho).T)*1e5/np.sum(Vplanet*(1-m.du_melt).T*m.dv.T*m.rho) #entropy of the unmelted mantle
                if(EntSolid > init_du[:][min(10,int(m.Mtotal/init_Mass*5-4))][-1]): Ent = 3160 
                else: Ent=1100+206*np.where(EntSolid < init_du[:][min(10,int(m.Mtotal/init_Mass*5-4))])[0][0]
            else: Ent=3160
            if(Ent>entropy and m.DP_PoolOcnConv[1,0]>0): entropy = Ent # update entropy if greater than init_du.dat, conventional model will have no du gain
    
# ------------ # saving data to file
    print('\n')
    if(SavingData): # steam model, constant model, atmosphere
        if(Compressed):
            if(RunSteamModel):
                np.savez_compressed('coolData/sim_'+str(sim)+'_'+str(initH2O)+'w'+InitMolten*'M'+Insolation*'S'+str(initEnt)+'e',meltDSim=meltDSim[0],atmBuildup=atmBuildup)
            elif(RunSteamModel == 0):
                np.savez_compressed('coolData/sim_'+str(sim)+'_'+str(initH2O)+'w'+InitMolten*'M'+Insolation*'S'+str(initEnt)+'e',meltDSim=meltDSim[1],atmBuildup=atmBuildup)
        elif(Compressed==False): 
            if(RunSteamModel):
                np.savetxt('coolData/sim_'+str(sim)+'_'+str(initH2O)+'w'+InitMolten*'M'+Insolation*'S'+str(initEnt)+'e.txt',[meltDSim[0,:],atmBuildup[:]],'%.1f')
            elif(RunSteamModel == 0):
                np.savetxt('coolData/sim_'+str(sim)+'_const_'+InitMolten*'M'+Insolation*'S'+str(initEnt)+'e.txt',[meltDSim[1,:],atmBuildup[:]],'%.1f')
        