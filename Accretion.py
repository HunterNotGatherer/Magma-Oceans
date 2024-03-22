from MeltModelJupyter import Model
import os
import numpy as np
from scipy.interpolate import interp1d

Settings = [line.strip() for line in open('Acc_Settings.txt')] # Defualts commented below
SimInput = len(os.listdir(Settings[2]))
SavingData = int(Settings[4])        # 1
Compressed = int(Settings[6])        # 1
stepSize = int(Settings[8])          # 365*25 days

RunSteamModel = int(Settings[12])    # 1
InitMolten = int(Settings[14])       # 0
InitH2O = int(Settings[16])          # 410 ppm
InitEnt = int(Settings[18])          # 1100 J/kg/K
F_const = float(Settings[20])        # 120w/m2 for  CO2 rich, wet atm (Halevy 2009; doi:10.1029/2009JD011915)
Lm = float(Settings[22])             # 7.18e5 J/kg
Cv = int(Settings[24])               # 1000 J/kg
Koliv = float(Settings[26])          # 0.007 

F_Solar = 60+125*RunSteamModel       # 340*0.75 = 255 for Young Sun, 15-40% albedo for cloudless steam atm -> 185, 76% albedo for venus like (Limaye; doi.org/10.1007/s11214-018-0525-2) -> 60
F_bar = [float(x) for x in [line for line in open('F_bar.dat')][1].split(',')[:-1]] # 285 w/m2 upper limit
barF = interp1d(np.arange(len(F_bar)), F_bar) #turning data into a continuous function -> useful for < 25 bar
dt = 24*3600 # seconds in a day; Epoch in days, Flux in seconds
UpCoolLim = 365*400000000 # 40 Myr upperbound

# linear interpolation functions for melt energy
def e2depth(e,sumULayer,ULayer): # surface to core
    if(e>=sumULayer[-1]*.9999): return .9999
    if(e<=0): return 0.0
    return (sum(sumULayer[1:]<e) + (e-max((sumULayer<e)*sumULayer))/ULayer[sum(sumULayer[1:]<e)])/30

def depth2e(d,sumULayer): # surface to core
    if(d>=.9999): return sumULayer[-1]*.9999
    if(d<=0): return 0.0
    return sumULayer[int(d*30)]+(d*30%1)*ULayer[int(d*30)]

print('Running Accretion Model with',RunSteamModel*(str(InitH2O)+' ppm H2O,')+(1-RunSteamModel)*(str(int(F_const))+' w/m^2 OLR,'),
      F_Solar,'w/m^2 insolation, and',InitMolten*'initially molten'+(1-InitMolten)*(str(InitEnt)+' initial entropy'),'\n',flush=True)

for sim in np.arange(1,SimInput+1):
    print('Sim',sim,'\nImpact: ',end='')
# ------------ # data parsing and init; input format -> ['Name', ' Epoch', ' Angle', ' Mtotal', ' Impact_velocity', ' Gamma']
    data = np.array([[float(q) for q in x] for x in [(z[0].split(', ')[1:6]) for z in [x.split('\n') for x in open('Nakajima_Inputs/AccSim'+str(sim))][1:]]])
    dataSorted = data[data[:,0].argsort()] #sorted by impact date
        
# ------------ # Initialize arrays
    meltDSim = np.zeros(int((dataSorted[-1][0]+UpCoolLim)/stepSize)) # depth of melt
    atmBuildup = np.zeros(int((dataSorted[-1][0]+UpCoolLim)/stepSize)) # depth of melt
    mantleXw = np.ones(30)*InitH2O/1e6; meltXw = InitH2O/1e6; du_mantle = 0 # mantle water %wt, surface to core - Steam model only
    atmW = d0 = d1 = 0 # saturation pressure of H2O above magma (bar) and init depth
    if(InitMolten): d1 = .9999; InitEnt = 3160; dataSorted = np.insert(dataSorted,0,[0,0,3.1023287,1,1e-3]).reshape(-1,5) # dummy impact to record initial molten state and begin cooling
    
# ------------ # Running impacts    
    for k in np.arange(0,len(dataSorted)):  
# ------------ # Melt model (Nakajima 2021, doi:10.1016/j.epsl.2021.116983)            
        if(k < len(dataSorted)-InitMolten): print(k+1,end=' ',flush=True) # prevents printing of dummy impact...
        m = Model(Mtotal=dataSorted[k][2], gamma=dataSorted[k][4], vel=max(1,dataSorted[k][3]), entropy0=InitEnt, impact_angle=dataSorted[k][1]); resp = m.run_model()
        surfaceArea = 4*np.pi*m.radiusPlanet**2 # m^2
        mv = (4/3*np.pi*((m.radiusPlanet*m.rr)**3 - (m.radiusPlanet*(m.rr-(m.rr[1] - m.rr[0])))**3)*m.rho)[::-1] # mass for each volume section, surface to core
        
# ------------ # update Mantle Energy from impact; find E of melt        
        if(k==0): du_mantle = m.du*1e5*(m.radiusPlanet**3*m.dv.T*m.rho).T
        else: du_mantle += m.du_gain*1e5*(m.radiusPlanet**3*m.dv.T*m.rho).T

        isMolten = (du_mantle > m.tmelt*(m.radiusPlanet**3*m.dv.T*m.rho).T) 
        meltE = np.sum((du_mantle-m.tmelt*(m.radiusPlanet**3*m.dv.T*m.rho).T)*isMolten)
        du_mantle = du_mantle*(1-isMolten) + m.tmelt*(m.radiusPlanet**3*m.dv.T*m.rho).T*isMolten 
                
# ------------ # E calcs to Melt, and increase T of magma for each layer U(J/kg) * V * density = joules        
        ULayer=sum((Lm + (5e6-m.tmelt[-1][0]))*(m.radiusPlanet**3*m.dv.T*m.rho)) # 5000K surface max T + Latent Heat      
        sumULayer = np.array([sum(ULayer[0:L]) for L in range(len(ULayer)+1)]) # summed from surface to core        
        
# ------------ # Cooling Time until next impact 
        epoch = int(dataSorted[k][0]/stepSize)  #epoch in data is days (86400s) -> stepsize 
        if (k < len(dataSorted)-1): coolingTime = int(dataSorted[k+1][0]/stepSize) - epoch 
        else: coolingTime = int(UpCoolLim/stepSize)

# ------------ # Steam Model - > E loss, and depth change, atm equilibrium
        if(RunSteamModel):  
            isoMeltE = min(meltE + depth2e(d1,sumULayer),sumULayer[-1]*.9999)
            d0 = d1 = e2depth(isoMeltE,sumULayer,ULayer); d0ndx = d0*len(mv)
            meltDSim[epoch] = d0 *(1-m.rr[0])*m.radiusPlanet/1000
            atmBuildup[epoch] = atmW
            if(d0>0): meltXw = sum((mantleXw*mv)[0:int(d0ndx)],(mantleXw*mv)[int(d0ndx)]*(d0ndx%1))/sum(mv[0:int(d0ndx)],mv[int(d0ndx)]*(d0ndx%1))

            for step in np.arange(1,max(2,coolingTime)):    
                if(m.DP_PoolOcnConv[1,0]>0 and d0>0):
                    PrStep=m.DP_PoolOcnConv[1,1]/m.DP_PoolOcnConv[1,0]*d0*(1-m.rr[0])*m.radiusPlanet/1000 
                    Tdelta=m.tmelt.T[0][::-1][int(d0ndx)]/1000 - 1645
                    kCoeff=0.0194*PrStep+1.38 # approximated from doi:10.1029/2021GL093806
                    if(Tdelta <= 955): visc=10**(PrStep*0.0675-2.258) # Viscosity data approximated from doi:10.1038/s41467-022-35171-y
                    if(Tdelta >= 1855): visc=10**(PrStep*0.0216-2.952)
                    if(Tdelta > 955 and Tdelta < 1855): visc=10**(PrStep*0.0413-2.763)
                    Rayleigh=9.8*5e-5*Tdelta*((d0*(1-m.rr[0])*m.radiusPlanet)**3)/(kCoeff/m.rho[::-1][int(d0ndx)]/Cv)/(visc/m.rho[::-1][int(d0ndx)])
                    Prandtl=visc/(kCoeff/m.rho[::-1][int(d0ndx)]/Cv)/m.rho[::-1][int(d0ndx)]
                    # Rayleigh and F_hard calculated from doi:10.1016/B978-0-444-53802-4.00155-X, doi:10.1103/PhysRevA.42.3650 
                    F_hard=0.22*kCoeff*Tdelta*Rayleigh**(2/7)*Prandtl**(-1/7)*(2*np.pi*m.radiusPlanet/(d0*(1-m.rr[0])*m.radiusPlanet))**(-3/7)/(d0*(1-m.rr[0])*m.radiusPlanet)
                else: F_hard=1e6
                    
                # net radiation; if all energy is lost break and fill depth with 0's, atm stays constant
                isoMeltE += (F_Solar - min(F_hard,barF(min(atmW,299))))*surfaceArea*dt*stepSize 
                if(isoMeltE <= 0): meltDSim[epoch+step:] = 0; atmBuildup[epoch+step:] = atmW; d1 = 0; break 

                # update depth
                d0 = d1; d1 = e2depth(isoMeltE,sumULayer,ULayer) # new % melt depth of mantle for this step
                meltDSim[epoch+step] = d1 *(1-m.rr[0])*m.radiusPlanet/1000
                
# ------------ # calculations for water content and outgassing of steam # following (Nikolaou et al., 2019)
                # mass calcs 
                d0ndx = d0*len(mv); d1ndx = d1*len(mv)
                newMeltM = sum(mv[0:int(d1ndx)],mv[int(d1ndx)]*(d1ndx%1))
                mantleMw = sum(mantleXw*mv) - sum((mantleXw*mv)[0:int(d0ndx)],(mantleXw*mv)[int(d0ndx)]*(d0ndx%1)) # total solid mantle before cooling
                solidM = sum(mv[0:int(d0ndx)],mv[int(d0ndx)]*(d0ndx%1)) - newMeltM
                solidMw = solidM * meltXw * (0.01 + Koliv) # 1% of melt trapped in solid -> mass water in newly solidified chunk
                labileW = sum(mv)*InitH2O/1e6 - mantleMw - solidMw # total water mass that can equilibrate 
                
                # Xwater calcs 
                XwSolvr = np.roots(np.array([(surfaceArea / ((6.67430*10**-11)*(m.Mmar*m.Mtotal)/m.radiusPlanet**2))/(6.8e-8)**(1/.7),0,0,newMeltM,0,0,0,0,0,0,-labileW]))
                XwRoot = (XwSolvr.real[(abs(XwSolvr.imag)<1e-5)*(XwSolvr>0)][0])**7 # post cooling Xh20      
                atmW = ((min(max(XwRoot,0), 1)/6.8e-8)**(1/.7))/1e5 # update psat and mass atm # (Caroll & Holloway 1994)
                atmBuildup[epoch+step] = atmW

                # mantle water content
                mantleXw[int(d0ndx)] = ((1-d0ndx%1)*(mantleXw*mv)[int(d0ndx)] + min(d0ndx-d1ndx,d0ndx%1)*mv[int(d0ndx)]*meltXw*(0.01 + Koliv))/((1-d0ndx%1)*mv[int(d0ndx)] + min(d0ndx-d1ndx,d0ndx%1)*mv[int(d0ndx)]) # this mantle chunk has fraction Xw of unmelted + cooled melt
                mantleXw[0:int(d1ndx)] = XwRoot # only fully melted chunks assigned melt Xw value
                if(int(d1ndx) < int(d0ndx)): mantleXw[int(d1ndx):int(d0ndx)] = meltXw * (0.01 + Koliv) # chunk only contains newly cooled solid and melt
                meltXw = XwRoot
                
# ------------ # Cooling model with constant atm
        if(not RunSteamModel):
            constMeltE = min(meltE + depth2e(d1,sumULayer),sumULayer[-1]*.9999); constMeltD = np.zeros(max(1,coolingTime))
            constECool = constMeltE+np.arange(max(1,coolingTime))*(F_Solar - F_const)*surfaceArea*dt*stepSize 
            for i in range(len(ULayer)):
                constMeltD += (((constECool/ULayer[i])>=1) + (constECool/ULayer[i])*((constECool/ULayer[i])<1)*((constECool/ULayer[i])>0))/30
                constECool -= ULayer[i]
                
            meltDSim[epoch:epoch+coolingTime] = constMeltD *(1-m.rr[0])*m.radiusPlanet/1000
            d1 = constMeltD[-1]

# ------------ # saving data to file
    print('\n')
    if(SavingData): 
        lastMO=np.nonzero(meltDSim)[0][-1] + 80000 # dont save extra 0's, 2 myr buffer
        if(RunSteamModel):
            modelString='_'+str(InitH2O)+'w'+str(int(F_Solar))+'s'+str(InitEnt)+'e'+InitMolten*'M'
            if(Compressed): np.savez_compressed('coolData/sim_'+str(sim)+modelString,meltDSim=meltDSim[:lastMO],atmBuildup=atmBuildup[:lastMO],mantleXw=mantleXw)
            elif(not Compressed): np.savetxt('coolData/sim_'+str(sim)+modelString+'.txt',[meltDSim[:lastMO],atmBuildup[:lastMO],mantleXw],'%.4f')
        elif(not RunSteamModel):
            modelString='_'+str(int(F_const))+'c'+str(int(F_Solar))+'s'+str(InitEnt)+'e'+InitMolten*'M'
            if(Compressed): np.savez_compressed('coolData/sim_'+str(sim)+modelString,meltDSim=meltDSim[:lastMO])
            elif(not Compressed): np.savetxt('coolData/sim_'+str(sim)+modelString+'.txt',meltDSim[:lastMO],'%.4f')