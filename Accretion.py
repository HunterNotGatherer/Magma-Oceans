from MeltModelJupyter import Model
import os
import numpy as np
from scipy.interpolate import interp1d

Settings = [line.strip() for line in open('Acc_Settings.txt')] # Defualts commented below
SimInput = len(os.listdir(Settings[2]))
SavingData = int(Settings[4])        # 1
Compressed = int(Settings[6])        # 1
stepSize = int(Settings[8])          # 365*25
RunSteamModel = int(Settings[12])    # 1
InitMolten = int(Settings[14])       # 0
InitH2O = int(Settings[16])          # 410
InitEnt = int(Settings[18])          # 1100
Insolation = int(Settings[20])       # 1
Lm = float(Settings[22])             # 7.18e5
Cv = int(Settings[24])               # 1000
Koliv = float(Settings[26])          # 0.007 

dt = 24*3600 # Epoch in days, F in seconds; F*dt*stepSize = 1 cooling step
UpCoolLim = 365*100000000 # years - 100my upperbound

Init_Mass = float([line for line in open('Init_du.dat')][1])
Init_du = [[float(y) for y in x] for x in [line.strip().split(' ') for line in open('Init_du.dat')][3:]]

F_const = 340*0.75*(1-.76)*Insolation # venus like atm, young sun (Limaye et al 2018; DOI:10.1007/s11214-018-0525-2)
F_Solar = 340*0.75*(1-.275)*Insolation # 15-40% albedo for cloudless steam atm
F_bar = [float(x) for x in [line for line in open('F_bar.dat')][1].split(',')[:-1]] # ~285 w/m2 upper limit
barF = interp1d(np.arange(len(F_bar)), F_bar) #turning log fit into a continuous function -> useful for pressure < 25 bar

# linear interpolation
def e2depth(e,sumUlayer,Ulayer): 
    if(e>=sumUlayer[-1]*.9999): return .9999
    if(e<=0): return 0.0
    return (sum(sumUlayer[1:]<e) + (e-max((sumUlayer<e)*sumUlayer))/Ulayer[::-1][sum(sumUlayer[1:]<e)])/30

def depth2e(d,sumUlayer): 
    if(d>=.9999): return sumUlayer[-1]*.9999
    if(d<=0): return 0.0
    return sumUlayer[int(d*30)]+(d*30%1)*Ulayer[::-1][int(d*30)]

print('Running Accretion Model with',RunSteamModel*(str(InitH2O)+' ppm H2O,')+(1-RunSteamModel)*'constant atmosphere,',Insolation*'solar insolation,',InitMolten*'and initially molten'+(1-InitMolten)*('and '+str(InitEnt)+' initial entropy'),'\n',flush=True)
for sim in np.arange(1,SimInput+1):
    print('sim',sim,'\nImpact: ',end='')
# ------------ # data parsing and init - JimaInputs -> ['Name', ' Epoch', ' Angle', ' Mtotal', ' Impact_velocity', ' Gamma']
    data = np.array([[float(q) for q in x] for x in [(z[0].split(', ')[1:6]) for z in [x.split('\n') for x in open('Nakajima_Inputs/AccSim'+str(sim))][1:]]])
    JIsorted=data[data[:,0].argsort()] #sorted by impact date
        
# ------------ # Initialize arrays
    entropy=InitEnt # start w/ initial value
    meltDSim = np.zeros(int((JIsorted[-1][0]+UpCoolLim)/stepSize)) # depth for melt - Steam/Const cooling model
    atmBuildup = np.zeros(int((JIsorted[-1][0]+UpCoolLim)/stepSize)) # atm only tracked in F_steam cooling model
    mantleXw = np.ones(30)*InitH2O/1e6; meltXw = InitH2O/1e6 # mantle water resovior vs depth, init water %wt 
    atmW = d0 = d1 = 0 # P saturation above magma (bar), depth for previous and current step 
    if(InitMolten): d1 = .9999 
    
# ------------ # Running impacts    
    for k in np.arange(0,len(JIsorted)):  
# ------------ # Melt model (Nakajima 2021, doi:10.1016/j.epsl.2021.116983)            
        print(k,end=' ',flush=True)
        m = Model(Mtotal=JIsorted[k][2], gamma=JIsorted[k][4], vel=max(1,JIsorted[k][3]), entropy0=entropy, impact_angle=JIsorted[k][1]); resp = m.run_model()
        Vplanet = 4/3*np.pi*m.radiusPlanet**3 # volume planet in m^3
        surfaceArea = 4*np.pi*m.radiusPlanet**2 # m^2

# ------------ # E calcs to Heat, Melt, and increase T of magma above for each layer U(J/kg) * V * density = joules
        Udiff = (m.tmelt/1e5 - (m.du-m.du_gain)).T[0] # energy diff from initial du to melt T vs depth (*1e-5 J/Kg)
        mv = (4/3*np.pi)*((m.radiusPlanet*m.rr)**3 - (m.radiusPlanet*(m.rr-(m.rr[1] - m.rr[0])))**3)*m.rho; mvr=mv[::-1] # mass for each volume section
        Ulayer = (Udiff*1e5 + Lm)*mv + np.append((m.tmelt[:-1]-m.tmelt[1:]).T[0]/1000*Cv,0)*[sum(mvr[0:L+1]) for L in np.arange(len(mv))] # deltaT + Lm + heat to raise magama to meltT
        sumUlayer = np.array([sum(Ulayer[::-1][0:L]) for L in np.arange(len(Ulayer)+1)]) # summed from surface to core
        totalMw = sum(mv)*InitH2O/1e6

        # E of impact
        isoMeltE = depth2e(m.DP_PoolOcnConv[1,0]*1e3/((1-m.rr[0])*m.radiusPlanet),sumUlayer)
        if(m.DP_PoolOcnConv[1,0] == 0): isoMeltE = depth2e(m.DP_PoolOcnConv[2,0]*1e3/((1-m.rr[0])*m.radiusPlanet)*0.44,sumUlayer) # empirical factor averaging isoMelt/convMelt
        
# ------------ # Cooling Time until next impact 
        epoch = int(JIsorted[k][0]/stepSize)  #epoch in data is days (86400s) -> stepsize 
        if k < len(JIsorted)-1: coolingTime = int(JIsorted[k+1][0]/stepSize) - epoch 
        else: coolingTime = int(UpCoolLim/stepSize)
            
# ------------ # Steam Model - > E loss, and depth change, atm equilibrium
        if(RunSteamModel):  
            for step in np.arange(1,max(2,coolingTime)):
                if(step == 1): #Initializing depth, water conc of melt
                    isoMeltEStep = min(isoMeltE + depth2e(d1,sumUlayer),sumUlayer[-1]*.9999); d0 = d1 = e2depth(isoMeltEStep,sumUlayer,Ulayer); d0ndx = d0*len(mv)
                    meltDSim[epoch] = (1-m.rr[0])*m.radiusPlanet*d0; # Impact E + remaining E from previous MO
                    atmBuildup[epoch] = atmW
                    meltXw = sum((mantleXw*mvr)[0:int(d0ndx)],(mantleXw*mvr)[int(d0ndx)]*(d0ndx%1))/sum(mvr[0:int(d0ndx)],mvr[int(d0ndx)]*(d0ndx%1))
                    
                # net radiation out = E loss
                isoMeltEStep -= (barF(min(atmW,299)) - F_Solar)*surfaceArea*dt*stepSize 

                # if all energy is lost break and fill depth with 0's, atm stays constant
                if(isoMeltEStep <= 0): meltDSim[epoch+step:] = 0; atmBuildup[epoch+step:] = atmW; d1 = 0; break 

                # update depth
                d0 = d1; d1 = e2depth(isoMeltEStep,sumUlayer,Ulayer) # new % melt depth of mantle for this step
                meltDSim[epoch+step] = (1-m.rr[0])*m.radiusPlanet*d1
                
# ------------ # calculations for water content and outgassing of steam # Most of this comes from (Nikolaou et al., 2019)
                # mass calcs 
                d0ndx = d0*len(mv); d1ndx = d1*len(mv)
                newMeltM = sum(mvr[0:int(d1ndx)],mvr[int(d1ndx)]*(d1ndx%1))
                mantleMw = sum(mantleXw*mvr) - sum((mantleXw*mvr)[0:int(d0ndx)],(mantleXw*mvr)[int(d0ndx)]*(d0ndx%1)) # total solid mantle before cooling
                solidM = sum(mvr[0:int(d0ndx)],mvr[int(d0ndx)]*(d0ndx%1)) - newMeltM
                solidMw = solidM * meltXw * (0.01 + Koliv) # 1% of melt trapped in solid -> mass water in newly solidified chunk
                labileW = totalMw - mantleMw - solidMw # total water mass that can equilibrate 
                
                # Xwater calcs 
                XwSolvr = np.roots(np.array([(surfaceArea / ((6.67430*10**-11)*(m.Mmar*m.Mtotal)/m.radiusPlanet**2))/(6.8e-8)**(1/.7),0,0,newMeltM,0,0,0,0,0,0,-labileW]))
                XwRoot = (XwSolvr.real[(abs(XwSolvr.imag)<1e-5)*(XwSolvr>0)][0])**7 # post cooling Xh20      
                atmW = ((min(max(XwRoot,0), 1)/6.8e-8)**(1/.7))/1e5 # update psat and mass atm # (Caroll & Holloway 1994)
                atmBuildup[epoch+step] = atmW

                # mantle water content
                mantleXw[int(d0ndx)] = ((1-d0ndx%1)*(mantleXw*mvr)[int(d0ndx)] + min(d0ndx-d1ndx,d0ndx%1)*mvr[int(d0ndx)]*meltXw*(0.01 + Koliv))/((1-d0ndx%1)*mvr[int(d0ndx)] + min(d0ndx-d1ndx,d0ndx%1)*mvr[int(d0ndx)]) # this mantle chunk has fraction Xw of unmelted + cooled melt
                mantleXw[0:int(d1ndx)] = XwRoot # only fully melted chunks assigned melt Xw value
                if(int(d1ndx) < int(d0ndx)): mantleXw[int(d1ndx):int(d0ndx)] = meltXw * (0.01 + Koliv) # chunk only contains newly cooled solid and melt
                meltXw = XwRoot

# ------------ # updating remaining energy and entropy for mantle                
            if(np.sum(1-m.du_melt) > 0):
                EntSolid = np.sum(Vplanet*m.dv*(1-m.du_melt)*(m.du_gain.T*m.rho).T)*1e5/np.sum(Vplanet*(1-m.du_melt).T*m.dv.T*m.rho) #entropy of the unmelted mantle
                if(EntSolid > Init_du[:][min(10,int(m.Mtotal/Init_Mass*5-4))][-1]): Ent = 3160 
                else: Ent=1100+206*np.where(EntSolid < Init_du[:][min(10,int(m.Mtotal/Init_Mass*5-4))])[0][0]
            else: Ent=3160
            if(Ent>entropy and m.DP_PoolOcnConv[1,0]>0): entropy = Ent # update entropy if greater than Init_du.dat, conventional model will have no du gain

# ------------ # Cooling model with constant atm
        if(not RunSteamModel):
            constMeltE = min(isoMeltE + depth2e(d1,sumUlayer),sumUlayer[-1]*.9999)
            constECool = constMeltE-np.arange(0,max(1,coolingTime))*(F_const*surfaceArea*dt*stepSize) # array - energy after cooling for steps
            constMeltD = np.array([e2depth(i,sumUlayer,Ulayer) for i in constECool])
            meltDSim[epoch:epoch+coolingTime] = (1-m.rr[0])*m.radiusPlanet*constMeltD # Impact E + remaining E from previous MO
            d1 = constMeltD[-1]
            
# ------------ # saving data to file
    print('\n')
    if(SavingData): # steam model, constant model, atmosphere
        if(RunSteamModel):
            modelString='_'+str(InitH2O)+'w'+InitMolten*'M'+Insolation*'S'+str(InitEnt)+'e'
            lastMO=np.nonzero(meltDSim)[0][-1] + 80000 # dont save extra 0's, 2 myr buffer
            if(Compressed): np.savez_compressed('coolData/sim_'+str(sim)+modelString,meltDSim=meltDSim[:lastMO],atmBuildup=atmBuildup[:lastMO],mantleXw=mantleXw)
            elif(not Compressed): np.savetxt('coolData/sim_'+str(sim)+modelString+'.txt',[meltDSim[:lastMO],atmBuildup[:lastMO],mantleXw],'%.1f')
        elif(not RunSteamModel):
            modelString='_const_'+InitMolten*'M'+Insolation*'S'+str(InitEnt)+'e'
            lastMO=np.nonzero(meltDSim)[0][-1] + 80000 # dont save extra 0's, 2 myr buffer
            if(Compressed): np.savez_compressed('coolData/sim_'+str(sim)+modelString,meltDSim=meltDSim[:lastMO])
            elif(not Compressed): np.savetxt('coolData/sim_'+str(sim)+modelString+'.txt',meltDSim[:lastMO],'%.1f')