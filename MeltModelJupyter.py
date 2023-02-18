import numpy as np
import sys
import scipy.optimize as op
from scipy.interpolate import interp1d

debugPrint = 0 # 0 no printing, 1 default, 2 verbose

# TODO
# magma ocean depth is fixed at psi = 0 
# L polynomials 5 and 6 - not fixed as the model was fit with bad params, >8000K diff in temps observed
# Fix Dv calculation ( > 100% volume)

#Changelog
# Em changed to 9e6
# Pool depth capture angle reduced to pi/4
# deleted plotting

class Model:

    def __init__(self, Mtotal=2.0, gamma=0.5, vel=2.0, entropy0=1100, impact_angle=90):
        self.Mmar = 6.4171e23  # mass of Mars
        self.R0 = 1.5717e6  # impactor radius
        self.M0 = 6.39e22  # scaling coefficient
        self.a0 = 0.3412  # planetary mass-radius relationship
        self.a1 = -8.90e-3  # planetary mass-radius relationship
        self.a2 = 9.1442e-4  # planetary mass-radius relationship
        self.a3 = -7.4332e-5  # planetary mass-radius relationship
        self.GG = 6.67408e-11  # gravitational constant
        self.impact_angle = float(impact_angle)  # impactor impact angle with target

        # default values
        self.Mtotal = Mtotal  # total mass
        self.gamma = gamma  # impactor-to-total-mass ratio
        self.vel = vel  # impact velocity normalized by the escape velocity (this means that vimp = vesc(i.e. v_inf = 0))
        self.entropy0 = entropy0  # initial entropy (assuming adiabatic, dS/dr = 0)
        self.radiusPlanet = 0
        # default value of 1100 J/K/kg represents an adiabatic mantle with a surface temperature of 300K
        # default value of 3160 J/K/kg represents an adiabatic mantle with a surface temperature of 2000K
        self.check_model_integrity()
        
        self.entropyfile = 'rho_u_S9999.dat' 
#        self.entropyfile = 'rho_u_S{}.dat'.format(self.entropy0)
#        self.outputfigurename = outputfigurename  # output figure name

        self.Mt = (1.0 - self.gamma) * self.Mtotal  # target mass

        self.Mi = self.gamma * self.Mtotal  # impactor mass

        self.EM = 9e6  # specific energy needed in order for melting to occur
        self.latent_heat = 7.18e5  # latent heat

        # relationship between rho-P assuming S0=3160 J/K/kg. We also assume that rho-P structure is the same at S0=1100 J/K/kg.
        self.rho_P = [line.split() for line in open(self.entropyfile)]  
        self.levels = np.arange(-2, 100, 2)
        self.vmin_value = 5
        self.vmax_value = 40
        # --- end of input data ---

        # calculating rho-P relationship of planetary interior assuming that the mantle has a constant entropy.
        self.rho_input = np.zeros(shape=(0, 0))
        self.P_input = np.zeros(shape=(0, 0))
        self.U_input = np.zeros(shape=(0, 0))  # density, pressure, internal energy model

        entropyN = (self.entropy0-1100)//206 #reading corresponding entropy values
        for m in range(1, len(self.rho_P)):
            self.rho_input = np.append(self.rho_input, float(self.rho_P[m][0]))
            self.P_input = np.append(self.P_input, float(self.rho_P[m][1]) * 1e9)  # converting from GPa to Pa.
            self.U_input = np.append(self.U_input, float(self.rho_P[m][2+entropyN]))

        self.rho_P_function = interp1d(self.P_input, self.rho_input)  # generating interpolation
        self.rho_U_function = interp1d(self.rho_input, self.U_input)
        
        self.theta_angle = None
        self.rr = None
        self.du = None
        self.du_gain = None
        self.du_melt = None
        self.du_gain_melt = None
        self.tmelt = None
        self.dv = None
        self.rho = None
        self.DP_PoolOcnConv = None
        self.check_model_integrity()

    def check_model_integrity(self):
        if float(self.entropy0) < 1100.0 or float(self.entropy0) > 3160.0:
            print("Please choose an entropy (entropy0) value between 1100 and 3160.")
            sys.exit(1)
        if float(self.impact_angle) < 0 or float(self.impact_angle) > 90:
            print("Please choose an impact angle between 0 and 90 degrees.")
            sys.exit(1)

    # legendre polynomial functions, solutions to a legendre DE
    def __legendre(self, n, x):
        if   n == 0: return 1
        elif n == 1: return x
        elif n == 2: return (3 * x ** 2.0 - 1.0)/2.0
        elif n == 3: return (5 * x ** 3.0 - 3 * x)/2.0
        elif n == 4: return (35 * x ** 4.0 - 30 * x ** 2.0 + 3)/8.0
        elif n == 5: return (63 * x ** 5.0 - 70 * x ** 3.0 - 15 * x)/8.0 # +15x not -15x
        elif n == 6: return (231 * x ** 6.0 - 315 * x ** 4.0 + 105 * x ** 2.0 - 5)/8.0 # 1/16 not 1/8

    # mass-radius relationship (see Section S.1.1. in our paper)
    def __radius(self, mass):
        lnMM0 = np.log(mass / self.M0)
        gamma = self.a0 + self.a1 * lnMM0 + self.a2 * lnMM0 ** 2 + self.a3 * lnMM0 ** 3
        return self.R0 * (mass / self.M0) ** gamma
    
    # pressure calculation. If an input pressure is smaller than 0, the minium density is returned
    def __compute_density(self, P):
        if abs(P) < self.P_input[0]:
            return self.rho_input[0]
        else:
            return self.rho_P_function(abs(P))

    # compute pressure at radius = Rmelt
    def __compute_pressure(self, Mt, Rmelt):
        Rt = self.__radius(Mt)
        Rmelt = Rmelt * Rt

        if Rmelt == 1.0: return 0.0

        dr = (Rt - Rmelt) / 100.0

        P = 0.0
        r = Rt
        Mass = Mt

        while (r > Rmelt):
            rho = self.__compute_density(P)
            P = P + rho * self.GG * Mass / r ** 2.0 * dr
            Mass = Mass - 4 * np.pi * rho * r ** 2.0 * dr
            r = r - dr

        if Rmelt == 1.0:
            return 0.0
        else:
            return P * 1e-9

    # --- end of computing the structure of a planet

    def run_model(self):
        
        self.check_model_integrity()

        Mt = self.Mt * self.Mmar  # recalculate target mass
        Mi = self.Mi * self.Mmar  # recalculate impactor mass
        Rt = self.__radius(Mt)  # calculate radius of the target using mass-radius relationships
        Ri = self.__radius(Mi)  # calculate radius of the impactor using mass-radius relationships
        Rti = self.__radius(Mt + Mi)  # calculate radius of the target + impactor (perfectly merged body) using mass-radius relationships

        vesc = np.sqrt(2.0 * self.GG * (Mt + Mi) / (Rt + Ri))  # calculate escape velocity
        ratio = Mi / Mt  # impactor-to-target mass ratio (not the same as gamma!)
        ang = self.impact_angle / 180.0 * np.pi  # convert impact angle degrees to radians
        targetmassfraction = Mt / (Mt + Mi)  # mass fraction of the target relative to total mass (target + impactor)

        # potential energy
        dPE = (- 0.6 - 0.6 * ratio ** 2.0 / (Ri / Rt) - ratio / (1.0 + Ri / Rt) + 0.6 * (1.0 + ratio) ** 2.0 / (
                Rti / Rt)) * self.GG * Mt ** 2.0 / Rt
        # where the first two terms are gravitational binding energies of the target and impactor bodies
        # and the third term is the gravitational energy of the impactor body in the gravity potential of the target body
        # and the fourth term is the gravitational binding energy of the post-impact body under the assumption that the target and impactor perfectly merge

        # initial kinetic energy
        dKE = ratio / (1 + Ri / Rt) * self.GG * Mt ** 2.0 / Rt * self.vel ** 2.0

        # reading parameter coefficients for melt model
        parameter_list = [line.split() for line in open('parameter.txt')]
        para0 = np.array(parameter_list[0][:]).astype(np.float)  # parameters for vimp=vesc cases. See Table S.5
        para1 = np.array(parameter_list[1][:]).astype(np.float)  # parameters for vimp>1.1vesc cases. See Table S.6
            
        # reading all the error information
        error_read = [line.split() for line in open('Model_sigma_Grid.txt')]
        sigma0 = np.zeros(shape=(3,len(error_read)-1)) #parameter by angle v=esc
#         sigma1 = np.zeros(shape=(3,len(error_read)-1)) # v > 1.1esc #sigma1 never used ??!?
        Fsigma = np.zeros(len(error_read)-1)
        
        for m in range(0, 3):
            for n in range(0,len(error_read)-1):
                sigma0[m][n] = float(error_read[n+1][m])
#                 sigma1[m][n] = float(error_read[n+1][m+3])
        for n in range(0,len(error_read)-1):
            Fsigma[n] =  float(error_read[n+1][6])

        # merging criteria and crit velocity from Genda et al 2012 (equation 16)
        theta_G = 1 - np.sin(ang)
        GammaG = ((Mt - Mi) / (Mt + Mi))
        critical_velocity = 2.43 * GammaG ** 2.0 * theta_G ** 2.5 - 0.0408 * GammaG + 1.86 * theta_G ** 2.50 + 1.08
        
        if self.vel <= critical_velocity:  # merging
            # mantle mass fitting model at vimp=vesc. See Equation 7
            Mantle_mass_model = para0[10] * self.__legendre(0, np.cos(ang)) + para0[11] * self.__legendre(1, np.cos(ang))
            # mantle heating partitioning model at vimp=vesc. See Equation 6 
            h_model = para0[0] * self.__legendre(0, np.cos(ang)) + para0[1] * self.__legendre(1, np.cos(ang)) + para0[2] * self.__legendre(2, np.cos(ang))  # fitting model
            ee = para0[3:10] 
            
        else:  # no merging
            Mantle_mass_model = Mt/(Mi + Mt)
            # mantle heating partitioning model at vimp>1.1vesc.  See Equation 6
            h_model = para1[0] * self.__legendre(0, np.cos(ang)) + para1[1] * self.__legendre(1, np.cos(ang)) + para1[2] * self.__legendre(2, np.cos(ang))  # fitting model
            ee = para1[3:10]           

        IE_model = (ee[0] * self.__legendre(0, np.cos(ang)) + ee[1] * self.__legendre(1, np.cos(ang)) + 
                    ee[2] * self.__legendre(2, np.cos(ang)) + ee[3] * self.__legendre(3, np.cos(ang)) + 
                    ee[4] * self.__legendre(4, np.cos(ang)) + ee[5] * self.__legendre(5, np.cos(ang)) + 
                    ee[6] * self.__legendre(6, np.cos(ang)))  # internal energy model

        # computing the internal energy (Equation 3)
        # this is Delta U = Delta IE. See quation 10 and Section 3.2
        u_ave = h_model * IE_model * (dPE + dKE) / (0.70 * Mantle_mass_model * (Mt + Mi)) 
        # Mantle melt mass fraction. See Equation 9.
        f_model = min(1.0, h_model * IE_model * (dPE + dKE) / (0.70 * Mantle_mass_model * (Mt + Mi)) / self.EM) 
        
        angle_int=int(self.impact_angle)
        dz = np.sqrt((sigma0[0][angle_int]/IE_model)**2.0 +  (sigma0[1][angle_int]/h_model)**2.0  +  (sigma0[2][angle_int]/Mantle_mass_model)**2.0)
        u_ave_std =  u_ave  * dz
        f_model_std = f_model * dz

        # Heat distribution model within mantle. See equation 13 and Table S.7
        coef_read = [line.split() for line in open('CoefGrid.txt')]
        theta = np.zeros(shape=(len(coef_read), len(coef_read[1])))
        for m in range(0, len(coef_read)):
            for n in range(0, len(coef_read[1])):
                theta[m][n] = float(coef_read[m][n])
     
        melt_model = np.zeros(4)
        Mplanet = Mantle_mass_model * (Mt + Mi)  # planetary mass
        self.MadP = Mplanet
        
        # core radius calc
        dr = self.__radius(Mplanet) / 1000.0
        P = 0.0
        r = self.__radius(Mplanet)
        Mass = Mplanet
        CoreMass = 0.3 * Mplanet

        while (Mass > CoreMass):
            rho = self.__compute_density(P)
            P = P + rho * self.GG * Mass / r ** 2.0 * dr
            Mass = Mass - 4 * np.pi * rho * r ** 2.0 * dr
            r = r - dr

        rcore = r / self.__radius(Mplanet)  # core radius # changes with entropy 
    
        press = 0.0
        Mass = Mplanet
        u = np.zeros(shape=(0, 0))
        P = np.zeros(shape=(0, 0))
        
        rr = np.linspace(1.0, rcore, 20) * self.__radius(Mplanet)
        dr = np.abs(rr[1] - rr[0])

        for i in range(0, len(rr)):
            rho = self.__compute_density(press)
            u = np.append(u, np.maximum(0.0, self.rho_U_function(rho)))        
            press = press + rho * self.GG * Mass / rr[i] ** 2.0 * dr
            P = np.append(P, press)
            Mass = Mass - 4 * np.pi * rho * rr[i] ** 2.0 * dr
        
        rplanet = self.__radius(Mplanet)
        self.radiusPlanet = rplanet
        r = rr/rplanet       
        
        r_U_function = interp1d(r, u)  # making a function of the internal energy as a function of planetary radius
        r_P_function = interp1d(r, P)  # making a function of the pressure  as a function of planetary radius

        # grid spacing for calculating the magma ocean geometry
        self.rr = np.linspace(rcore, 1.0, 30)  
        # radial spacing - this value 30 can be changed to a different value depending on the radial resolution you need

        self.theta_angle = np.linspace(-np.pi, np.pi, 60)  
        # angle spacing (psi) - this value 60 can be changed to a different value depending on the angle resoultion you need
        
        nt = int(len(self.theta_angle))  # size of angle (psi) array
        nr = int(len(self.rr))  # size of radius array

        drr = (self.rr[1] - self.rr[0])# radial grid size
        dangle = self.theta_angle[1] - self.theta_angle[0]  # angle grid size
        self.du = np.zeros(shape=(nr, nt))  # internal energy
        self.du_sd = np.zeros(shape=(nr, nt))  # internal energy
        self.du_gain = np.zeros(shape=(nr, nt))  # internal energy gain
        number = np.zeros(shape=(nr, nt))
        self.tmelt = np.zeros(shape=(nr, nt))
        self.dv = np.zeros(shape=(nr, nt))
        self.rho = np.zeros(nr)
        
        # melt model w considering the initial temperature profile. this is 0 or 1; if a given part of the mantle is molten, this value is 1 otherwise 0
        self.du_melt = np.zeros(shape=(nr, nt), dtype=int)  
        # melt model w/o considering the initial temperature profile. this is 0 or 1; if a given part of the mantle is molten, this value is 1 otherwise 0
        self.du_gain_melt = np.zeros(shape=(nr, nt), dtype=int)  

        rmax_meltpool_model = 1.0  # magma ocean depth. 1.0: no magma ocean, 0.0: the entire mantle is molten
        rmax_meltpool_model_max_sd = 1.0
        rmax_meltpool_model_min_sd = 1.0   
        
        # make the internal energy as a function of r   
        for m in range(0, nr):
            for n in range(0, nt):
                y_model = 0.0
                k = 0
                for i in range(0, 5):
                    for j in range(0, 3):
                        y_model = y_model + theta[int(self.impact_angle)][k] * self.rr[m] ** (i - 2) * self.__legendre(j, np.cos(self.theta_angle[n]))
                        k = k + 1
                self.du[m][n] = y_model
                self.du_gain[m][n] = self.du[m][n]

        # error estimate
        for m in range(0, nr):
            for n in range(0, nt):
                self.du_sd[m][n] =  self.du[m][n] * np.sqrt(dz**2.0 + (Fsigma[angle_int]/self.du[m][n])**2.0) 
            
        du = self.du * u_ave
        du_gain = self.du_gain * u_ave
        du_sd = self.du_sd * u_ave
        du_max_sd  = np.zeros(shape=(nr, nt)) 
        du_min_sd  = np.zeros(shape=(nr, nt)) 
        
        meltV = 0.0  # melt volume
        totalV = 0.0  # total volume
        meltV_max_sd = 0.0
        meltV_min_sd = 0.0

        for m in range(0, nr):
            du_initial = float(r_U_function(self.rr[m]))  # initial internal energy profile
            for n in range(0, nt):
                du[m][n] = du[m][n] + du_initial  # total internal energy
                du_max_sd[m][n] =  du[m][n] +  du_sd[m][n] # + sigma
                du_min_sd[m][n] =  du[m][n] -  du_sd[m][n] # - sigma  
        
        for m in range(0, nr):
            self.rho[m] = self.__compute_density(1e9*self.__compute_pressure(Mt, self.rr[m]))
            for n in range(0, nt):
                # calculating an incremental  volume at this radius and angle - except its wrong :) totalVol > 100%
                dV = np.abs(np.pi * self.rr[m] ** 2.0 * np.sin(self.theta_angle[n]) * drr * dangle)  
                totalV = totalV + dV
                self.dv[m][n]= dV
                
                Press = r_P_function(self.rr[m])
                #Tmelt = (2500.0 + 26.0 * Press * 1e-9 - 0.052 * (Press * 1e-9) ** 2.0) * 1000.0 #Solomatov & Stevenson model. 1000 represents Cv
                if Press*1e-9 < 24.0: #Rubie et al., (2015) melt model
                    Tmelt = (1874.0 + 55.43 * Press * 1e-9 - 1.74 * (Press * 1e-9)**2.0  + 0.0193 * (Press * 1e-9)**3.0) * 1000.0
                else:    
                    Tmelt = (1249.0 + 58.28 * Press * 1e-9 - 0.395 * (Press * 1e-9)**2.0  + 0.011 * (Press * 1e-9)**3.0) * 1000.0      
                self.tmelt[m][n] = Tmelt
                
                # the best case scenario
                if du[m][n] > Tmelt:
                    self.du_melt[m][n] = 1  # this portion of the mantle is molten
                    meltV = meltV + dV
                    if rmax_meltpool_model > self.rr[m] and np.abs(
                            self.theta_angle[n]) < np.pi / 4.0: 
                        rmax_meltpool_model = self.rr[m]
                else:
                    self.du_melt[m][n] = 0 # this portion of the mantle is not molten

                # + sigma
                if du_max_sd[m][n] > Tmelt:
                    meltV_max_sd =  meltV_max_sd  + dV
                    if rmax_meltpool_model_max_sd > self.rr[m] and np.abs(self.theta_angle[n]) < np.pi / 4.0:  
                        rmax_meltpool_model_max_sd = self.rr[m]

                # - sigma
                if du_min_sd[m][n] > Tmelt:
                    meltV_min_sd =  meltV_min_sd  + dV
                    if rmax_meltpool_model_min_sd > self.rr[m] and np.abs(
                            self.theta_angle[n]) < np.pi / 4.0:  
                        rmax_meltpool_model_min_sd = self.rr[m]

        for m in range(0, nr):
            for n in range(0, nt):
                if du_gain[m][n] > self.latent_heat:  # if du_gain is larger than latent heat
                    self.du_gain_melt[m][n] = 1.0  # this part is considered molten
        
        melt_model = meltV / totalV  # calculating melt volume
        melt_model_max_sd =  meltV_max_sd / totalV
        melt_model_min_sd =  meltV_min_sd / totalV        
        
        # --- estimating the magma ocean depth and corresponding pressure

        # rmax_meltpool_model = max(rcore, rmax_meltpool_model)
        Pmax_meltpool_model = self.__compute_pressure(Mplanet, rmax_meltpool_model)
        Pmax_meltpool_model_max_sd = self.__compute_pressure(Mplanet, rmax_meltpool_model_max_sd)
        Pmax_meltpool_model_min_sd = self.__compute_pressure(Mplanet, rmax_meltpool_model_min_sd)        
        
        # assuming the same melt volume as the melt pool
        rmax_global_model = max(rcore, (1.0 - meltV/(4.0/3.0*np.pi))**(1/3))
        rmax_global_model_max_sd = max(rcore, (1.0 - meltV_max_sd/(4.0/3.0*np.pi))**(1/3))
        rmax_global_model_min_sd = max(rcore, (1.0 - meltV_min_sd/(4.0/3.0*np.pi))**(1/3))

        
        Pmax_global_model = self.__compute_pressure(Mplanet, rmax_global_model)
        Pmax_global_model_max_sd = self.__compute_pressure(Mplanet, rmax_global_model_max_sd)
        Pmax_global_model_min_sd = self.__compute_pressure(Mplanet, rmax_global_model_min_sd)        

        Pcmb = self.__compute_pressure(Mplanet, rcore)
        
        # assuming the conventional melt model (Eq 4)
        rmax_conventional_model = max(rcore, (1.0 - f_model * totalV/(4.0/3.0*np.pi))**(1/3))
        rmax_conventional_model_max_sd = max(rcore, (1.0 - min(1.0,(f_model + f_model_std)) * totalV/(4.0/3.0*np.pi))**(1/3))
        rmax_conventional_model_min_sd = max(rcore, (1.0 - max(0.0,(f_model - f_model_std)) * totalV/(4.0/3.0*np.pi))**(1/3))
        
        Pmax_conventional_model = self.__compute_pressure(Mplanet, rmax_conventional_model)
        Pmax_conventional_model_max_sd = self.__compute_pressure(Mplanet, rmax_conventional_model_max_sd)
        Pmax_conventional_model_min_sd = self.__compute_pressure(Mplanet, rmax_conventional_model_min_sd)

        if(debugPrint > 1):
            print("du gain", self.du_gain, "du gain of melt",self.du_gain_melt, "du", self.du, "du melt", self.du_melt)    
        
        if(debugPrint > 0):
            print("planetary radius: " +  str(float("{0:.2f}".format(rplanet * 1e-3))) + " km")
            print("mantle depth: " +  str(float("{0:.2f}".format(rplanet*(1.0-rcore) * 1e-3))) + " km")
            print("mantle volume fraction: " +  str(float("{0:.2f}".format(melt_model)))
                  + ' (+' + str(float("{0:.2f}".format(melt_model_max_sd-melt_model)))
                  + ', -' + str(float("{0:.2f}".format(melt_model-melt_model_min_sd)))+ ')') 
            
            print("magma ocean depth and pressure for a melt pool model: " + str(float("{0:.2f}".format(rplanet * 1e-3 * (1.0 - rmax_meltpool_model))))
                  + "(+" +  str(float("{0:.2f}".format(rplanet * 1e-3 * (- rmax_meltpool_model_max_sd + rmax_meltpool_model))))
                  + ", -" +  str(float("{0:.2f}".format(rplanet * 1e-3 * (rmax_meltpool_model_min_sd - rmax_meltpool_model)))) 
                  + ") km, "
                  +  str(float("{0:.2f}".format(Pmax_meltpool_model))) + "(+" +  str(float("{0:.2f}".format(Pmax_meltpool_model_max_sd-Pmax_meltpool_model))) +', -' +  str(float("{0:.2f}".format(Pmax_meltpool_model-Pmax_meltpool_model_min_sd))) + ") GPa")

            print("magma ocean depth and pressure for a global magma ocean model: " +  str(float("{0:.2f}".format(
                rplanet * 1e-3 * (1.0 - rmax_global_model))))
                  + "(+" +  str(float("{0:.2f}".format(rplanet * 1e-3 *  (-rmax_global_model_max_sd + rmax_global_model))))
                  + ", -" +  str(float("{0:.2f}".format(rplanet * 1e-3 *  (rmax_global_model_min_sd - rmax_global_model))))               
                  + ") km, "
                  +  str(float("{0:.2f}".format(Pmax_global_model))) + "(+" +  str(float("{0:.2f}".format(Pmax_global_model_max_sd-Pmax_global_model))) +', -' +  str(float("{0:.2f}".format(Pmax_global_model-Pmax_global_model_min_sd))) + ") GPa")

            print("magma ocean depth and pressure for a conventional model: " +  str(float("{0:.2f}".format(
                rplanet * 1e-3 * (1.0 - rmax_conventional_model))))
                  + "(+" +  str(float("{0:.2f}".format(rplanet * 1e-3 * (- rmax_conventional_model_max_sd + rmax_conventional_model))))
                  + ", -" +  str(float("{0:.2f}".format(rplanet * 1e-3 * (rmax_conventional_model_min_sd - rmax_conventional_model)))) + ") km, "
                  +  str(float("{0:.2f}".format(Pmax_conventional_model))) + "(+" +  str(float("{0:.2f}".format(Pmax_conventional_model_max_sd-Pmax_conventional_model))) +', -' +  str(float("{0:.2f}".format(Pmax_conventional_model-Pmax_conventional_model_min_sd))) + ") GPa")
        
        self.du = du * 1e-5  # normalized by 10^5 J/kg
        self.du_gain = du_gain * 1e-5
        
        self.DP_PoolOcnConv = np.zeros(shape=(3,2)) #depth in km, pressure in GPa
        self.DP_PoolOcnConv[0][0] = rplanet * 1e-3 * (1.0 - rmax_meltpool_model)
        self.DP_PoolOcnConv[0][1] = Pmax_meltpool_model
        self.DP_PoolOcnConv[1][0] = rplanet * 1e-3 * (1.0 - rmax_global_model)
        self.DP_PoolOcnConv[1][1] = Pmax_global_model
        self.DP_PoolOcnConv[2][0] = rplanet * 1e-3 * (1.0 - rmax_conventional_model)
        self.DP_PoolOcnConv[2][1] = Pmax_conventional_model
        
        d = {
            "impact velocity": self.vel,
            "impact angle": self.impact_angle,
            "critical velocity": critical_velocity,
            "planetary mass": Mplanet,
            "core radius": rcore,
            "max depth (global model) (km)": rplanet * 1e-3 * (1.0 - rmax_global_model),
            "max pressure (global model)": [Pmax_global_model,Pmax_global_model_max_sd-Pmax_global_model, -(Pmax_global_model-Pmax_global_model_min_sd)],         
            "max depth (conventional model) (km)": rplanet * 1e-3 * (1.0 - rmax_conventional_model),
            "max pressure (conventional model)": [Pmax_conventional_model, Pmax_conventional_model_max_sd-Pmax_conventional_model, -(Pmax_conventional_model-Pmax_conventional_model_min_sd)],
            "max depth (melt pool model) (km)": rplanet * 1e-3 * (1.0 - rmax_meltpool_model),
            "max pressure (melt pool model)": [Pmax_meltpool_model, Pmax_meltpool_model_max_sd-Pmax_meltpool_model, -(Pmax_meltpool_model-Pmax_meltpool_model_min_sd)],            
            "melt fraction": f_model,
            "rmax conventional" : rmax_conventional_model, 
            "melt model": meltV/totalV,
            "core mantle boundary pressure": Pcmb,
            "total volume": totalV,
            "internal energy": self.du,
            "average internal energy": u_ave,
            "internal energy gain": self.du_gain,
            "internal energy of the melt (considering initial temperature profile)": self.du_melt,
            "internal energy of the melt (not considering initial temperature profile)": self.du_gain_melt,
            "normalized depth (melt pool model)": [1.0-rmax_meltpool_model, rmax_meltpool_model_min_sd - rmax_meltpool_model,  rmax_meltpool_model_max_sd -rmax_meltpool_model],            
            "normalized depth (global model)": [1.0 - rmax_global_model, rmax_global_model_min_sd - rmax_global_model,  rmax_global_model_max_sd -rmax_global_model],
            "normalized depth (conventional model)": [1.0 - rmax_conventional_model,rmax_conventional_model_min_sd - rmax_conventional_model,  rmax_conventional_model_max_sd -rmax_conventional_model],  
            "radius planet": self.radiusPlanet
        }
        return d