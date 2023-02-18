import os
Dep=os.listdir("Dependencies/")
for i in Dep:
    os.rename("Dependencies/"+i,i)
    
import BarF_Interp, Coef_Interp, Sigma_Interp, Rho_P_U_Interp, Init_Du

for i in Dep:
    os.rename(i,"Dependencies/"+i)
if not os.path.exists("coolData"): os.makedirs("coolData")