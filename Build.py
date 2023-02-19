import os
Dep=os.listdir("Dependencies/")
for i in Dep:
    with open(i,'w') as f:
        [print(line,end='',file=f) for line in open("Dependencies/"+i)]
    
import BarF_Interp, Coef_Interp, Sigma_Interp, Rho_P_U_Interp, Init_Du

for i in Dep:
    if os.path.exists(i): os.remove(i)

if not os.path.exists("coolData"): os.makedirs("coolData")