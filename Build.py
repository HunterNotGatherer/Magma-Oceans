import os
Dep=os.listdir('Dependencies/')
for i in Dep:
    with open(i,'w') as f:
        [print(line,end='',file=f) for line in open('Dependencies/'+i)]
    
import BarF_Interp, Coef_Interp, Sigma_Interp, Rho_P_U_Interp
isBuilt=True
for i in ['CoefGrid.txt','Model_sigma_Grid.txt','F_bar.dat','rho_u_S9999.dat']:
    if not os.path.exists(i): isBuilt=False
if isBuilt: import Init_Du

for i in Dep:
    if (os.path.exists(i) and (not i=='parameter.txt')): os.remove(i)

if not os.path.exists('coolData'): os.makedirs('coolData')