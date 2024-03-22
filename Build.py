import os

# Cleaning built files and generated data
for i in ['CoefGrid.txt','Model_sigma_Grid.txt','F_bar.dat','rho_u_S9999.dat','parameter.txt']:
    if os.path.exists(i): os.remove(i)
if os.path.exists('coolData/'): [os.remove('coolData/'+i) for i in os.listdir('coolData/')]; os.rmdir('coolData/')
if os.path.exists('plotsAndStats/'): [os.remove('plotsAndStats/'+i) for i in os.listdir('plotsAndStats/')]; os.rmdir('plotsAndStats/')

# Building
Dep=os.listdir('Dependencies/')
for i in Dep:
    if not os.path.exists(i):
        with open(i,'w') as f:
            [print(line,end='',file=f) for line in open('Dependencies/'+i)]
    
import BarF_Interp, Coef_Interp, Sigma_Interp, Rho_P_U_Interp

for i in Dep:
    if (os.path.exists(i) and (not i=='parameter.txt')): os.remove(i)

if not os.path.exists('coolData'): os.makedirs('coolData')
# if not os.path.exists('plotsAndStats'): os.makedirs('plotsAndStats') 