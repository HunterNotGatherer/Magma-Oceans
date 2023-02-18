import os
for i in ["CoefGrid.txt","Model_sigma_Grid.txt","F_bar.dat","init_du.dat","rho_u_S9999.dat"]:
    if os.path.exists(i): os.remove(i)
x=[os.remove("coolData/"+i) for i in os.listdir("coolData/")]