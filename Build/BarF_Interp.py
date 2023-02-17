import numpy as np
np.set_printoptions(threshold=4000, linewidth=4000)

barA = np.array([0, 4, 25, 50, 100, 200, 300]) # input pressures and corresponding outgoing longwave radiation below
OLRA = np.array([200000, 16700, 1360, 670, 400, 300, 285]) # 900k -> black body; 200k -> trace volatiles, silicon vapor etc
barGrid = np.arange(barA[0],barA[-1]+1)

OLRFit = np.polyfit(barA[0:3], 3.7*np.log10(OLRA[0:3])-13, 1); #log fit for first 3 data points, 
OLRGrid = np.array(10**(OLRFit[0]*barGrid[0:(barA[2]+1)]+OLRFit[1])+OLRA[2])

for p in np.arange(len(barA[2:])-1): #Appending linear interpolation for points 3 to 7
    thisGrid = np.arange(barGrid[barA[2+p]],barGrid[barA[3+p]]+1)
    s,b = np.polyfit((thisGrid[0],thisGrid[-1]), (OLRA[2+p], OLRA[2+p+1]), 1)
    OLRGrid = np.append(OLRGrid, s*thisGrid[1:]+b)
    
with open('F_bar_data.dat', 'w') as f:
    print("barGrid = [",end='',file=f); [print(x,end = ',', file=f) for x in barGrid[0:-1]]; print(barGrid[-1],']',file=f)
    print("OLRGrid = [",end='',file=f); [print(round(x,3),end = ',', file=f) for x in OLRGrid[0:-1]]; print(OLRGrid[-1],']',file=f)
