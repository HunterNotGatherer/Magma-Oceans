import numpy as np
np.set_printoptions(threshold=4000, linewidth=4000)

grid = np.arange(91) # can be made finer
Angles = [0, 30, 60, 90] # default input options

# Reading coef.txt data - 4 angles 15 coef each # Check if values updated on github occasionally
coef = [line.split() for line in open("coef.txt")] #data files changed, add original to read from
coefDat = np.zeros(shape=(4,15))
for c in range(0, len(coef)):
    coefDat[c] = [float(i) for i in coef[c]]

# coef.txt interpolation
fitCoefExp = [1,2,2]
CoefGrid=np.zeros(shape=(15,91))
for k in np.arange(len(coefDat[0])):
    fitCoef = np.polyfit(Angles, [coefDat[0][k], coefDat[1][k], coefDat[2][k], coefDat[3][k]], fitCoefExp[k%3])
    YCoef=0
    for m in np.arange(len(fitCoef)):
        YCoef+=fitCoef[m]*grid**(len(fitCoef)-m-1)
    CoefGrid[k]=YCoef

with open('CoefGrid.txt', 'w') as f:
    for r in np.arange(len(CoefGrid[0])):
        strings = ["%.8f" % number for number in CoefGrid.T[r]]
        print(' '.join(strings), file=f)
