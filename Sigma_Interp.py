import numpy as np
np.set_printoptions(threshold=4000, linewidth=4000)

grid = np.arange(91) # can be made finer
Angles = [0, 30, 60, 90] # default input options

# IE, vimp>vesc; # h model, vimp>vesc; # mantle model, vimp>vesc; # F model
sigma = [line.split() for line in open("Model_sigma.txt")] #data files may jave changed, double check from source
sigDat = np.zeros(shape=(7,4))
for sig in range(1, len(sigma)):
    sigDat[sig-1] = [float(i) for i in sigma[sig][0:4]]

# Polyfit model_sigma.txt data for interpolation
Y=np.zeros(shape=(len(sigDat),len(grid))) # Y holds the resulting interpolation
for j in np.arange(len(sigDat)):
    fit=np.polyfit(Angles,sigDat[j],3) # polynomial fit
    for i in np.arange(len(fit)):
        Y[j,:]+=fit[i]*grid**(len(fit)-i-1)
        
with open('Model_sigma_Grid.txt', 'w') as f: 
    print('# angle 0-90, first line is the DIE error and second is the F error', file=f)
    for q in np.arange(len(Y[0])):
        strings = ["%.8f" % number for number in Y.T[q][0:-1]]
        print(' '.join(strings),end=" ", file=f)
        print(f"{Y.T[q][-1]:.18f}", file=f)
