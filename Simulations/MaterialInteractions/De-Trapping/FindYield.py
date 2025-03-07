import numpy as np
from libRustBCA import *
from scripts.materials import *

#example ions array like [[energy1,angle1],[energy2,angle2],...]
#the example is with 10000 ions but 100000 also doesn't take to much time
N = 1000 #amount of particles to simulate
Z = 1 #atoomnummer inkomend deeltje
Zimp = 8 #atoomnummer vuiligheid
m = 1.0072766 #in amu
mimp = 15.999 #in amu
Ecut = 3.5#cutoff energie inkomend deeltje
Esur = 4.84#surface binding energie inkomend deeltje

ux = list(np.ones(N))  #xvec
uy = list(np.zeros(N)) #yvec
uz = list(np.zeros(N)) #zvec
Z1 = list(np.ones(N)*Z) 
m1 = list(np.ones(N)*m) 
Ec1 = list(np.ones(N)*Ecut) 
Es1 = list(np.ones(N)*Esur) 
Z2 = list([5,Zimp]) 
m2 = list([10.811,mimp]) 
Ec2 = list([5.0,0.2]) 
Es2 = list([5.76,0.2]) 
Eb2 = list([0.0,0.0]) 
n2 = list([0.115864,0.01]) #density in #/angstroms^3

energies = list(np.random.default_rng().uniform(0.0, 20.0, N)) #0-20eV

output = compound_bca_list_py(ux,uy,uz,energies,Z1,m1,Ec1,Es1,Z2,m2,Ec2,Es2,n2,Eb2)
