import numpy as np
from libRustBCA import *
from scripts.materials import *
import matplotlib.pyplot as plt

def Simulate(E,theta,N=10000):
    #example ions array like [[energy1,angle1],[energy2,angle2],...]
    ion = hydrogen #inkomend deeltje
    impurity = oxygen

    angle = theta
    #In RustBCA's geometry, +x -> into the surface
    ux = np.cos(angle*np.pi/180.)*np.ones(N)
    uy = np.sin(angle*np.pi/180.)*np.ones(N)
    uz = np.zeros(N)

    Zmat = [5,oxygen['Z']]
    mmat = [10.811,oxygen['m']]
    Ecmat =[5.0,0.2]
    Esmat =[5.76,0.2]
    Ebmat =[0.0,0.0]
    nmat = [[0.115864,0.05]] #density in #/angstroms^3
    Dmat = [100]

    energies = np.ones(N)*E

    output,incident,stopped = compound_bca_list_1D_py(
            ux,uy,uz,energies,[ion['Z']]*N,
            [ion['m']]*N,[ion['Ec']]*N,[ion['Es']]*N,
            Zmat,mmat,Ecmat,Esmat,Ebmat,nmat,Dmat) 
    # output is (NX9 list of lists of f64) each row represents an output particle
    # (implanted,sputtered, or reflected)
    # elke rij: [Z, m (amu), E (eV), x(angstrom), y(angstrom), z(angstrom), ux, uy, uz]
    # incident: true als incident ion, false als uit target
    output = np.array(output)
    Z = output[:,0]
    m = output[:,1]
    E = output[:,2]
    x = output[:,3]
    y = output[:,4]
    z = output[:,5]
    ux = output[:,6]
    uy = output[:,7]
    uz = output[:,8]

    #print(N - len(Z[np.where(np.array(stopped)==False)]))
    impurities = len(np.where(Z[np.where(np.array(stopped)==False)]==8)[0])
    nboron = len(np.where(Z[np.where(np.array(stopped)==False)]==5)[0])
    implanted = len(np.where(Z[np.logical_and(incident, stopped)]==ion['Z'])[0])
    return {'sputtered impurities':impurities,
            'sputtered boron':nboron,
            'implanted ions':implanted}

if __name__ == '__main__':
    angles = np.linspace(0.0001,89.0001,90)
    impurities = []
    base = []
    implanted = []
    for angle in angles:
        result = Simulate(300,angle,10000)
        impurities.append(result['sputtered impurities'])
        base.append(result['sputtered boron'])
        implanted.append(result['implanted ions'])
    #print("sputtered impurities: " + str(result['sputtered impurities']))
    #print("sputtered Boron: "+ str(result['sputtered boron']))
    #print("implanted Hydrogen: "+ str(result['implanted ions']))
    plt.plot(angles,impurities,label='impurities')
    plt.plot(angles,base,label='boron')
    plt.plot(angles,implanted,label='implanted')
    plt.legend()
    plt.show()
