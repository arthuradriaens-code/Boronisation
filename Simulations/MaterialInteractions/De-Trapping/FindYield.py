import csv
import numpy as np
from libRustBCA import *
from scripts.materials import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from alive_progress import alive_bar


def SimulateHOM(E,theta,n_i,N=10000):
    #example ions array like [[energy1,angle1],[energy2,angle2],...]
    ion = deuterium #inkomend deeltje
    impurity = tritium

    angle = theta
    #In RustBCA's geometry, +x -> into the surface
    ux = np.cos(angle*np.pi/180.)*np.ones(N)
    uy = np.sin(angle*np.pi/180.)*np.ones(N)
    uz = np.zeros(N)

    Zmat = [5,impurity['Z']]
    mmat = [10.811,impurity['m']]
    Ecmat =[0.1,0.1]
    Esmat =[5.73,0.0]
    Ebmat =[0.13,0.0]
    nmat = [[0.06128,n_i]] #density in #/angstroms^3
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
    impurities = len(np.where(Z[np.where(np.array(stopped)==False)]==impurity['Z'])[0])
    nboron = len(np.where(Z[np.where(np.array(stopped)==False)]==5)[0])
    implanted = len(np.where(Z[np.logical_and(incident, stopped)]==ion['Z'])[0])
    #print(x[np.where(Z[np.where(np.array(stopped)==False)]==8)[0]])

    return {'sputtered impurities':impurities,
            'sputtered boron':nboron,
            'implanted ions':implanted}
    
def SimulateLAY(E,theta,n_i,N=10000):
    ion = hydrogen 
    impurity = oxygen
    angle = theta
    ux = np.cos(angle*np.pi/180.)*np.ones(N)
    uy = np.sin(angle*np.pi/180.)*np.ones(N)
    uz = np.zeros(N)
    Zmat = [5,oxygen['Z']]
    mmat = [10.811,oxygen['m']]
    Ecmat =[5.0,0.2]
    Esmat =[5.76,0.2]
    Ebmat =[0.0,0.0]
    nmat = [[0.115864,n_i]] #density in #/angstroms^3
    Dmat = [[100/len(n_i)]*len(n_i)] #equally spaced slabs
    energies = np.ones(N)*E
    output,incident,stopped = compound_bca_list_1D_py(
            ux,uy,uz,energies,[ion['Z']]*N,
            [ion['m']]*N,[ion['Ec']]*N,[ion['Es']]*N,
            Zmat,mmat,Ecmat,Esmat,Ebmat,nmat,Dmat) 
    output = np.array(output)
    Z = output[:,0]
    impurities = len(np.where(Z[np.where(np.array(stopped)==False)]==8)[0])
    nboron = len(np.where(Z[np.where(np.array(stopped)==False)]==5)[0])
    implanted = len(np.where(Z[np.logical_and(incident, stopped)]==ion['Z'])[0])
    return {'sputtered impurities':impurities,
            'sputtered boron':nboron,
            'implanted ions':implanted}

def TimeEvolution(n0,energy,angle,timesteps):
    deltat = timesteps[1]-timesteps[0]
    concentration = n0
    Ibars = []
    epsilon = n0*0.04
    for i,timestep in enumerate(timesteps):
        Ibar = Simulate(energy,angle,concentration,10000)['sputtered impurities']
        Ibars.append(Ibar)
        concentration -= Ibar/100*deltat
        if abs(concentration) < epsilon: #stop when depleted
            break
    return Ibars

def powerlaw(xdata,amp,power,offset):
    return amp*(xdata**power) + offset

if __name__ == '__main__':
    timestep = False
    impurities = []
    base = []
    implanted = []
    energy = 50
    boronconcentration = 0.06128
    concentration = boronconcentration*0.2

    if timestep:
        timesteps = np.linspace(0,10,1000)
        Ibars = TimeEvolution(concentration,energy,40,timesteps)
        import csv
        with open('csvfiles/result.csv','w',newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|')
            for i,timestep in enumerate(timesteps[1:len(Ibars)]):
                spamwriter.writerow([timestep,Ibars[i+1]])
    else:
        angles = np.linspace(0.001,89.0001,178)
        with alive_bar(len(angles)) as bar:
            for angle in angles:
                result = SimulateHOM(energy,angle,concentration,50000)
                impurities.append(result['sputtered impurities'])
                base.append(result['sputtered boron'])
                implanted.append(result['implanted ions'])
                bar()
        plt.plot(angles,impurities,label='released tritium')
        plt.plot(angles,base,label='boron')
        plt.plot(angles,implanted,label='implanted deuterium')
        plt.xlabel('angle (degrees)')
        plt.ylabel('atoms')
        plt.title('effect of 50000 deuterium atoms at 50eV for different angles')
        plt.legend()
        plt.show()
