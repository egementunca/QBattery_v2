from qutip import *
import numpy as np

seed = 17

X = sigmax()
Y = sigmay()
Z = sigmaz()
I = qeye(2)

def Hxxx(N, edges, h=1):
    global X,Y,Z,I
    matrices = [I for i in range(N)]
    sumsigmaz = 0
    sumsigmaxx = 0
    sumsigmayy = 0
    sumsigmazz = 0
    if type(h) is int:
        h *= np.ones(N)
#     if type(J) is int:
#         J *= np.ones(len(edges))        
    for i in range(N):
        matrices[i] = Z
        sumsigmaz += h[i]*tensor(matrices)
        matrices[i] = I    
    for i in range(len(edges)):
        matrices[edges[i][0]] = X
        matrices[edges[i][1]] = X
        sumsigmaxx += edges[i][2]*tensor(matrices)
#         matrices[edges[i][0]] = I
#         matrices[edges[i][1]] = I
        matrices[edges[i][0]] = Y
        matrices[edges[i][1]] = Y
        sumsigmayy += edges[i][2]*tensor(matrices)
#         matrices[edges[i][0]] = I
#         matrices[edges[i][1]] = I
        matrices[edges[i][0]] = Z
        matrices[edges[i][1]] = Z
        sumsigmazz += edges[i][2]*tensor(matrices)
        matrices[edges[i][0]] = I
        matrices[edges[i][1]] = I  
    H = sumsigmaxx+sumsigmazz+sumsigmayy+ sumsigmaz
    return H

def excstate(N):
    up = tensor([basis(2,0) for n in range(N)])
    state = up*up.dag()
    return state

def most_excited_state(H0):
    excited_state = H0.eigenstates()[1][-1]
    state = excited_state*excited_state.dag()
    return state

def calc_erg(H_vals, H_vecs, rho_vals, rho_vecs):
    erg = 0
    for i in range(H_vals.shape[0]):
        for j in range(H_vals.shape[0]):
            delta = 0
            if i==j:
                delta = 1
            erg += (rho_vals[i]*H_vals[j]) * ((np.abs((rho_vecs[i].dag()*H_vecs[j])[0,0]))**2 - delta)
    return erg

def charge(H0, Htot, ancilla, target, tracekeep, collisionNum, t_steps):
    fidel = []
    ergotropy = []

    HsystEigEn, HsysEigVec = H0.eigenstates(sparse=False, sort='low', eigvals=0, tol=0, maxiter=100000)
    _, gsket = H0.groundstate()
    rhoGs = gsket*gsket.dag()
    rhoGsVals, rhoGsVecs = rhoGs.eigenstates(sparse=False, sort='high', eigvals=0, tol=0, maxiter=100000)

    ergotropy.append(calc_erg(HsystEigEn, HsysEigVec, rhoGsVals, rhoGsVecs))
    fidel.append(fidelity(target,rhoGs))
    
    Rin = tensor(ancilla, rhoGs)
    
    for r in range(collisionNum):    
        Rt = mesolve(Htot,Rin,t_steps)
        
        Rtstates = Rt.states[-1].unit()
        Rsyst_red = ptrace(Rtstates,tracekeep).unit()
        RsystEigEn, RsystEigVec = Rsyst_red.eigenstates(sparse=False, sort='high', eigvals=0, tol=0, maxiter=100000)

        ergotropy.append(calc_erg(HsystEigEn, HsysEigVec, RsystEigEn, RsystEigVec))
        fidel.append(fidelity(target,Rsyst_red))
        Rin=tensor(ancilla,Rsyst_red)

    return ergotropy, fidel, Rsyst_red

#Battery Configs
h = 1 
N0 = 4

#Collision Configs
chargeTime = 100
collisionNum, realizations = 100 , 200
t_steps = np.linspace(0, (chargeTime/collisionNum), 100)

#Plot Configs
tau = np.linspace(0,1,collisionNum+1)

#Realization Configs
#std_list = np.linspace(.1,.8,8)
std_list = np.array([1.2, 2.0])

#1-0
N = 5 
fidelity1 = np.zeros(shape=(len(std_list), collisionNum+1))
ergotropy1 = np.zeros(shape=(len(std_list), collisionNum+1))


np.random.seed(seed)
for i, sigma in enumerate(std_list):

    fid_ = np.zeros(shape=(realizations, collisionNum+1))
    erg_ = np.zeros(shape=(realizations, collisionNum+1))

    for j in range(realizations):

        J = np.random.normal(loc=0, scale=sigma, size=N0)
        g = [1,1,1,1]
        p = [1,1,1,1]
        edges0 = [[0,1,J[0]],[1,2,J[1]],[2,3,J[2]],[3,1,J[3]]]
        edges1 = [[0,1,g[0]],[1,2,J[0]],[2,3,J[1]],[3,4,J[2]],[4,1,J[3]]]

        H0 = Hxxx(N0, edges0)
        Htot = Hxxx(N, edges1)
        ancilla = excstate(N-N0)
        target = excstate(N0)
        tracekeep = np.arange((N-N0),N,1)
        e , f, rhoFinal  = charge(H0, Htot, ancilla, target, tracekeep, collisionNum, t_steps)
        fid_[j,::] = np.array(f)
        erg_[j,::] = np.array(e)

    fidelity1[r,::] = np.mean(fid_, axis=0)
    ergotropy1[r,::] = np.mean(erg_, axis=0)

#2-0
N = 6
fidelity20 = np.zeros(shape=(len(std_list), collisionNum+1))
ergotropy20 = np.zeros(shape=(len(std_list), collisionNum+1))

np.random.seed(seed)
for i, sigma in enumerate(std_list):

    fid_ = np.zeros(shape=(realizations, collisionNum+1))
    erg_ = np.zeros(shape=(realizations, collisionNum+1))

    for j in range(realizations):

        J = np.random.normal(loc=0, scale=sigma, size=N0)
        g = [1,1,1,1]
        p = [1,1,1,1]
        edges0 = [[0,1,J[0]],[1,2,J[1]],[2,3,J[2]],[3,1,J[3]]]
        edges20= [[0,2,g[0]],[1,3,g[1]],[2,3,J[0]],[3,4,J[1]],[4,5,J[2]],[5,2,J[3]]]

        H0 = Hxxx(N0, edges0)
        Htot = Hxxx(N, edges20)
        ancilla = excstate(N-N0)
        target = excstate(N0)
        tracekeep = np.arange((N-N0),N,1)
        e , f, rhoFinal  = charge(H0, Htot, ancilla, target, tracekeep, collisionNum, t_steps)
        fid_[j,::] = np.array(f)
        erg_[j,::] = np.array(e)

    fidelity20[i,::] = np.mean(fid_, axis=0) 
    ergotropy20[i,::] = np.mean(erg_, axis=0)

#2-1
N = 6
fidelity21 = np.zeros(shape=(len(std_list), collisionNum+1))
ergotropy21 = np.zeros(shape=(len(std_list), collisionNum+1))

np.random.seed(seed)
for i, sigma in enumerate(std_list):

    fid_ = np.zeros(shape=(realizations, collisionNum+1))
    erg_ = np.zeros(shape=(realizations, collisionNum+1))

    for j in range(realizations):

        J = np.random.normal(loc=0, scale=sigma, size=N0)
        g = [1,1,1,1]
        p = [1,1,1,1]
        edges0 = [[0,1,J[0]],[1,2,J[1]],[2,3,J[2]],[3,1,J[3]]]
        edges21= [[0,1,p[0]],[0,2,g[0]],[1,3,g[1]],[2,3,J[0]],[3,4,J[1]],[4,5,J[2]],[5,2,J[3]]]

        H0 = Hxxx(N0, edges0)
        Htot = Hxxx(N, edges21)
        ancilla = excstate(N-N0)
        target = excstate(N0)
        tracekeep = np.arange((N-N0),N,1)
        e , f, rhoFinal  = charge(H0, Htot, ancilla, target, tracekeep, collisionNum, t_steps)
        fid_[j,::] = np.array(f)
        erg_[j,::] = np.array(e)

    fidelity21[i,::] = np.mean(fid_, axis=0) 
    ergotropy21[i,::] = np.mean(erg_, axis=0)

#3-0
N = 7
fidelity30 = np.zeros(shape=(len(std_list), collisionNum+1))
ergotropy30 = np.zeros(shape=(len(std_list), collisionNum+1))

np.random.seed(seed)
for i, sigma in enumerate(std_list):

    fid_ = np.zeros(shape=(realizations, collisionNum+1))
    erg_ = np.zeros(shape=(realizations, collisionNum+1))

    for j in range(realizations):

        J = np.random.normal(loc=0, scale=sigma, size=N0)
        g = [1,1,1,1]
        p = [1,1,1,1]
        edges0 = [[0,1,J[0]],[1,2,J[1]],[2,3,J[2]],[3,1,J[3]]]
        edges30= [[0,3,g[0]],[1,4,g[1]],[2,5,g[2]],[3,4,J[0]],[4,5,J[1]],[5,6,J[2]],[6,3,J[3]]]

        H0 = Hxxx(N0, edges0)
        Htot = Hxxx(N, edges30)
        ancilla = excstate(N-N0)
        target = excstate(N0)
        tracekeep = np.arange((N-N0),N,1)
        e , f, rhoFinal  = charge(H0, Htot, ancilla, target, tracekeep, collisionNum, t_steps)
        fid_[j,::] = np.array(f)
        erg_[j,::] = np.array(e)

    fidelity30[i,::] = np.mean(fid_, axis=0) 
    ergotropy30[i,::] = np.mean(erg_, axis=0)

#3-3
N = 7
fidelity33 = np.zeros(shape=(len(std_list), collisionNum+1))
ergotropy33 = np.zeros(shape=(len(std_list), collisionNum+1))

np.random.seed(seed)
for i, sigma in enumerate(std_list):

    fid_ = np.zeros(shape=(realizations, collisionNum+1))
    erg_ = np.zeros(shape=(realizations, collisionNum+1))

    for j in range(realizations):

        J = np.random.normal(loc=0, scale=sigma, size=N0)
        g = [1,1,1,1]
        p = [1,1,1,1]
        edges0 = [[0,1,J[0]],[1,2,J[1]],[2,3,J[2]],[3,1,J[3]]]
        edges33= [[0,1,p[0]],[1,2,p[1]],[2,0,p[2]],[0,3,g[0]],[1,4,g[1]],[2,5,g[2]],[3,4,J[0]],[4,5,J[1]],[5,6,J[2]],[6,3,J[3]]]

        H0 = Hxxx(N0, edges0)
        Htot = Hxxx(N, edges33)
        ancilla = excstate(N-N0)
        target = excstate(N0)
        tracekeep = np.arange((N-N0),N,1)
        e , f, rhoFinal  = charge(H0, Htot, ancilla, target, tracekeep, collisionNum, t_steps)
        fid_[j,::] = np.array(f)
        erg_[j,::] = np.array(e)

    fidelity33[i,::] = np.mean(fid_, axis=0) 
    ergotropy33[i,::] = np.mean(erg_, axis=0)

#4-0
N = 8
fidelity40 = np.zeros(shape=(len(std_list), collisionNum+1))
ergotropy40 = np.zeros(shape=(len(std_list), collisionNum+1))

np.random.seed(seed)
for i, sigma in enumerate(std_list):

    fid_ = np.zeros(shape=(realizations, collisionNum+1))
    erg_ = np.zeros(shape=(realizations, collisionNum+1))

    for j in range(realizations):

        J = np.random.normal(loc=0, scale=sigma, size=N0)
        g = [1,1,1,1]
        p = [1,1,1,1]
        edges0 = [[0,1,J[0]],[1,2,J[1]],[2,3,J[2]],[3,1,J[3]]]
        edges40= [[0,4,g[0]],[1,5,g[1]],[2,6,g[2]],[3,7,g[3]],[4,5,J[0]],[5,6,J[1]],[6,7,J[2]],[7,4,J[3]]]

        H0 = Hxxx(N0, edges0)
        Htot = Hxxx(N, edges40)
        ancilla = excstate(N-N0)
        target = excstate(N0)
        tracekeep = np.arange((N-N0),N,1)
        e , f, rhoFinal  = charge(H0, Htot, ancilla, target, tracekeep, collisionNum, t_steps)
        fid_[j,::] = np.array(f)
        erg_[j,::] = np.array(e)

    fidelity40[i,::] = np.mean(fid_, axis=0) 
    ergotropy40[i,::] = np.mean(erg_, axis=0)

#4-4
N = 8
fidelity44 = np.zeros(shape=(len(std_list), collisionNum+1))
ergotropy44 = np.zeros(shape=(len(std_list), collisionNum+1))

np.random.seed(seed)
for i, sigma in enumerate(std_list):

    fid_ = np.zeros(shape=(realizations, collisionNum+1))
    erg_ = np.zeros(shape=(realizations, collisionNum+1))

    for j in range(realizations):

        J = np.random.normal(loc=0, scale=sigma, size=N0)
        g = [1,1,1,1]
        p = [1,1,1,1]
        edges0 = [[0,1,J[0]],[1,2,J[1]],[2,3,J[2]],[3,1,J[3]]]
        edges44= [[0,1,p[0]],[1,2,p[1]],[2,3,p[2]],[3,1,p[3]],[0,4,g[0]],[1,5,g[1]],[2,6,g[2]],[3,7,g[3]],[4,5,J[0]],[5,6,J[1]],[6,7,J[2]],[7,4,J[3]]]

        H0 = Hxxx(N0, edges0)
        Htot = Hxxx(N, edges44)
        ancilla = excstate(N-N0)
        target = excstate(N0)
        tracekeep = np.arange((N-N0),N,1)
        e , f, rhoFinal  = charge(H0, Htot, ancilla, target, tracekeep, collisionNum, t_steps)
        fid_[j,::] = np.array(f)
        erg_[j,::] = np.array(e)

    fidelity44[i,::] = np.mean(fid_, axis=0) 
    ergotropy44[i,::] = np.mean(erg_, axis=0)


np.save(f'last_realization_big_data_seed_{seed}', (fid_, erg_))

charger_names = ['1-0', '2-0', '2-1', '3-0', '3-3', '4-0', '4-4']
fids = np.array([fidelity1, fidelity20, fidelity21, fidelity30, fidelity33, fidelity40, fidelity44])
ergs = np.array([ergotropy1, ergotropy20, ergotropy21, ergotropy30, ergotropy33, ergotropy40, ergotropy44])
np.save(f'gaussianHeisenberg{seed}.npy', (fids, ergs))