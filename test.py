from qiskit.aqua.algorithms import VQE, NumPyEigensolver
import matplotlib.pyplot as plt
import numpy as np
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
# from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.aqua.components.optimizers import AQGD,COBYLA,L_BFGS_B,SLSQP,ADAM
from qiskit.aqua.operators import Z2Symmetries
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, HFMethodType
from qiskit.chemistry import FermionicOperator
from qiskit.providers.aer import StatevectorSimulator as SS
from qiskit import Aer
import time
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.converters import circuit_to_dag, dag_to_circuit

# map_type='bravyi_kitaev'
map_type='parity'
# map_type='jordan_wigner'
def get_qubit_op(dist):
    driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(dist)+"; H .0 .0 "+str(dist*2)+";H .0 .0 "+str(dist*3)#+";H .0 .0 "+str(dist*4)+";H .0 .0 "+str(dist*5)
                         , unit=UnitsType.ANGSTROM
                         , spin=0,charge=0, basis='sto3g')
    molecule = driver.run()
    # freeze_list=[0]
    # remove_list=[-3,-2]
    repulsion_energy = molecule.nuclear_repulsion_energy
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    # remove_list = [x % molecule.num_orbitals for x in remove_list]
    # freeze_list = [x % molecule.num_orbitals for x in freeze_list]
    # remove_list = [x - len(freeze_list) for x in remove_list]
    # remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]
    # freeze_list += [x + molecule.num_orbitals for x in freeze_list]
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    # ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
    # num_spin_orbitals -= len(freeze_list)
    # num_particles -= len(freeze_list)
    # ferOp = ferOp.fermion_mode_elimination(remove_list)
    # num_spin_orbitals -= len(remove_list)
    qubitOp = ferOp.mapping(map_type=map_type, threshold=0.00000001)
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
    # shift =  energy_shift + repulsion_energy
    shift =  repulsion_energy
    return qubitOp, num_particles, num_spin_orbitals, shift, molecule.hf_energy

def makewave(n):
    params = ParameterVector('θ',int(3*n*(n-1)+6*n))
    # params = np.random.rand(42)
    wavefunction = QuantumCircuit(n)
    t=0
    wavefunction.x(0)
    wavefunction.x(3)
    for i in range(n):
        for j in range(n):
            if i!=j:
                wavefunction.cry(params[t],i,j)
                t+=1
                # wavefunction.crx(params[t],i,j)
                # t+=1
                
    for i in range(n):
        # wavefunction.rx(params[t],i)
        # t+=1
        # wavefunction.ry(params[t],i)
        # t+=1
        # wavefunction.rz(params[t],i)
        # t+=1
        # wavefunction.rx(params[t],i)
        # t+=1
        wavefunction.ry(params[t],i)
        t+=1
        wavefunction.rz(params[t],i)
        t+=1
        
    # wavefunction.measure([0,1,2,3,4,5],[0,1,2,3,4,5])        
    return wavefunction

def drawwave(n,params):
    # params = ParameterVector('θ',int(3*n*(n-1)+6*n))
    # params = np.random.rand(42)
    wavefunction = QuantumCircuit(n,n)
    t=0
    # for i in range(n):
    #     wavefunction.rx(params[t],i)
    #     t+=1
    #     wavefunction.ry(params[t],i)
    #     t+=1
    #     wavefunction.rz(params[t],i)
    #     t+=1
    #     wavefunction.rx(params[t],i)
    #     t+=1
    #     wavefunction.ry(params[t],i)
    #     t+=1
    #     wavefunction.rz(params[t],i)
    #     t+=1
    wavefunction.x(0)
    wavefunction.x(3)
    for i in range(n):
        for j in range(n):
            if i!=j:
                wavefunction.cry(params[t],i,j)
                t+=1
                
    for i in range(n):
        # wavefunction.rz(params[t],i)
        # t+=1
        # wavefunction.ry(params[t],i)
        # t+=1
        # wavefunction.rz(params[t],i)
        # t+=1
        wavefunction.rx(params[t],i)
        t+=1
        wavefunction.ry(params[t],i)
        t+=1
        wavefunction.rz(params[t],i)
        t+=1
        
    wavefunction.measure([0,1,2,3,4,5],[0,1,2,3,4,5])        
    return wavefunction

dist=1
qubitOp, num_particles, num_spin_orbitals, shift, hfenergy = get_qubit_op(dist)
print("num of qubit",qubitOp.num_qubits)
vqe_t=[]
vqe_ut=[]
vqe_i=[]
dc=0
vqe_dt=1
def runone(distances):
    doit=1
    endit=0
    red=0
    th=1e-3
    while(doit):
        exact_energies = []
        vqe_energies = []
        vqe_degrees= []
        vqe_u=[]
        for d in range(len(distances)):
            dist=distances[d]
            qubitOp, num_particles, num_spin_orbitals, shift, hfenergy = get_qubit_op(dist)
            result = NumPyEigensolver(qubitOp).run()
            exact_energies.append(np.real(result.eigenvalues) + shift)
            initial_state = HartreeFock(
                num_spin_orbitals,
                num_particles,
                qubit_mapping=map_type
                )
            var_form = UCCSD(
                num_orbitals=num_spin_orbitals,
                num_particles=num_particles,
                initial_state=initial_state,
                qubit_mapping=map_type,
                two_qubit_reduction=1
                )
            vqe = VQE(qubitOp,var_form=wavefunction)#,optimizer=optimizer)
            vqeu = VQE(qubitOp,var_form=var_form)#, optimizer=optimizer)
            t1=time.time()
            if d>0:
                vqe.initial_point=vqe_degrees[-1]
        # if d==0 and endit>0:
        #     vqe.initial_point=vqe_init
            vqe_result=np.real(vqe.run(backend)['eigenvalue'] + shift)
            t2=time.time()
            vqe_t.append(t2-t1)
            t1=time.time()
            vqeu_result = np.real(vqeu.run(backend)['eigenvalue'] + shift)
            t2=time.time()
            vqe_ut.append(t2-t1)
            vqe_energies.append(vqe_result)
            vqe_u.append(vqeu_result)
            vqe_degrees.append(vqe.optimal_params)
            print("Interatomic Distance:", np.round(dist, 2), "VQE Result:", vqe_result, "Exact Energy:",
                  exact_energies[-1],"VQE-UCCSD Result:", vqeu_result)
            if np.linalg.norm(vqe_energies[0]-exact_energies[0])>th:
                red+=1
                break
            else:
                if d==0:
                    endit=endit+1
        if red>10:
            th=th*2
            red=0
        if endit==1:
            break
    return vqe_degrees,exact_energies,vqe_energies,vqe_u

def oneoptimize(vqe_degrees):
    vqe_init=np.zeros(wavefunction.num_parameters)
    vqe_degrees=np.array(vqe_degrees)
    vqe_i.append(vqe_degrees)
    vtmp=[]
    for i in range(len(vqe_degrees[0,:])):
        vtmp.append(np.linalg.norm(vqe_degrees[:,i]))
        vsort=np.sort(vtmp)
    for i in range(len(vtmp)):
        if vtmp[i]==vsort[0]:
            if vqe_degrees[:,i][0]<1:
                vqe_init[i]=vqe_degrees[:,i][0]
    return vqe_init

def randreduct(wavefunction):
    num=wavefunction.num_parameters
    dag=circuit_to_dag(wavefunction)
    erase=0
    while erase<num/2:
        k=np.random.randint(0,len(dag.op_nodes()))
        dag.remove_op_node(dag.op_nodes()[k])
        erase=erase+1
        
    aa=dag_to_circuit(dag)
    return aa

backend = SS(method="statevector")
# distances = np.arange(0.5, 4.0, 0.1)
nump=0

def simprun():
    d=0
    dist=1
    qubitOp, num_particles, num_spin_orbitals, shift, hfenergy = get_qubit_op(dist)
    wavefunction=makewave(qubitOp.num_qubits)
    # wavefunction=randreduct(wavefunction)
    result = NumPyEigensolver(qubitOp).run()
    result=np.real(result.eigenvalues) + shift
    # optimizer = ADAM(maxiter=10000)
    mini=0
    while(1):
        vqe = VQE(qubitOp,var_form=wavefunction)#,optimizer=optimizer)
        vqe_result=np.real(vqe.run(backend)['eigenvalue'] + shift)
        if vqe_result<mini:
            print("Inital optimize:  ", "VQE Result:", vqe_result, "Exact Energy:", result)
            mini=vqe_result
        if np.linalg.norm(vqe_result-result)<1e-3:
            break
        else:
            d+=1
        if d>10:
            break
            # wavefunction=makewave(qubitOp.num_qubits)
            # wavefunction=randreduct(wavefunction)
            # d=0
    
    return wavefunction,vqe
    
wavefunction,vv=simprun()

while nump<30:
    distances = np.round(np.random.uniform(0.5,3,2),2)
    distances = list(distances)
    # optimizer = COBYLA()
    vqe_degrees,exact_energies,vqe_energies,vqe_u=runone(distances)
    eraser=np.nonzero(oneoptimize(vqe_degrees))
    if len(eraser[0]):
        dag=circuit_to_dag(wavefunction)
        dag.remove_op_node(dag.op_nodes()[eraser[0][0]])
        wavefunction=dag_to_circuit(dag)
    nump+=1
    print('iter',nump)
    
    


print("All energies have been calculated")
plt.plot(distances, exact_energies, label="Exact Energy")
plt.plot(distances, vqe_energies, label="VQE Energy")
plt.plot(distances, vqe_u, label="UCCSD")
plt.xlabel('Atomic distance (Angstrom)')
plt.ylabel('Energy')
plt.legend()
plt.savefig('H2-C5.png')
