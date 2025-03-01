import numpy as np
from qiskit.algorithms.optimizers import SPSA
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.drivers import MethodType, PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.settings import settings

from dft_embedding_solver import DFTEmbeddingSolver

settings.tensor_unwrapping = False
settings.use_pauli_sum_op = False
settings.use_symmetry_reduced_integrals = True

def run_molecule_simulation(name, atom_str):
    print(f"Molecule: {name}")
    print(f"Atomic Coordinates: {atom_str}\n")
    
    omega = 1.0
    driver = PySCFDriver(
        atom=atom_str,
        basis="6-31g*",
        method=MethodType.RKS,
        xc_functional=f"ldaerf + lr_hf({omega})",
        xcf_library="xcfun",
    )
    
    active_space = ActiveSpaceTransformer(4, 4)
    mapper = ParityMapper(num_particles=(2, 2))
    converter = QubitConverter(mapper=mapper)
    estimator = Estimator()
    ansatz = TwoLocal(4, ['ry', 'rz'], 'cz', reps=4, entanglement='linear')
    optimizer = SPSA(maxiter=600)
    vqe = VQE(estimator, ansatz, optimizer)
    vqe.filter_criterion = lambda state, val, aux: np.isclose(aux["ParticleNumber"][0], 4.0)
    algo = GroundStateEigensolver(converter, vqe)
    dft_solver = DFTEmbeddingSolver(active_space, algo)
    result = dft_solver.solve(driver, omega)
    print(result)

if __name__ == "__main__":
    molecules = {
    "CH2NH--H2O": "C -1.25287800 -0.56100700 0.02006900;H -2.29015400 -0.89421300 -0.02315300;H -0.47861700 -1.32268900 0.07987300;N -0.90210100 0.65073100 0.00389800;H -1.69982300 1.28064600 -0.05584700;H 1.05442400 0.43647600 -0.03364900;O 1.84392400 -0.12431000 -0.07451100;H 2.49475100 0.30519200 0.48116300",
    "CH2NH": "C 0.056287 0.581426 0.000000;H -0.844298 1.198425 0.000000;H 1.008536 1.107687 0.000000;N 0.056287 -0.678726 0.000000;H -0.895964 -1.043591 0.000000",
    "CH2O---NH3": "C 0.00000000 1.10058500 0.00000000;H 0.58057500 1.15742200 0.93373000;H 0.58057500 1.15742200 -0.93373000;N 0.86970200 -1.57900100 0.00000000;H 1.21429200 -2.07357000 -0.81381200;H 1.21429200 -2.07357000 0.81381200;O -1.19226400 0.99458000 0.00000000;H -0.13953600 -1.67484600 0.00000000",
    "CH2O": "C 0.00000000 -0.52447100 0.00000000;H -0.93797900 -1.10444600 0.00000000;H 0.93797800 -1.10444800 0.00000000;O 0.00000000 0.66946500 0.00000000",
    "Water": "H -0.02110 -0.00200 0.00000;O 0.83450 0.45190 0.00000;H 1.47690 -0.27300 0.00000",
    "NH3": "N 0.00000000 0.00000000 0.11344200;H 0.00000000 0.93915700 -0.26469800;H -0.81333400 -0.46957900 -0.26469800;H 0.81333400 -0.46957900 -0.26469800",
    "TS1": "C 0.33637300 0.67471600 0.02082100;H 0.29127600 1.23488300 0.95406000;H 0.16603700 1.27530300 -0.86564000;N 1.07737800 -0.44447500 -0.12130200;H 1.40511100 -0.77001800 0.78306100;H -0.21668600 -0.86183800 -0.24843300;O -1.20368200 -0.18360000 -0.08593900;H -1.57616600 -0.34649600 0.78864700",
    "TS2": "C -0.03598900 0.54626600 0.04282600;H -0.14847100 1.12461600 0.96255600;H -0.01039800 1.24136800 -0.79573100;N 1.19209500 -0.24374900 0.11077200;H 0.95820500 -1.20847200 -0.08967400;H 1.85018300 0.04433400 -0.60086700;O -1.16909100 -0.26590200 -0.15009100;H -1.42553000 -0.64598700 0.69208700",
    "TS3": "C 0.00000000 0.66943500 0.00000000;H 0.27911700 1.24168300 0.89795900;H 0.27911700 1.24168300 -0.89795900;N 0.89042000 -0.62874600 0.00000000;H 1.44033500 -0.79699100 -0.83597100;H 1.44033500 -0.79699100 0.83597100;O -1.18311200 0.06227600 0.00000000;H -0.20694900 -1.00297100 0.00000000",
    "NH2CH2OH-1": "C -0.03408900 0.53637700 0.04810600;H -0.07386700 1.07583700 0.99620800;H -0.07886200 1.25906300 -0.76295800;N 1.22008500 -0.15859300 -0.01980500;H 1.35295500 -0.78152900 0.76613300;H 1.28647300 -0.70238700 -0.87072100;O -1.19277200 -0.26399400 -0.11507500;H -1.28058700 -0.84714400 0.64193400",
    "NH2CH2OH-2": "C 0.04291000 0.53866700 0.01615000;H 0.03012400 1.19291300 -0.85997400;H 0.08716800 1.16186200 0.91177900;N -1.12467700 -0.31425600 -0.06016000;H -1.34225400 -0.72495800 0.83867400;H -1.93757000 0.18982900 -0.38714400;O 1.20850400 -0.24093200 0.05448200;H 1.10978500 -0.92440300 -0.61497200"
}
    
    for name, atom_str in molecules.items():
        run_molecule_simulation(name, atom_str)
