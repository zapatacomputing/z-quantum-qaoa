from openfermion import QubitOperator


def create_all_x_mixer_hamiltonian(number_of_qubits):
    mixer_hamiltonian = QubitOperator()
    for i in range(number_of_qubits):
        mixer_hamiltonian += QubitOperator((i, "X"))
    return mixer_hamiltonian
