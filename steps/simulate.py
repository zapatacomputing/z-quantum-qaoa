import numpy as np

def run_circuit_and_get_distribution(ansatz_specs, params, backend_specs):
    ansatz = create_object(ansatz_specs)

    parameters = np.array(parameters)
    circuit = ansatz._generate_circuit(parameters)

    backend = create_object(backend_specs)
    bitstring_distribution = backend.get_bitstring_distribution(circuit)

    dist = bitstring_distribution.distribution_dict
    return dist
