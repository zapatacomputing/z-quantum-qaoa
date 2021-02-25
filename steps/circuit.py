import numpy as np
from zquantum.core.circuit import save_circuit, Circuit, load_circuit_template_params
from zquantum.core.openfermion import load_qubit_operator
from zquantum.core.utils import create_object, load_from_specs
from typing import Union, List, Optional, Dict
import json

Specs = Union[str, Dict]


def build_qaoa_ansatz_circuit(
    ansatz_specs: Specs,
    cost_hamiltonian: Union[str, List],
    mixer_hamiltonian: Union[str, List] = None,
    params: Optional[Union[str, List]] = None,
):

    if isinstance(ansatz_specs, str):
        DeprecationWarning(
            "Loading ansatz_specs as a string will be depreciated in future, please change it to a dictionary."
        )
        ansatz_specs = json.loads(ansatz_specs)

    cost_hamiltonian = load_qubit_operator(cost_hamiltonian)
    if mixer_hamiltonian:
        mixer_hamiltonian = load_qubit_operator(mixer_hamiltonian)
    ansatz_specs["cost_hamiltonian"] = cost_hamiltonian
    ansatz_specs["mixer_hamiltonian"] = mixer_hamiltonian
    ansatz = load_from_specs(ansatz_specs)
    if params is not None:
        if isinstance(params, str):
            params = load_circuit_template_params(params)
        else:
            params = np.array(params)
        circuit = ansatz.get_executable_circuit(params)
    elif ansatz.supports_parametrized_circuits:
        circuit = ansatz.parametrized_circuit
    else:
        raise (
            Exception(
                "Ansatz is not parametrizable and no parameters has been provided."
            )
        )
    save_circuit(circuit, "circuit.json")
