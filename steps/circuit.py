################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import json
from typing import Dict, List, Optional, Union

import numpy as np
from zquantum.core.circuits import save_circuit
from zquantum.core.openfermion import load_qubit_operator
from zquantum.core.serialization import load_array
from zquantum.core.utils import create_object, load_from_specs, load_list

Specs = Union[str, Dict]


def build_qaoa_ansatz_circuit(
    ansatz_specs: Specs,
    cost_hamiltonian: Union[str, List],
    mixer_hamiltonian: Union[str, List] = None,
    params: Optional[Union[str, List]] = None,
    thetas: Optional[Union[str, List]] = None,
):

    if isinstance(ansatz_specs, str):
        DeprecationWarning(
            "Loading ansatz_specs as a string will be depreciated in future, please "
            "change it to a dictionary."
        )
        ansatz_specs = json.loads(ansatz_specs)

    cost_hamiltonian = load_qubit_operator(cost_hamiltonian)
    ansatz_specs["cost_hamiltonian"] = cost_hamiltonian
    if mixer_hamiltonian:
        mixer_hamiltonian = load_qubit_operator(mixer_hamiltonian)
        ansatz_specs["mixer_hamiltonian"] = mixer_hamiltonian
    if thetas:
        thetas = np.array(load_list(thetas))
        ansatz_specs["thetas"] = thetas

    ansatz = load_from_specs(ansatz_specs)
    if params is not None:
        if isinstance(params, str):
            params = load_array(params)
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
