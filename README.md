# z-quantum-qaoa

## What is it?

`z-quantum-qaoa` is a module with basic implementation of Quantum Approximate Optimization Algorithm (QAOA) to be used with [Orquestra](https://www.zapatacomputing.com/orquestra/) – platform for performing computations on quantum computers developed by [Zapata Computing](https://www.zapatacomputing.com).

Currently it includes the following features:
- Creating ansatz from the [Farhi's 2014 paper](https://arxiv.org/abs/1411.4028).
- Creating Hamiltonian for solving MaxCut problem given a graph.
- Solving MaxCut problem using exhaustive search.

## Usage

### Workflow
In order to use `z-quantum-qaoa` in your workflow, you need to add it as a resource:

```
resources:
- name: z-quantum-qaoa
  type: git
  parameters:
    url: "git@github.com:zapatacomputing/z-quantum-qaoa.git"
    branch: "master"
```

and then import in a specific step:

```
- - name: my-task
    template: template-1
    arguments:
      parameters:
      - param_1: 1
      - resources: [z-quantum-qaoa]
```

Once it's done you can:
- use any template from `templates/` directory
- use tasks which import `zquantum.qaoa` in the python code (see below).

### Python

Here's an example how to do use methods from `z-quantum-qaoa` in a python task:

```python
from zquantum.qaoa.ansatz import build_farhi_qaoa_circuit_template
from qeopenfermion import load_qubit_operator
hamiltonian = load_qubit_operator('hamiltonian.json')
ansatz = build_farhi_qaoa_circuit_template(hamiltonian)
```

Even though it's intended to be used with Orquestra, you can also use it as a standalone python module.
In order to install it run `pip install .` from the `src` directory.


## Development and contribution

You can find the development guidelines in the [`z-quantum-core` repository](https://github.com/zapatacomputing/z-quantum-core).

### Running tests

In order to run tests please run `pytest .` from the main directory.
