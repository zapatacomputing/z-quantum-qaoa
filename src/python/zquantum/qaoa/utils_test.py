import unittest
from openfermion import QubitOperator
from .utils import create_all_x_mixer_hamiltonian


class TestUtils(unittest.TestCase):
    def test_create_all_x_mixer_hamiltonian(self):
        # Given
        number_of_qubits = 4
        target_operator = (
            QubitOperator("X0")
            + QubitOperator("X1")
            + QubitOperator("X2")
            + QubitOperator("X3")
        )

        # When
        operator = create_all_x_mixer_hamiltonian(number_of_qubits)

        # Then
        self.assertEqual(operator, target_operator)
