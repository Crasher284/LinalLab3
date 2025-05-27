import numpy as np
import matplotlib.pyplot as plt

class Qubit:
    num = 0
    state = []
    
    def __init__(self, num, state):
        self.num = num
        if len(state) != np.pow(2, num):
            raise ValueError(f'Invalid state size: not 2^{num}.')
        self.state = np.array(state, dtype=complex)
        self.norm()

    def copy(self):
        return Qubit(self.num, self.state.copy())

    def norm(self):
        norm = np.linalg.norm(self.state)
        if norm == 0:
            raise ValueError("Invalid qubit state: norm cannot be 0.")
        self.state /= norm

    def bra(self):
        return self.state.conj()

    def ket(self):
        return self.state.reshape(-1, 1)

class Gate:
    num = 0
    state = []

    def __init__(self, num, state):
        self.num = num
        if len(state) != np.pow(2, num):
            raise ValueError(f'Invalid state size: not 2^{num}.')
        self.state = state

class Utils:

    def base(self, bit):
        match bit:
            case 0:
                return Qubit(1, [1, 0])
            case 1:
                return Qubit(1, [0, 1])
            case _:
                raise ValueError(f'"{bit}" - invalid index for base qubit.')

    def link(self, a, b):
        return Qubit(a.num + b.num, np.kron(a.ket(), b.ket()).flatten())

    def inner_product(self, one, other):
        return np.vdot(one.bra(), other.bra())
    
    def outer_product(self, one, other):
        return np.outer(one.ket(), other.bra())

    def pauli(self, num):
        match num:
            case 0:
                return Gate(1, np.array([[1, 0], [0, 1]], dtype=complex))
            case 1:
                return Gate(1, np.array([[0, 1], [1, 0]], dtype=complex))
            case 2:
                return Gate(1, np.array([[0, -1j], [1j, 0]], dtype=complex))
            case 3:
                return Gate(1, np.array([[1, 0], [0, -1]], dtype=complex))
            case _:
                raise ValueError(f'"{num}" - invalid index for Pauli gate.')
        
    def cnot(self):
        return Gate(2, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex))

    def hadamard(self):
        return Gate(1, np.array([[1, 1], [1, -1]], dtype=complex))

    def mul_hadamard(self, num):
        if num == 1:
            return self.hadamard()
        step = self.hadamard().state
        start = self.hadamard().state
        for i in range(num-1):
            start = np.kron(start, step)
        return Gate(num, start)

    def constant_oracle(self, num):
        return Gate(num, np.eye(np.pow(2, num), dtype=complex))

    def balanced_oracle(self):
        return self.cnot()

    def form_grover_oracle(self, num, state):
        n = 2**num
        oracle = np.eye(n, dtype=complex)
        for idx in state:
            if not (0 <= idx < n):
                raise ValueError(f"Target state index {idx} is out of range for system of {num} qubits.")
            oracle[idx, idx] = -1
        return Gate(num, oracle)

    def form_diffusion(self, num):
        state = np.zeros(np.pow(2, num), dtype=complex)
        state[0] = 1.0
        qubit = Qubit(num, state)
        qubit = self.apply(qubit, self.mul_hadamard(num))
        out = np.eye(np.pow(2, qubit.num), dtype = complex)
        phase = self.outer_product(qubit, qubit)
        return Gate(num, 2 * phase - out)

    def apply(self, qubit, gate):
        if qubit.num!=gate.num:
            raise ValueError(f'Unequal sizes: qubit system has {qubit.num}, and gate - {gate.num}.')
        out = Qubit(qubit.num, gate.state @ qubit.state)
        out.norm()
        return out

    def grover_algorithm(self, num, target):
        n = 2**num
        m = len(target)
        if m == 0 or m > n:
            raise ValueError("Invalid number of target states.")
        
        init = np.zeros(n, dtype=complex)
        init[0] = 1.0
        start = Qubit(num, init)
        start = self.apply(start, self.mul_hadamard(num))

        current = start.copy()

        its = int(np.floor(np.pi / 4 * np.sqrt(n / m)))
        
        oracle = self.form_grover_oracle(num, target)
        diffusion = self.form_diffusion(num)

        probs = [self.probs(current)]
        
        for _ in range(its):
            current = self.apply(current, oracle)
            current = self.apply(current, diffusion)
            probs.append(self.probs(current))
        
        return current, probs

    def probs(self, qubit):
        return np.abs(qubit.state)**2

utils = Utils()

num = 3
target = [5]
result, probs = utils.grover_algorithm(num, target)
names = []
for i in range(np.pow(2, num)):
    names.append("|"+bin(i)[2:]+">")
print("Probabilities per iteration:")
for i, prs in enumerate(probs):
    print(f"Iteration {i}:", dict(zip(names, prs)))

its = list(range(len(probs)))
n = len(names)
probs = np.array(probs)

plt.figure(figsize=(8, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
bottom = np.zeros(len(its))

for i in range(n):
    plt.bar(its, probs[:, i], bottom=bottom, label=names[i], color=colors[i])
    bottom += probs[:, i]

plt.xlabel('Iteration')
plt.ylabel('Probability')
plt.title('Grover algorithm: probability changes')
plt.xticks(its, [f'Iter {i}' for i in its])
plt.legend()
plt.tight_layout()
plt.show()
