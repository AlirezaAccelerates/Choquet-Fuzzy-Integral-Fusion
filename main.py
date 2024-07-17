import numpy as np
import itertools
from cvxopt import solvers, matrix

class ChoquetIntegral:
    
    def __init__(self):
        """
        Initialize a ChoquetIntegral instance.
        
        This sets up the Choquet Integral (ChI) without any input parameters.
        You can pass your own values instead of learning from data.
        
        Example:
            chi = ChoquetIntegral()
        """
        self.trainSamples, self.trainLabels = [], []
        self.testSamples, self.testLabels = [], []
        self.N, self.numberConstraints, self.M = 0, 0, 0
        self.g = 0
        self.fm = []
        self.type = []

    def fit(self, x1, l1):
        """
        Train this instance of the ChoquetIntegral with given samples and labels.
        
        :param x1: Training samples of size N x M (inputs x number of samples)
        :param l1: Training labels of size 1 x M (label per sample)
        :return: self
        """
        self.type = 'quad'
        self.trainSamples = x1
        self.trainLabels = l1
        self.N = self.trainSamples.shape[0]
        self.M = self.trainSamples.shape[1]
        print("Number Inputs: ", self.N, "; Number Samples: ", self.M)
        self.fm = self.produce_lattice()

        return self

    def predict(self, x2):
        """
        Produce an output for the given test sample using the trained Choquet Integral.
        
        :param x2: Testing sample
        :return: Output of the Choquet integral
        """
        if self.type == 'quad':
            n = len(x2)
            pi_i = np.argsort(x2)[::-1][:n] + 1
            ch = x2[pi_i[0] - 1] * (self.fm[str(pi_i[:1])])
            for i in range(1, n):
                latt_pti = np.sort(pi_i[:i + 1])
                latt_ptimin1 = np.sort(pi_i[:i])
                ch += x2[pi_i[i] - 1] * (self.fm[str(latt_pti)] - self.fm[str(latt_ptimin1)])
            return ch
        else:
            print("If using Sugeno measure, you need to use chi_sugeno.")

    def produce_lattice(self):
        """
        Build the lattice (or FM variables) by solving a quadratic program.
        
        :return: Learned FM variables (Lattice)
        """
        fm_len = 2 ** self.N - 1
        E = np.zeros((fm_len, fm_len))
        L = np.zeros(fm_len)
        index_keys = self.get_keys_index()
        
        for i in range(self.M):
            l = self.trainLabels[i]
            fm_coeff = self.get_fm_class_img_coeff(index_keys, self.trainSamples[:, i], fm_len)
            L += (-2) * l * fm_coeff
            E += np.outer(fm_coeff, fm_coeff)

        G, h, A, b = self.build_constraint_matrices(index_keys, fm_len)
        sol = solvers.qp(matrix(2 * E, tc='d'), matrix(L.T, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'), matrix(A, tc='d'), matrix(b, tc='d'))

        g = sol['x']
        Lattice = {key: g[index_keys[key]] for key in index_keys}
        return Lattice

    def build_constraint_matrices(self, index_keys, fm_len):
        """
        Build the necessary constraint matrices for the quadratic program.
        
        :param index_keys: Dictionary to reference lattice components
        :param fm_len: Length of the fuzzy measure
        :return: Constraint matrices (G, h, A, b)
        """
        vls = np.arange(1, self.N + 1)
        G = np.zeros((1, fm_len))
        G[0, index_keys[str(np.array([1]))]] = -1.
        h = np.zeros((1, 1))
        
        for i in range(2, self.N + 1):
            line = np.zeros(fm_len)
            line[index_keys[str(np.array([i]))]] = -1.
            G = np.vstack((G, line))
            h = np.vstack((h, np.array([0])))

        for i in range(2, self.N + 1):
            parent = np.array(list(itertools.combinations(vls, i)))
            for latt_pt in parent:
                for j in range(len(latt_pt) - 1, len(latt_pt)):
                    children = np.array(list(itertools.combinations(latt_pt, j)))
                    for latt_ch in children:
                        line = np.zeros(fm_len)
                        line[index_keys[str(latt_ch)]] = 1.
                        line[index_keys[str(latt_pt)]] = -1.
                        G = np.vstack((G, line))
                        h = np.vstack((h, np.array([0])))

        line = np.zeros(fm_len)
        line[index_keys[str(vls)]] = 1.
        G = np.vstack((G, line))
        h = np.vstack((h, np.array([1])))

        A = np.zeros((1, fm_len))
        A[0, -1] = 1
        b = np.array([1])

        return G, h, A, b

    def get_fm_class_img_coeff(self, Lattice, h, fm_len):
        """
        Create a FM map with the name as the key and the index as the value.
        
        :param Lattice: Dictionary with FM
        :param h: Sample
        :param fm_len: Length of the fuzzy measure
        :return: FM coefficients
        """
        n = len(h)
        fm_coeff = np.zeros(fm_len)
        pi_i = np.argsort(h)[::-1][:n] + 1
        for i in range(1, n):
            fm_coeff[Lattice[str(np.sort(pi_i[:i]))]] = h[pi_i[i - 1] - 1] - h[pi_i[i] - 1]
        fm_coeff[Lattice[str(np.sort(pi_i[:n]))]] = h[pi_i[n - 1] - 1]
        np.matmul(fm_coeff, np.transpose(fm_coeff))
        return fm_coeff

    def get_keys_index(self):
        """
        Set up a dictionary for referencing FM.
        
        :return: Dictionary with keys and indices for FM
        """
        vls = np.arange(1, self.N + 1)
        count = 0
        Lattice = {}
        for i in range(self.N):
            Lattice[str(np.array([vls[i]]))] = count
            count += 1
        for i in range(2, self.N + 1):
            A = np.array(list(itertools.combinations(vls, i)))
            for latt_pt in A:
                Lattice[str(latt_pt)] = count
                count += 1
        return Lattice
