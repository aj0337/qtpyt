from math import pi
from typing import Any

from qtpyt import xp
from qtpyt.base._kernels import get_lambda
from qtpyt.base.greenfunction import BaseGreenFunction
from qtpyt.block_tridiag import solvers
from qtpyt.block_tridiag.btmatrix import BTMatrix, empty_buffer


class GreenFunction(BaseGreenFunction):
    """Block Tridiagonal Green's Function.

    Allows to compute observables for a scattering region
    embedded in an environment described by selfenergies.
    The Hamiltonian and the Overlap are given as two lists
    containing the diagonal and upper diagonal block-matrices.

    Args:
        hs_list_ii : (list(xp.ndarray(complex, shape=(2,m,m))), len=N)
            List of block matrices on the diagonal.

                hs_list_ii[i][0] = Hamilton-block[0,0]
                hs_list_ii[i][1] = Overlap-block[0,0]

        hs_list_ii : (list(xp.ndarray(complex, shape=(2,m,m))), len=N-1)
            List of block matrices on the upper diagonal.

                hs_list_ij[i][0] = Hamilton-block[0,1]
                hs_list_ij[i][1] = Overlap-block[0,1]

        selfenergies : (list(block, indices, selfenergy))
            List of selfenergies that couple to this Hamiltonian.

                Hamiltonian[block, block][indices] << selfenergy

        solver : ('str')
            One of the solvers available in Solver given as string.
            Used to compute selected parts of the Green's function.
            (i.e. invert the matrix.)

    """

    def __init__(
        self,
        hs_list_ii,
        hs_list_ij,
        selfenergies=[],
        idxleads=[0, 1],
        eta=1e-5,
        mu=0.0,
        kt=0.0,
        solver="spectral",
    ):
        Solver = solvers.__dict__.get(solver.capitalize(), None)
        if Solver is None:
            raise RuntimeError(f"Invalid solver {solver}")
        self.solver = Solver(self)

        self.hs_list_ii = hs_list_ii
        self.hs_list_ij = hs_list_ij

        # Here the Hamiltonian and Overlap may be on GPU.
        def _fill(buffer, i):
            buffer.m_qii[0] = xp.asarray(self.hs_list_ii[0][i])
            for q in range(1, buffer.N):
                buffer.m_qii[q] = xp.asarray(self.hs_list_ii[q][i])
                buffer.m_qij[q - 1] = xp.asarray(self.hs_list_ij[q - 1][i])
                buffer.m_qji[q - 1] = buffer.m_qij[q - 1].T  # reference

        N = len(hs_list_ii)
        S = empty_buffer(N)
        H = empty_buffer(N)

        _fill(S, 1)
        _fill(H, 0)

        # S = BTMatrix(S.m_qii, S.m_qij, S.m_qji)
        # H = BTMatrix(H.m_qii, H.m_qij, H.m_qji)

        self.S = BTMatrix(S.m_qii, S.m_qij, S.m_qji)
        self.H = BTMatrix(H.m_qii, H.m_qij, H.m_qji)

        self.Ginv = empty_buffer(N)

        self.selfenergies = selfenergies
        self.idxleads = idxleads
        self.gammas = [None] * len(idxleads)

        self.H = self.H
        self.S = self.S

        super().__init__(H, S, eta, mu, kt)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "eta":
            for _, selfenergy in self.selfenergies:
                selfenergy.eta = value
        super().__setattr__(name, value)

    def get_Ginv(self, energy):
        z = self.z(energy)

        for q, (h_ii, s_ii) in enumerate(zip(self.H.m_qii, self.S.m_qii)):
            self.Ginv.m_qii[q] = z * s_ii - h_ii

        # Upper and lower diagonals.
        for q, (h_ij, s_ij) in enumerate(zip(self.H.m_qij, self.S.m_qij)):
            self.Ginv.m_qij[q] = z * s_ij - h_ij
            self.Ginv.m_qji[q] = z * s_ij.T.conj() - h_ij.T.conj()

        # Add selfenergies
        for i, (block, selfenergy) in enumerate(self.selfenergies):
            # Possibly offload to GPU.
            sigma = xp.asarray(selfenergy.retarded(energy))
            self.Ginv.m_qii[block] -= sigma
            if i in self.idxleads:
                self.gammas[i] = get_lambda(sigma)

        return self.Ginv

    def get_transmission(self, energy, ferretti=True):
        return self.solver.get_transmission(energy, ferretti)

    def get_spectrals(self, energy):
        return self.solver.get_spectrals(energy)

    def retarded(self, energy, inverse=False):
        return self.solver.get_retarded(energy)

    def get_dos(self, energy):
        G = self.solver.get_retarded(energy)
        return -1 / pi * G.dottrace(self.S).imag

    def get_pdos(self, energy):
        G = self.solver.get_retarded(energy)
        return -1 / pi * G.dotdiag(self.S).imag

    # def add_screening(self, V):
    #     if not hasattr(self, 'V'):
    #         self.V = np.zeros(self.nbf)
    #     assert V.size == self.nbf
    #     #Add screening and remove (if exists) current.
    #     h_qii = self.hs_list_ii[0]
    #     if sum(self.V) != 0:
    #         add_diagonal(h_qii, - self.V)
    #     self.V[:] = V
    #     add_diagonal(h_qii, self.V)

    # def remove_screening(self):
    #     h_qii = self.hs_list_ii[0]
    #     add_diagonal(h_qii, -self.V)
    #     self.V[:] = 0.
