from qtpyt import xp
from qtpyt.base._kernels import dagger
from qtpyt.block_tridiag.btmatrix import BTMatrix

# from qtpyt.block_tridiag.greenfunction import GreenFunction
from qtpyt.block_tridiag.recursive import (
    coupling_method_N1,
    dyson_method,
    spectral_method,
)


class Solver:
    """The parent class of a generic solver.

    The principal task of Solver is to setup the inverse Green's
    function matrix which is then inverted by the child solver.
    It also offloas the Hamiltonian and Overlap matrices to GPU.

    """

    def __init__(self, gf, method: str):
        self.gf = gf
        self.method = method

    @property
    def energy(self):
        return self.gf.energy

    @energy.setter
    def energy(self, energy):
        self.gf.energy = energy

    def inv(self, Ginv, *args, **kwargs):
        return self.method(Ginv.m_qii, Ginv.m_qij, Ginv.m_qji, *args, **kwargs)

    def get_transmission(self, energy, *args, **kwargs):
        raise NotImplementedError(
            "{self.__class__.__name__} does not implement transmission."
        )

    def get_retarded(self, energy):
        raise NotImplementedError(
            "{self.__class__.__name__} does not implement retarded."
        )

    def get_spectrals(self, energy):
        raise NotImplementedError(
            "{self.__class__.__name__} does not implement spectrals."
        )


class Spectral(Solver):
    def __init__(self, gf):
        super().__init__(gf, spectral_method)
        self.A1 = None
        self.A2 = None

    def get_spectrals(self, energy):
        if self.energy != energy:
            self.energy = energy
            self.A1, self.A2 = map(
                lambda A: BTMatrix(*A),
                self.inv(
                    self.gf.get_Ginv(energy), self.gf.gammas[0], self.gf.gammas[1]
                ),
            )
        return self.A1, self.A2

    def get_transmission(self, energy):
        A2 = self.get_spectrals(energy)[1]
        gamma_L = self.gf.gammas[0]
        T_e = gamma_L.dot(A2[0, 0]).real.trace()
        return T_e

    def get_retarded(self, energy):
        A1, A2 = self.get_spectrals(energy)
        return A1 + A2


class Coupling(Solver):
    def __init__(self, gf):
        super().__init__(gf, coupling_method_N1)

    def get_transmission(self, energy):
        if self.energy != energy:
            self.energy = energy
            Ginv = self.gf.get_Ginv(energy)
            g_N1 = self.inv(Ginv)
            gamma_L = self.gf.gammas[0]
            gamma_R = self.gf.gammas[1]
            T_e = xp.einsum(
                "ij,jk,kl,lm->im", gamma_R, g_N1, gamma_L, dagger(g_N1), optimize=True
            ).real.trace()
        return T_e


class Dyson(Solver):
    def __init__(self, gf, trans=True):
        super().__init__(gf, dyson_method)
        self.G = None
        self.g_1N = None
        self.trans = trans

    def get_retarded(self, energy):
        if self.energy != energy:
            self.energy = energy
            Ginv = self.gf.get_Ginv(energy)
            self.G = None
            if self.trans:
                g_1N, G = self.inv(Ginv, trans=True)
                self.g_1N = g_1N
                self.G = BTMatrix(*G)
            else:
                G = self.inv(Ginv, trans=False)
                self.G = BTMatrix(*G)
        return self.G

    # def get_transmission(self, energy):
    #     _ = self.get_retarded(energy)
    #     gamma_L = self.gf.gammas[0]
    #     gamma_R = self.gf.gammas[1]
    #     T_e = xp.einsum(
    #         "ij,jk,kl,lm->im",
    #         gamma_L,
    #         self.g_1N,
    #         gamma_R,
    #         dagger(self.g_1N),
    #         optimize=True,
    #     ).real.trace()
    #     return T_e

    def get_transmission(self, energy, ferretti):
        _ = self.get_retarded(energy)
        gamma_L = self.gf.gammas[0]
        gamma_R = self.gf.gammas[1]

        if (len(self.gf.idxleads) == len(self.gf.selfenergies)) or (not (ferretti)):
            print("No Ferretti correction")
            T_e = xp.einsum(
                "ij,jk,kl,lm->im",
                gamma_L,
                self.g_1N,
                gamma_R,
                dagger(self.g_1N),
                optimize=True,
            ).real.trace()
            return T_e

        # Ferretti correction
        print("Ferretti correction")
        delta = xp.zeros_like(self.gf.get_Ginv(energy), dtype=complex)
        for i, (indices, selfenergy) in enumerate(self.gf.selfenergies):
            if i not in self.gf.idxleads:
                sigma = selfenergy.retarded(energy)
                delta[indices] += 1.0j * (sigma - sigma.T.conj())

        # Solve for delta using the same system:
        delta[:] = xp.linalg.solve(
            gamma_L + gamma_R + 2 * self.gf.eta * self.gf.S, delta
        )
        delta.flat[:: len(delta) + 1] += 1.0

        # Compute transmission using delta correction
        T_e = xp.einsum(
            "ij,jk,kl,lm,mn->in",
            gamma_L,
            self.g_1N,
            gamma_R,
            delta,
            dagger(self.g_1N),
            optimize=True,
        ).real.trace()

        return T_e
