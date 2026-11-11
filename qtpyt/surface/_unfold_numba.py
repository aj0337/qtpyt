"""

The algorithm can be described as follows:
    
1. First the elements in the graph below are filled. Starting from
block (0,0) and then proceed with 1,+-j blocks on the first row and
column.

2. A nested block toepliz algorithm is used to fill the rest. For each
block, first fill the upper diagonal entries (including rhe diagonals)
and then the lower diagonals. 

A block Toeplitz matrix has the form

A , B , C , D 
E , A , B , C
F , E , A , B
G , F , E , A

# The following is generated with web app Diagon

+---------+---------+-------+----------+                             +------------------+------------------+-------+-----------------+
| 0       |1*k_i[1] |       | n*k_i[1] |                             | 1*k_i[0]         | 1*k_i[0]+1*k_i[1]|       |1*k_i[0]+n*k_i[1]|
+---------+---------+-------+----------+                             +------------------+------------------+-------+-----------------+
|-1*k_i[1]|         |       |          |                             | 1*k_i[0]-1*k_i[1]|                  |       |                 |
+---------+---------+-------+----------+                             +------------------+------------------+-------+-----------------+
|         |         |       |          |                             |                  |                  |       |                 |
+---------+---------+-------+----------+                             +------------------+------------------+-------+-----------------+
|-n*k_i[1]|         |       |          |                             | 1*k_i[0]-n*k_i[1]|                  |       |                 |
+---------+---------+-------+----------+                             +------------------+------------------+-------+-----------------+

+-------------------+------------------+-------+-----------------+
| -1*k_i[0]         | 1*k_i[0]+1*k_i[1]|       |1*k_i[0]+n*k_i[1]|
+-------------------+------------------+-------+-----------------+
| -1*k_i[0]-1*k_i[1]|                  |       |                 |
+-------------------+------------------+-------+-----------------+
|                   |                  |       |                 |
+-------------------+------------------+-------+-----------------+
| -1*k_i[0]-n*k_i[1]|                  |       |                 |
+-------------------+------------------+-------+-----------------+

.
.
.

+-------------------+-------------------+-------+------------------+
| -m*k_i[0]         | -m*k_i[0]+1*k_i[1]|       |-m*k_i[0]+n*k_i[1]|
+-------------------+-- ----------------+-------+------------------+
| -m*k_i[0]-1*k_i[1]|                   |       |                  |
+-------------------+-------------------+-------+------------------+
|                   |                   |       |                  |
+-------------------+-------------------+-------+------------------+
| -m*k_i[0]-n*k_i[1]|                   |       |                  |
+-------------------+-------------------+-------+------------------+


"""
import numpy as np
from numba import njit, prange


@njit(
    "(c16[:,:,:],f8[:,:],i8,i8,c16[:,:,:,:])", cache=False, parallel=True, fastmath=True
)
def _bloch_unfold(A, kpts, m, n, out):
    """Unfold k-matrices to supercell.
    
    """

    nk = A.shape[0]
    nr = A.shape[1]
    nc = A.shape[2]

    out[:n, :, :, :] = complex(0.0)
    out[n:, :, :n, :] = complex(0.0)

    weigth = complex(1.0 / nk)
    for k in range(nk):
        mat = A[k]
        kpt = kpts[k]

        # (0,0)
        out[0, :, 0, :] += mat * weigth

        for r in prange(nr):
            phase_step_j = np.exp(2.0j * np.pi * kpt[1])

            phase = phase_step_j * weigth
            for j in range(1, n):
                conj_phase = phase.real - 1.0j * phase.imag  # .conj()
                # for r in range(nr):
                for c in range(nc):
                    # (0,+j)
                    out[0, r, j, c] += mat[r, c] * phase
                    # (0,-j)
                    out[j, r, 0, c] += mat[r, c] * conj_phase
                phase *= phase_step_j

            ###################################
            # NOTE Here 1D expansion is done! #
            ###################################

            phase_step_i = np.exp(2.0j * np.pi * kpt[0])

            phase_i = phase_step_i * weigth
            for i in range(1, m):
                conj_phase_i = phase_i.real - 1.0j * phase_i.imag  # .conj()
                I = i * n
                # for r in range(nr):
                for c in range(nc):
                    # (+i,0)
                    out[0, r, I, c] += mat[r, c] * phase_i
                    # (-i,0)
                    out[I, r, 0, c] += mat[r, c] * conj_phase_i

                phase_j = phase_step_j
                for j in range(1, n):
                    conj_phase_j = phase_j.real - 1.0j * phase_j.imag  # .conj()
                    phase_0 = phase_i * phase_j
                    phase_1 = phase_i * conj_phase_j
                    phase_2 = conj_phase_i * phase_j
                    phase_3 = conj_phase_i * conj_phase_j
                    J = I + j
                    # for r in range(nr):
                    for c in range(nc):
                        # (+i,+j)
                        out[0, r, J, c] += mat[r, c] * phase_0
                        # (+i,-j)
                        out[j, r, I, c] += mat[r, c] * phase_1
                        # (-i,+j)
                        out[I, r, j, c] += mat[r, c] * phase_2
                        # (-i,-j)
                        out[J, r, 0, c] += mat[r, c] * phase_3
                    phase_j *= phase_step_j

                phase_i *= phase_step_i

    for p in prange(m):
        I = p * n

        # (0,+j)
        for r in range(nr):
            for i in range(n - 1):
                for j in range(1, n - i):
                    out[j, r, I + i + j, :] = out[0, r, I + i, :]

            for i in range(n):
                for s in range(1, m - p):
                    J = s * n
                    for j in range(n - i):
                        out[j + J, r, I + i + j + J, :] = out[0, r, I + i, :]

        # (+j,0)
        for r in range(nr):
            for i in range(1, n - 1):
                for j in range(1, n - i):
                    out[i + j, r, I + j, :] = out[i, r, I, :]

            for i in range(1, n):
                for s in range(1, m - p):
                    J = s * n
                    for j in range(n - i):
                        out[i + j + J, r, I + j + J, :] = out[i, r, I, :]

    for p in prange(1, m):
        I = p * n

        # (0,+j)
        for r in range(nr):
            for i in range(n - 1):
                for j in range(1, n - i):
                    out[I + j, r, i + j, :] = out[I, r, i, :]

            for i in range(n):
                for s in range(1, m - p):
                    J = s * n
                    for j in range(n - i):
                        out[I + j + J, r, i + j + J, :] = out[I, r, i, :]

        # (+j,0)
        for r in range(nr):
            for i in range(1, n - 1):
                for j in range(1, n - i):
                    out[I + i + j, r, j, :] = out[i + I, r, 0, :]

            for i in range(1, n):
                for s in range(1, m - p):
                    J = s * n
                    for j in range(n - i):
                        out[I + i + j + J, r, j + J, :] = out[i + I, r, 0, :]
