from mpi4py import MPI
import numpy as np
from discretesampling.base.algorithms.smc_components.distributed_fixed_size_redistribution.prefix_sum import inclusive_prefix_sum
from discretesampling.base.algorithms.smc_components.variable_size_redistribution import variable_size_redistribution


def check_stability(ncopies):
    comm = MPI.COMM_WORLD
    loc_n = len(ncopies)
    N = loc_n * comm.Get_size()
    rank = comm.Get_rank()

    sum_of_ncopies = np.array(1, dtype=ncopies.dtype)
    comm.Allreduce(sendbuf=[np.sum(ncopies), MPI.INT], recvbuf=[sum_of_ncopies, MPI.INT], op=MPI.SUM)

    if sum_of_ncopies != N:
        # Find the index of the last particle to be copied
        idx = np.where(ncopies > 0)
        idx = idx[0][-1]+rank*loc_n if len(idx[0]) > 0 else np.array([-1])
        idx_MPI_dtype = MPI._typedict[idx.dtype.char]
        max_idx = np.zeros(1, dtype=idx.dtype)
        comm.Allreduce(sendbuf=[idx, idx_MPI_dtype], recvbuf=[max_idx, idx_MPI_dtype], op=MPI.MAX)
        # Find the core which has that particle, and increase/decrease its ncopies[i] till sum_of_ncopies == N
        if rank*loc_n <= max_idx <= (rank + 1)*loc_n - 1:
            ncopies[max_idx - rank*loc_n] -= sum_of_ncopies - N

    return ncopies


def get_number_of_copies(logw, rng):
    comm = MPI.COMM_WORLD
    N = len(logw) * comm.Get_size()

    cdf = inclusive_prefix_sum(np.exp(logw)*N)
    cdf_of_i_minus_one = cdf - np.reshape(np.exp(logw) * N, newshape=cdf.shape)

    u = np.array(rng.uniform(0.0, 1.0), dtype=logw.dtype)
    comm.Bcast(buf=[u, MPI._typedict[u.dtype.char]], root=0)

    ncopies = (np.ceil(cdf - u) - np.ceil(cdf_of_i_minus_one - u)).astype(int)

    ncopies = check_stability(ncopies)

    return ncopies #.astype(int)


def systematic_resampling(particles, logw, rng):
    loc_n = len(logw)
    N = loc_n * MPI.COMM_WORLD.Get_size()

    ncopies = get_number_of_copies(logw.astype('float32'), rng)
    particles = variable_size_redistribution(particles, ncopies)
    logw = np.log(np.ones(loc_n) / N)

    return particles, logw
