import numpy as np
from mpi4py import MPI
from discretesampling.domain.decision_tree.tree import Tree


def pad(x):
    """
    Description
    -----------
    This function computes the size of the biggest particle, and extend the other particles with NaNs until all
    particles have the same size

    :param x: particles organized as a list of lists
    :return x_new: particle organized as a numpy 2D array
    """
    def max_dimension(x):
        comm = MPI.COMM_WORLD
        local_max = np.max(x)
        max_dim = np.zeros_like(1, dtype=local_max.dtype)
        comm.Allreduce(sendbuf=[local_max, MPI.INT], recvbuf=[max_dim, MPI.INT], op=MPI.MAX)
        return max_dim

    def encode_move(last):
        if last == "grow":
            return 0
        elif last == "prune":
            return 1
        elif last == "swap":
            return 2
        elif last == "change":
            return 3
        else:
            return -1

    trees = [np.array(particle.tree).flatten() for particle in x]
    leaves = [np.array(particle.leafs).flatten() for particle in x]
    last_actions = [encode_move(particle.lastAction) for particle in x]
    my_tree_dims = np.array([len(tree) for tree in trees])
    my_leaf_dims = np.array([len(leaf) for leaf in leaves])
    max_tree = max_dimension(x=my_tree_dims)
    max_leaf = max_dimension(x=my_leaf_dims)

    x_new = np.array([np.hstack((tree, np.repeat(-1.0, max_tree - len(tree)), last_action,
                                 leaf, np.repeat(-1.0, max_leaf - len(leaf)), max_tree))
                      for tree, last_action, leaf in zip(trees, last_actions, leaves)])
    return np.hstack((x_new, np.atleast_2d(my_tree_dims).transpose(), np.atleast_2d(my_leaf_dims).transpose()))



def restore(x, particles):
    def decode_move(code):
        if code == 0:
            return "grow"
        elif code == 1:
            return "prune"
        elif code == 2:
            return "swap"
        elif code == 3:
            return "change"
        else:
            return ""

    def extract_tree(tree):
        return [tree[i:i+4].astype(int).tolist() + [tree[i + 4]] + [tree[i + 5].astype(int)] for i in range(0, len(tree.tolist()), 6)]

    def extract_leafs(leaves):
        return leaves.tolist()

    my_leaf_dim = x[:, -1].astype(int)
    my_tree_dim = x[:, -2].astype(int)
    max_tree = int(x[0, -3])
    return [Tree(particles[0].X_train, particles[0].y_train, extract_tree(x[i, 0:my_tree_dim[i]]),
                 extract_leafs(x[i, max_tree+1:max_tree+1+my_leaf_dim[i]]), decode_move(x[i, max_tree]))
            for i in range(len(particles))]


def gather_all(particles):
    comm = MPI.COMM_WORLD
    loc_n = len(particles)
    N = loc_n * comm.Get_size()

    x = pad(particles)
    all_x = np.zeros([N, x.shape[1]], dtype='d')
    all_particles = [particles[0] for i in range(N)]

    comm.Allgather(sendbuf=[x, MPI.DOUBLE], recvbuf=[all_x, MPI.DOUBLE])

    all_particles = restore(all_x, all_particles)

    return all_particles

