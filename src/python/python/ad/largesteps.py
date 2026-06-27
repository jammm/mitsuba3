from __future__ import annotations as __annotations__ # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr

def mesh_laplacian(n_verts, faces, lambda_):
    """
    Compute the index and data arrays of the (combinatorial) Laplacian matrix of
    a given mesh.
    """
    import numpy as np

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = np.unique(np.stack([np.concatenate([ii, jj]), np.concatenate([jj, ii])], axis=0), axis=1)
    adj_values = np.ones(adj.shape[1], dtype=np.float64) * lambda_

    # Diagonal indices, duplicated as many times as the connectivity of each index
    diag_idx = np.stack((adj[0], adj[0]), axis=0)

    diag = np.stack((np.arange(n_verts), np.arange(n_verts)), axis=0)

    # Build the sparse matrix
    idx = np.concatenate((adj, diag_idx, diag), axis=1)
    values = np.concatenate((-adj_values, adj_values, np.ones(n_verts)))

    return idx, values

class SolveCholesky(dr.CustomOp):
    """
    DrJIT custom operator to solve a linear system using a Cholesky factorization.
    """

    def solve(self, u):
        x = dr.empty(mi.TensorXf, shape=u.shape)
        result = self.solver.solve(u, x)
        return x if result is None else result

    def eval(self, solver, u):
        self.solver = solver
        return mi.TensorXf(self.solve(u))

    def forward(self):
        self.set_grad_out(self.solve(self.grad_in('u')))

    def backward(self):
        self.set_grad_in('u', self.solve(self.grad_out()))

    def name(self):
        return "Cholesky solve"


class AMDConjugateGradientSolver:
    """
    GPU-side iterative solver used when cholespy cannot consume AMD Dr.Jit
    arrays. The system is symmetric positive definite by construction.
    """

    def __init__(self, n_verts, rows, cols, data, matrix_type):
        from cholespy import MatrixType

        if matrix_type != MatrixType.COO:
            raise RuntimeError("AMDConjugateGradientSolver expects COO input")

        self.n_verts = n_verts
        self.rows = mi.UInt(rows.array)
        self.cols = mi.UInt(cols.array)
        self.data = mi.Float(data.array)
        diag = dr.zeros(mi.Float, n_verts)
        diag_entries = dr.select(self.rows == self.cols, self.data, 0.0)
        dr.scatter_reduce(dr.ReduceOp.Add, diag, diag_entries, self.rows)
        self.inv_diag = dr.rcp(diag)
        self.max_iterations = 6

    def matvec(self, x):
        prod = dr.gather(mi.Point3f, x, self.cols) * self.data
        y = dr.zeros(mi.Point3f, self.n_verts)
        dr.scatter_reduce(dr.ReduceOp.Add, y, prod, self.rows)
        return y

    def solve(self, b, x=None):
        b_v = dr.unravel(mi.Point3f, b.array)
        x_v = dr.zeros(mi.Point3f, self.n_verts)
        r = b_v
        z = r * self.inv_diag
        p = z
        rz_old = dr.sum(dr.dot(r, z))

        for _ in range(self.max_iterations):
            ap = self.matvec(p)
            denom = dr.sum(dr.dot(p, ap))
            alpha = rz_old / dr.maximum(denom, 1e-20)
            x_v = x_v + alpha * p
            r = r - alpha * ap

            z = r * self.inv_diag
            rz_new = dr.sum(dr.dot(r, z))
            p = z + (rz_new / dr.maximum(rz_old, 1e-20)) * p
            rz_old = rz_new

        return mi.TensorXf(dr.ravel(x_v), shape=b.shape)


class LargeSteps():
    """
    Implementation of the algorithm described in the paper "Large Steps in
    Inverse Rendering of Geometry" (Nicolet et al. 2021).

    It consists in computing a latent variable u = (I + λL) v from the vertex
    positions v, where L is the (combinatorial) Laplacian matrix of the input
    mesh. Optimizing these variables instead of the vertex positions allows to
    diffuse gradients on the surface, which helps fight their sparsity.

    This class builds the system matrix (I + λL) for a given mesh and hyper
    parameter λ, and computes its Cholesky factorization.

    It can then convert vertex coordinates back and forth between their
    cartesian and differential representations. Both transformations are
    differentiable, meshes can therefore be optimized by using the differential
    form as a latent variable.
    """
    def __init__(self, verts, faces, lambda_=19.0):
        """
        Build the system matrix and its Cholesky factorization.

        Parameter ``verts`` (``mitsuba.Float``):
            Vertex coordinates of the mesh.

        Parameter ``faces`` (``mitsuba.UInt``):
            Face indices of the mesh.

        Parameter ``lambda_`` (``float``):
            The hyper parameter λ. This controls how much gradients are diffused
            on the surface. this value should increase with the tesselation of
            the mesh.

        """
        if mi.variant().endswith('double'):
            from cholespy import CholeskySolverD as CholeskySolver
        else:
            from cholespy import CholeskySolverF as CholeskySolver

        from cholespy import MatrixType
        import numpy as np

        v = verts.numpy().reshape((-1,3))
        f = faces.numpy().reshape((-1,3))

        # Remove duplicates due to e.g. UV seams or face normals.
        # This is necessary to avoid seams opening up during optimisation
        v_unique, index_v, inverse_v = np.unique(v, return_index=True, return_inverse=True, axis=0)
        inverse_v = inverse_v.flatten()
        f_unique = inverse_v[f]

        self.index = mi.UInt(index_v)
        self.inverse = mi.UInt(inverse_v)
        self.n_verts = v_unique.shape[0]

        # Solver expects matrices without duplicate entries as input, so we need to sum them manually
        indices, values = mesh_laplacian(self.n_verts, f_unique, lambda_)
        indices_unique, inverse_idx = np.unique(indices, axis=1, return_inverse=True)
        inverse_idx = inverse_idx.flatten()

        self.rows = mi.TensorXi(indices_unique[0])
        self.cols = mi.TensorXi(indices_unique[1])
        data = dr.zeros(mi.TensorXd, shape=(indices_unique.shape[1],))

        dr.scatter_reduce(dr.ReduceOp.Add, data.array, mi.Float64(values), mi.UInt(inverse_idx))

        if mi.variant().startswith('amd_'):
            self.solver = AMDConjugateGradientSolver(
                self.n_verts, self.rows, self.cols, data, MatrixType.COO)
        else:
            self.solver = CholeskySolver(
                self.n_verts, self.rows, self.cols, data, MatrixType.COO)
        self.data = mi.TensorXf(data)

    def to_differential(self, v):
        """
        Convert vertex coordinates to their differential form: u = (I + λL) v.

        This method typically only needs to be called once per mesh, to obtain
        the latent variable before optimization.

        Parameter ``v`` (``mitsuba.Float``):
            Vertex coordinates of the mesh.

        Returns ``mitsuba.Float`:
            Differential form of v.
        """
        # Manual matrix-vector multiplication
        v_unique = dr.gather(mi.Point3f, dr.unravel(mi.Point3f, mi.Float(v)), self.index)
        row_prod = dr.gather(mi.Point3f, v_unique, self.cols.array) * self.data.array
        u = dr.zeros(mi.Point3f, dr.width(v_unique))
        dr.scatter_reduce(dr.ReduceOp.Add, u, row_prod, self.rows.array)

        return dr.ravel(u)

    def from_differential(self, u):
        """
        Convert differential coordinates back to their cartesian form: v = (I +
        λL)⁻¹ u.

        This is done by solving the linear system (I + λL) v = u using the
        previously computed Cholesky factorization.

        This method is typically called at each iteration of the optimization,
        to update the mesh coordinates before rendering.

        Parameter ``u`` (``mitsuba.Float``):
            Differential form of v.

        Returns ``mitsuba.Float`:
            Vertex coordinates of the mesh.
        """
        v_unique = dr.unravel(mi.Point3f, dr.custom(SolveCholesky, self.solver, mi.TensorXf(u, (self.n_verts, 3))).array)
        return dr.ravel(dr.gather(mi.Point3f, v_unique, self.inverse))
