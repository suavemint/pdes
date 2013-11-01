import dolfin as d
import numpy as np

def cart_to_cyl(in_tuple):
    x, y, z = in_tuple
    r = np.sqrt(x*x + y*y + z*z)
    if x != 0.:
        phi = np.arctan2(y/x)
    else:
        phi = 0.
    return r, phi, z

#set_log_level(PROGRESS)
# The first test of the package and my abilities to work with PDE solvers
# + will be with the Poisson solver:

# Using Fenics:
# 1. Identify the PDE and its BCs
# 2. Reformulate the PDE problem as a variational one
# 3. Make script where formulas in the variational problem
# +  are coded, along with input data such as f, u0, and a mesh for \Omega.
# 4. Add statements in the script for solving the variational problem,
# +  computing derived quantities such as grad(u), or visualizing.

#print 'Running simple test of Poisson-solving...'
#
#mesh = d.UnitSquareMesh(32, 32)
#V = d.FunctionSpace(mesh, "Lagrange", 1)
#u = d.TrialFunction(V)
#v = d.TestFunction(V)
#
#f = d.Expression('x[0]*x[1]')
#
#dx = d.dx
#a = d.dot(d.grad(u), d.grad(v))*dx
#L = f*v*dx
#
#bc = d.DirichletBC(V, 0.0, d.DomainBoundary())
#
#u = d.Function(V)
#
#d.solve(a == L, u, bc)
#print u.vector()
#d.plot(u);d.interactive()

#########
## Next will be a little linear elastic problem:
#########

# Make a mesh for the 8x8x8 cubic-space elastic problem:

mesh = d.UnitCubeMesh(8,8,8)
#mesh = d.Cylinder(d.Point(0.,0.,0.),d.Point(0.,5.,0.),2.,6)
#
## Next make a function space
V = d.VectorFunctionSpace(mesh, 'Lagrange', 1)

# Create a test function and a trial function, and a source term:
u = d.TrialFunction(V)
w = d.TestFunction(V)
b = d.Constant((1.0,0.,0.))

# Elasticity parameters:
E, nu = 10., 0.3
mu, lambda_param = E / (2. * (1.+nu)), E * nu / ((1. + nu) * (1.-2. * nu))

# Stress tensor:
# + usually of form \sigma_{ij} = \lambda e_{kk}\delta_{ij} + 2\mu e_{ij},
# + or                          = \lambda Tr(e_{ij})I + 2\mu e_{ij}, for e_{ij} the strain tensor.
sigma = lambda_param*d.tr(d.grad(u)) * d.Identity(w.cell().d) + 2 * mu * d.sym(d.grad(u))

# Governing balance equation:
F = d.inner(sigma, d.grad(w)) * d.dx - d.dot(b,w)*d.dx

# Extract the bi- and linear forms from F:
a = d.lhs(F)
L = d.rhs(F)

# Dirichlet BC on entire boundary:
c = d.Constant((0.,0.,0.))
bc = d.DirichletBC(V, c, d.DomainBoundary())

## Testing some new boundary definitions:
def bzo_boundary(r_vec, on_boundary):

    # NB: input r_vec is a position vector
    r,theta,z = cart_to_cyl(x)  # NB: check the function cart_to_cyl can take non-tuple collecs.
    return d.near(r,bzo_radius)

## Testing some differing-boundary BCs:
bzo = d.DirichletBC(V, u_bzo, bzo_boundary)
ybco = d.DirichletBC(V, u_ybco, ybco_boundary)
##

# Finally, set up the PDE and solve it:
u = d.Function(V)
problem = d.LinearVariationalProblem(a, L, u, bcs=bc)
solver = d.LinearVariationalSolver(problem)
solver.parameters['symmetric'] = True
s = solver.solve()
#print u.vector()
d.plot(u);d.interactive()

#######
## Now we'll try a more low-level (mathematically) approach
#######

#element = d.FiniteElement('Vector Lagrange', 'tetrahedron', 1)
#
#v = d.BasisFunction(element)
#U = d.BasisFunction(element)
#f = d.Function(element)
#
#E, nu = 10., .3
#
#mu = E / (2 * (1 + nu))
#lambda_const = E * nu / ((1 + nu) * (1 - 2 * nu))
#
#def epsilon(v):
#    return 0.5 * (d.grad(v) + d.transp(d.grad(v)))
#
#def sigma(v):
#    return 2 * mu * epsilon(v) + lambda_const * d.mult(d.trace(epsilon(v)), d.Identity(len(v)))
#
#a = d.dot(d.grad(v), sigma(U)) * d.dx
#L = d.dot(v, f) * d.dx
