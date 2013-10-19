import numpy as np

# Going to test just converting a simple Cartesian mesh to a cylindrical one.
# coordinates: (x,y,z) --> (r,phi,z)
# conversion: r = sqrt(x^2+y^2+z^2); phi = arctan(y/x); z = z

def cart_to_cyl(in_tuple):
    x, y, z = in_tuple
    r = np.sqrt(x*x + y*y + z*z)
    if x != 0.:
        phi = np.arctan2(y/x)
    else:
        phi = 0.
    return r, phi, z

def make_cart_mesh(dims):
    # Assuming a cubic mesh, so dims is an edge measure.
    dx = 1.
    v = int(dims)
    return tuple([(dx*x,dx*y,dx*z) for x in xrange(v) for y in xrange(v) for z in xrange(v)])

def run_it():
    mesh = make_cart_mesh(5)  # This will make a 125-element tuple.
    new_mesh = tuple([cart_to_cyl(t) for t in mesh])
    #print new_mesh

def gradient(x):
    # The functional definition of a derivative is:
    # \lim_{h\rightarrow 0} \frac{f(x+h)-f(x)}{h}.

def strain_equation():
    # For now, this is a placeholder, with only a TeX declaration for the strain equation
    # + for strain that is uniform along the z-direction ("c" axis, in SS terms).
    
    # c11*D_{xx}u_x + c44*D_{yy}u_x + (c12 + c44)*D_{xy}u_y + c13*D_{xz}u_z = 0
    # c44*D_{xx}u_y + c11*D_{yy}u_y + (c12 + c44)*D_{xy}u_x + c13*D_{yz}u_z = 0
    # D_{xx}u_z + D_{yy}u_z = 0

    # The average elastic density for a YBCO/BZO cylindrical system is
    # E = \frac{1}{\piD^2}\left[\piR^2E_{BZO} + \pi(D^2 - R^2)E_{YBCO}\right]
    #   = 0.5*(c11^1+c12^1)*[c33^1*(c11^1+c12^1) - 2*c13^1*c13^1]*\frac{2(1-\rho)+\rho*log(\rho)*(2-log(\rho))}{(log(\rho))^2}*A_0^2

if __name__=='__main__':
    run_it()
