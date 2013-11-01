import numpy as np

# ASSUMPTION: Inputted elastic constants for BZO and YBCO are assumed to be in dict format,
# + keyed by {...,(i,j): #.###,...} !
def elastic_energy(c1, c2, R, D):
    """
    Calcuates the elastic energy, for cylindrical system of BZO NRs in a YBCO film;
    NB: Need to move many of the calculations for constants in this function to a
    more universal namespace; for every rho at which the total elastic energy
    density is calculated requires recomputation of constants.
    """

    import numpy as np

    rho = R*R / D*D  # Volume density of BZO. ** For further calculations, assume rho << 1. **

    lambda_1 = (c_1[(1,1)] + c_1[(1,2)]) / c_1[(1,3)]
    lambda_2 = (c_2[(1,1)] + c_2[(1,2)]) / c_2[(1,3)]

    # TODO: Does f_3 == f_z?? #####
    f_3 = (C2 - C1) / C1
    f_theta = np.sqrt(2)*f_x

    a_0 = (f_3(1+f_theta)+lambda_2*f_theta*(1+f3))/(c_1[(1,3)]*(lambda_1*(1+f_theta) - lambda_2*(1+f_3)))
    # b_0 =   # DNE

    a_1 = a_0 / (np.ln(D) - np.ln(R))
    b_1 = ( f3 + lambda_1*f_theta ) / (c_2[(1,3)] * ( lambda_1*(1+f_theta)-lambda_2*(1+f_3) ))


    v_1 = 0.5 * ( c1[(1,1)] + c1[(1,2)] ) - ( 1/c1[(3,3)] ) * ( c1[(1,3)] * c1[(1,3)] )
    v_2 = 0.5 * ( c2[(1,2)] + c2[(1,2)] ) - ( 1/c2[(3,3)] ) * ( c2[(1,3)] * c2[(1,3)] )

    E = (c1[(1,1)] + c1[(1,2)])*c[(3,3)]*v_1*((2*(1-rho)+rho*np.ln(rho*(2-np.ln(rho))))/(np.ln(rho)*np.ln(rho)))*a_0*a_0 + (c_2[(1,1)]+c_2[(1,2)])*c_2[(3,3)]*v_2*b_1*b_1*rho

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

def solve():
    # First, let's define some constants to use in the equations:
    a1 = e1/(c11 + c12)  # TODO: Need to add in an ability to choose materials' constants here.
    # In eq. above, what is e1?

    b1 = e2/(c11 + c12)
    b2 = c/(c11 + c12)

if __name__=='__main__':
    run_it()
