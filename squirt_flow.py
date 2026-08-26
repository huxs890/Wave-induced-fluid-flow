import numpy as np
from scipy.special import jv
'''
Squirt flow in penny shaped crack
==================================
Function Index:
==================================
squirt_flow_undrained(Kf,f)
    -> Kf_star

squirt_flow_drained(Kf,vis,h0,rb,f)
    -> Kf_star
'''
def squirt_flow_undrained(Kf,f):
    '''
    Author: xuesong hu
    Date: 2026-8-3
    Description: calculate the effective fluid bulk modulus for the undrained squirt-flow boundary condition

    Parameters
    Kf: Bulk modulus of the pore fluid [Pa]
    f: Frequency [Hz]

    Returns
    Kf_star: Frequency-dependent effective fluid bulk modulus [Pa]

    Reference
    Murphy - 1986- Acoustic relaxation in sedimentary rocks: Dependence on grain contacts and fluid saturation
    Gurevich et al. - 2010 - A simple model for squirt-flow dispersion and attenuation in fluid-saturated granular rocks
    '''
    Kf_star = np.linspace(Kf,Kf,len(f))
    return Kf_star

def squirt_flow_drained(Kf,vis,h0,rb,f):
    '''
    Author: xuesong hu
    Date: 2026-8-3
    Description: calculate the effective fluid bulk modulus for the drained squirt-flow boundary condition
                 The cracks are modeled as penny-shaped, and the linearized Navier-Stokes
                 equations are solved in cylindrical coordinates under harmonic oscillation

    Parameters
    Kf: bulk modulus of the pore fluid [Pa]
    vis: viscosity [Pa·s]
    h0: initial crack aperture [m]
    rb: radius of the penny-shaped crack [m]
    f: frequency [Hz]

    Returns
    Kf_star: complex frequency-dependent effective fluid bulk modulus [Pa]

    Reference
    Murphy - 1986- Acoustic relaxation in sedimentary rocks: Dependence on grain contacts and fluid saturation
    Gurevich et al. - 2010 - A simple model for squirt-flow dispersion and attenuation in fluid-saturated granular rocks
    '''
    omega   = 2*np.pi*f
    k       = np.sqrt(-12j*vis*omega/(h0**2 *Kf))
    Kf_star = (1-2*jv(1,k*rb)/(k*rb*jv(0,k*rb)))*Kf
    return Kf_star