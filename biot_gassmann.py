import numpy as np
'''
==================================
Function Index:
==================================
gassmann(Kd,Gd,Ks,Kf,phi) 
    -> Ksat,Gsat

biot_gassmann_wood(Kd,Gd,Ks,Kf,phi,fraction) 
    -> H_lower

biot_gassmann_hill(Kd,Gd,Ks,Kf,phi,fraction)
    -> H_upper

biot_global_flow(Ks,Gs,rhos,Kd,Gd,phi,Kf,rhof,vis,perm0,tau,f)
    -> Vp1,Vp2,Vs,inv_Qp1,inv_Qp2,inv_Qs
'''
def gassmann(Kd,Gd,Ks,Kf,phi):
    '''
    Author: xuesong hu
    Date: 2026-7-29
    Description: calculate the elastic moduli of a fluid-saturated porous rock using Gassmann's equation

    Parameters
    Kd: bulk modulus of the dry rock skeleton [Pa]
    Gd: shear modeulus of the dry rock skeleton [Pa]
    Ks: bulk modulus of the solid grain [Pa]
    Kf: bulk modulus of fluid [Pa]
    phi: porosity [-]

    Returns
    Ksat: bulk modulus of the fluid saturated rock [Pa]
    Gsat: shear modulus of the fluid saturated rock [Pa]
    '''
    alpha = 1 - Kd/Ks
    M = ((alpha - phi)/Ks + phi/Kf)**(-1)
    Ksat = Kd + alpha**2 * M
    Gsat = Gd
    return Ksat,Gsat

def biot_gassmann_wood(Kd,Gd,Ks,Kf,phi,fraction):
    '''
    Author: xuesong hu
    Date: 2026-7-29
    Description:
    Calculate low-frequency lower bound of the P-wave modulus for a porous rock partially saturated with multiple fluids.
    Rocks saturated with different fluids can be regarded as a springs series model.
    At high frequency limits, There is enough time for pore pressure diffusion.
    
    Parameters
    Kd: bulk modulus of the dry rock skeleton [Pa]
    Gd: shear modeulus of the dry rock skeleton [Pa]
    Ks: bulk modulus of the solid grain [Pa]
    Kf: bulk modulus of the solid fluid [Pa]
    phi: porosity [-]
    fraction: saturation fractions of the individual fluids, the sum of all fractions should be equal to 1

    Returns
    H_lower: low frequency lower bound of the saturated P-wave modulus

    '''
    if np.isclose(np.sum(fraction), 1.0):
        wood_sum = 0.0
        for i in range(len(fraction)):
            wood_sum = wood_sum + fraction[i]/Kf[i]
        Kf_low = 1/wood_sum
        H_lower = gassmann(Kd,Gd,Ks,Kf_low,phi)[0] + 4/3*Gd
        return H_lower
    else:
        print("The sum of all components is not equal to 1 !!!")
        return 0

def biot_gassmann_hill(Kd,Gd,Ks,Kf,phi,fraction):
    '''
    Author: xuesong hu
    Date: 2026-7-29
    Description:
    Calculate high-frequency upper bound of the P-wave modulus for a porous rock partially saturated with multiple fluids.
    Rocks saturated with different fluids are regarded as a springs series model.
    At high frequency limits, There is not enough time for pore pressure diffusion.

    Parameters
    Kd: bulk modulus of the dry rock skeleton [Pa]
    Gd: shear modeulus of the dry rock skeleton [Pa]
    Ks: bulk modulus of the solid grain [Pa]
    Kf: bulk modulus of the solid fluid [Pa]
    phi: porosity
    fraction: saturation fractions of the individual fluids, the sum of all fractions should be equal to 1
    
    Returns
    H_upper: high frequency upper bound of the saturated P-wave modulus
    '''
    if np.isclose(np.sum(fraction), 1.0):
        Ksat = np.zeros(len(fraction))
        hill_sum = 0.0
        for i in range(len(fraction)):
            Ksat[i] = gassmann(Kd,Gd,Ks,Kf[i],phi)[0]
            hill_sum = hill_sum + fraction[i] / (Ksat[i] + 4/3*Gd)      
        H_upper = 1/hill_sum
        return H_upper
    else:
        print("The sum of all components is not equal to 1 !!!")
        return 0

def biot_global_flow(Ks,Gs,rhos,Kd,Gd,phi,Kf,rhof,vis,perm0,tau,f):
    '''
    Author: xuesong hu
    Date: 2026-7-29
    Description:
    Calculate frequency-dependent wave velocities and attenuation ...
    in a fluid-saturated porous medium based on Biot's global-flow theory.
    The analytical solution can be derived by substituting the plane-wave solution into the dynamic Biot equations.

    Parameters
    Ks: bulk modulus of the solid grain [Pa]
    Gs: shear modulus of the solid grain [Pa]
    rhos: density of the solid grain [kg/m**3]
    Kd: bulk modulus of the dry rock skeleton [Pa]
    Gd: shear modulus of the dry rock skeleton [Pa]
    phi: porosity [-]
    Kf: bulk modulus of the pore fluid [Pa]
    rhof: density of the pore fluid [kg/m**3]
    vis: viscosity of the pore fluid [Pa]
    perm0: static permeability [m^2]
    tau: tortuosity [-]
    f: frequency [Hz]

    Returns
    Vp1: Fast P wave velocity [m/s]
    Vp2: Slow P wave velocity [m/s]
    Vs: S wave velocity [m/s]
    inv_Qp1: inverse quality factor of fast P wave [-]
    inv_Qp2: inverse quality factor of slow P wave [-]
    inv_Qs: inverse quality factor of S wave [-]

    Reference
    Biot - 1962 - Mechanics of Deformation and Acoustic Propagation in porous media

    '''
    rho    = phi*rhof + (1-phi)*rhos
    omega  = 2*np.pi*f
    alpha  = 1 - Kd/Ks
    M      = ((alpha - phi)/Ks + phi/Kf)**(-1)
    H      = Kd + alpha**2 * M + 4/3*Gd
    omegaB = phi*vis/(perm0*tau*rhof)
    perm   = perm0*(np.sqrt(1-1j*omega/(2*omegaB))-1j*omega/omegaB)**(-1)
    rho_hat=1j*vis/(omega*perm)
    # Fast P wave and slow P wave
    A     = (rho*M + rho_hat*H - 2*rhof*alpha*M) / (2*(M*H - (alpha*M)**2))
    B     = (rho*rho_hat - rhof**2) / (M*H - (alpha*M)**2)
    spf_2 = A - np.sqrt(A**2-B)
    spf   = np.sqrt(spf_2)
    sps_2 = A + np.sqrt(A**2-B)
    sps   = np.sqrt(sps_2)
    # Slow P wave
    ss    = np.sqrt((rho*rho_hat - rhof**2) / (rho_hat*Gd))
    # Wavenumber of all kinds of waves
    kp1 = omega*spf
    kp2 = omega*sps
    ks  = omega*ss
    # Velocity and inverse quality factors
    Vp1    = omega/np.real(kp1) 
    Vp2    = omega/np.real(kp2)
    Vs     = omega/np.real(ks)
    inv_Qp1 = 2*np.imag(kp1)/np.real(kp1)
    inv_Qp2 = 2*np.imag(kp2)/np.real(kp2)
    inv_Qs  = 2*np.imag(ks)/np.real(ks)

    return Vp1,Vp2,Vs,inv_Qp1,inv_Qp2,inv_Qs