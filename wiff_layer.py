import numpy as np
from utils import coth
'''
==================================
Function Index:
==================================
white_layer_1975(rock_up,rock_low,fluid_up,fluid_low,perm,d1,d2,f)
    -> H_white,Vp_white,invQp_white

biot_layer_matrix(rock_up,rock_low,fluid_up,fluid_low,perm,d1,d2,f,Pc,undrained,Pf_out)
    -> H,h,pf

dynamic_biot_layer_matrix(rock_up,rock_low,fluid_up,fluid_low,perm,d1,d2,f,s,Pc,undrained,Pf_out)
    -> H,h,pf
'''
def white_layer_1975(rock_up,rock_low,fluid_up,fluid_low,perm,d1,d2,f):
    '''
    Author: xuesong hu
    Date: 2026-8-4    
    Description: This function implements the White (1975) layered partial-saturation model ...
                 to calculate the frequency-dependent P-wave modulus, velocity, ...
                 and attenuation of a two-layer fluid-saturated porous medium.
                 this is an analytical solution of 1D quasi-static Biot's equation with undrained boundary.

    Parameters
    rock_up   = [Ks_up,Gs_up,rhos_up,Kd_up,Gd_up,phi_up]
    fluid_up  = [Kf_up,rhof_up,vis_up]
    rock_low  = [Ks_low,Gs_low,rhos_low,Kd_low,Gd_low,phi_low]
    fluid_low = [Kf_low,rhof_low,vis_low]
    perm      = [perm_up,perm_low] [m^2]
    d1: thickness of upper layer [m]
    d2: thickness of lower layer [m]
    f:  frequency [Hz]

    Returns
    H_white: P wave modulus of layered saturated rock with immiscible fluid layers [Pa]
    Vp_white: P wave velocity [m/s]
    invQp_white: inverse quality factor [-]
    '''
    L=d1+d2
    omega = 2*np.pi*f
    # Upper layer
    Ks_up   = rock_up[0]
    Gs_up   = rock_up[1] 
    rhos_up = rock_up[2]
    Kd_up   = rock_up[3]
    Gd_up   = rock_up[4]
    phi_up  = rock_up[5]
    Kf_up   = fluid_up[0]
    rhof_up = fluid_up[1]
    vis_up  = fluid_up[2]
    perm_up = perm[0]

    alpha_up = 1 - Kd_up/Ks_up
    M_up     = ((alpha_up-phi_up)/Ks_up+phi_up/Kf_up)**(-1)
    Kg_up    = Kd_up + alpha_up**2 * M_up
    Eg_up    = Kg_up + (4/3)*Gd_up
    r_up     = alpha_up*M_up/Eg_up
    Ed_up    = Kd_up + (4/3)*Gd_up
    KE_up    = Ed_up*M_up/Eg_up
    ke_up    = np.sqrt(1j*omega*vis_up/(perm_up*KE_up))
    I_up     = vis_up/(perm_up*ke_up)*coth(ke_up*d1/2)

    # Lower Layer
    Ks_low   = rock_low[0]
    Gs_low   = rock_low[1] 
    rhos_low = rock_low[2]
    Kd_low   = rock_low[3]
    Gd_low   = rock_low[4]
    phi_low  = rock_low[5]
    Kf_low   = fluid_low[0]
    rhof_low = fluid_low[1]
    vis_low  = fluid_low[2]
    perm_low = perm[1]

    alpha_low = 1 - Kd_low/Ks_low
    M_low     = ((alpha_low-phi_low)/Ks_low+phi_low/Kf_low)**(-1)
    Kg_low    = Kd_low + alpha_low**2 * M_low
    Eg_low    = Kg_low + (4/3)*Gd_low
    r_low     = alpha_low*M_low/Eg_low
    Ed_low    = Kd_low + (4/3)*Gd_low
    KE_low    = Ed_low*M_low/Eg_low
    ke_low    = np.sqrt(1j*omega*vis_low/(perm_low*KE_low))
    I_low     = vis_low/(perm_low*ke_low)*coth(ke_low*d2/2)

    # Main routions
    v1 = d1/L
    v2 = d2/L
    E0 = (v1/Eg_up+v2/Eg_low)**(-1)
    T  = (2*(r_low-r_up)**2)/(1j*omega*(d1+d2)*(I_up+I_low))
    H_white  = (1/E0+T)**(-1)
    rho_up   = rhof_up*phi_up + rhos_up*(1-phi_up)
    rho_low  = rhof_low*phi_low + rhos_low*(1-phi_low)
    rho      = rho_up*v1 + rho_low*v2
    Vp_white    = np.sqrt(np.real(H_white)/rho)
    invQp_white = np.imag(H_white)/np.real(H_white)

    return H_white,Vp_white,invQp_white

def biot_layer_matrix(rock_up,rock_low,fluid_up,fluid_low,perm,d1,d2,f,Pc,undrained,Pf_out):
    '''
    Author: xuesong hu
    Date: 2026-8-4    
    Description: solve the 1D quasi-static Biot equations for a two-layer porous medium using a matrix method.
                 both undrained and drained outer boundary conditions are considered.
                 adopt continuity conditions at the layer interface.

    Parameters:
    rock_up   = [Ks_up,Gs_up,rhos_up,Kd_up,Gd_up,phi_up]
    fluid_up  = [Kf_up,rhof_up,vis_up]
    rock_low  = [Ks_low,Gs_low,rhos_low,Kd_low,Gd_low,phi_low]
    fluid_low = [Kf_low,rhof_low,vis_low]
    perm      = [perm_up,perm_low] [m^2]
    d1: thickness of upper layer [m]
    d2: thickness of lower layer [m]
    f:  frequency [Hz]
    Pc: harmonic oscillation stress amplitude on upper and lower surface [Pa]
    undrained: boundary-condition flag, (undrained = 1): undrained outer boundary; (undrained = 0): drained outer boundaries;
    Pf_out: prescribed fluid pressure at the drained outer boundaries [Pa], only useful when undrained = 0

    Returns
    H: complex frequency-dependent effective P-wave modulus [Pa]
    h: spatial coordinates used for the pore-pressure profile [m]
    pf: complex fluid pressure distribution for the last input frequency [Pa]
    
    Reference
    Biot - 1941 - General Theory of Three-Dimensional Consolidation
    '''
    f = np.atleast_1d(np.asarray(f, dtype=float))
    a = d1/2
    b = d2/2
    # Upper layer
    Ks_up   = rock_up[0]
    Gs_up   = rock_up[1] 
    rhos_up = rock_up[2]
    Kd_up   = rock_up[3]
    Gd_up   = rock_up[4]
    phi_up  = rock_up[5]
    Kf_up   = fluid_up[0]
    rhof_up = fluid_up[1]
    vis_up  = fluid_up[2]
    perm_up = perm[0]

    alpha_up = 1 - Kd_up/Ks_up
    M_up     = ((alpha_up-phi_up)/Ks_up+phi_up/Kf_up)**(-1)
    lamda_up = Kd_up + alpha_up**2 *M_up - 2/3*Gd_up
    H_up     = lamda_up + 2*Gd_up
    E_up     = Kd_up + 4/3*Gd_up
    D_up     = perm_up/vis_up*E_up*M_up/H_up

    # Lower Layer
    Ks_low   = rock_low[0]
    Gs_low   = rock_low[1] 
    rhos_low = rock_low[2]
    Kd_low   = rock_low[3]
    Gd_low   = rock_low[4]
    phi_low  = rock_low[5]
    Kf_low   = fluid_low[0]
    rhof_low = fluid_low[1]
    vis_low  = fluid_low[2]
    perm_low = perm[1]

    alpha_low = 1 - Kd_low/Ks_low
    M_low     = ((alpha_low-phi_low)/Ks_low+phi_low/Kf_low)**(-1)
    lamda_low = Kd_low + alpha_low**2 *M_low - 2/3*Gd_low
    H_low     = lamda_low + 2*Gd_low
    E_low     = Kd_low + 4/3*Gd_low
    D_low     = perm_low/vis_low*E_low*M_low/H_low

    # main routines
    H = np.zeros(len(f),dtype=np.complex128)
    for i in range(len(f)):
        omega = 2*np.pi*f[i]
        k_up  = np.sqrt(1j*omega/D_up)
        k_low = np.sqrt(1j*omega/D_low)
        N_up  = E_up*M_up*k_up/H_up
        N_low = E_low*M_low*k_low/H_low
        beta_up  = alpha_up*M_up/H_up
        beta_low = alpha_low*M_low/H_low

        # assemble the matrix
        A = np.zeros((4,4),dtype=np.complex128)
        # w_up(0)=w_low(0)
        A[1,0] = 1
        A[1,1] = 1
        A[1,2] = -1
        A[1,3] = -1
        # p_up(0)=p_low(0)
        A[2,0] = N_up
        A[2,1] = -N_up
        A[2,2] = -N_low
        A[2,3] = N_low
        # undrained == 1 : undrained; undrained == 0 : drained
        if undrained == 1:
            # w_up(-a) = 0
            z = -a
            A[0,0] = np.exp(k_up*z)
            A[0,1] = np.exp(-k_up*z)
            A[0,2] = 0
            A[0,3] = 0
            # w_low(b) = 0
            z = b
            A[3,0] = 0
            A[3,1] = 0
            A[3,2] = np.exp(k_low*z)
            A[3,3] = np.exp(-k_low*z)
            Y = np.array([0,0,(beta_up-beta_low)*Pc,0],dtype=np.complex128)
        elif undrained == 0:
            # p_up(-a) = 0
            z = -a
            A[0,0] = N_up*np.exp(k_up*z)
            A[0,1] = -N_up*np.exp(-k_up*z)
            A[0,2] = 0
            A[0,3] = 0
            # p_low(b) = 0
            z = b
            A[3,0] = 0
            A[3,1] = 0
            A[3,2] = N_low*np.exp(k_low*z)
            A[3,3] = -N_low*np.exp(-k_low*z) 
            Y = np.array([beta_up*Pc-Pf_out,0,(beta_up-beta_low)*Pc,beta_low*Pc-Pf_out],dtype=np.complex128)
        else:
            raise ValueError("undrained must be 0 or 1")

        # solve AX=Y
        if not np.all(np.isfinite(A)):
            raise FloatingPointError(f"A contains NaN/Inf at f={f[i]} Hz")
        row_scale = np.max(np.abs(A), axis=1)
        if np.any(row_scale == 0):
            raise np.linalg.LinAlgError(f"A contains zero row at f={f[i]} Hz")
        A_scaled = A / row_scale[:, None]
        Y_scaled = Y / row_scale
        cond_scaled = np.linalg.cond(A_scaled)
        if cond_scaled > 1e10:
            print(f"Warning: A is ill-conditioned "f"at f={f[i]:.4e} Hz, "f"cond={cond_scaled:.3e}")
        X = np.linalg.solve(A_scaled,Y_scaled)
        # X = np.linalg.solve(A,Y)
        A1 = X[0]
        B1 = X[1]
        A2 = X[2]
        B2 = X[3]

        # u(b)-u(-a)
        z = -a
        u1      = -Pc/H_up*z - alpha_up*M_up/H_up*(A1*np.exp(k_up*z)+B1*np.exp(-k_up*z))
        z = b
        u2      = -Pc/H_low*z - alpha_low*M_low/H_low*(A2*np.exp(k_low*z)+B2*np.exp(-k_low*z))
        delta_C = beta_low*(A2+B2) - beta_up*(A1+B1) # C2-C1 
        du      = u2 - u1 + delta_C
        H[i]    = -(a+b)/du*Pc

    # fluid pressure,the last frequency
    # upper layer
    z_up = np.linspace(-a,-a/100,100)
    pf_up = alpha_up*M_up/H_up*Pc-E_up*M_up/H_up*(A1*k_up*np.exp(k_up*z_up)-B1*k_up*np.exp(-k_up*z_up))

    # lower layer
    z_low = np.linspace(0,b,100)
    pf_low = alpha_low*M_low/H_low*Pc-E_low*M_low/H_low*(A2*k_low*np.exp(k_low*z_low)-B2*k_low*np.exp(-k_low*z_low))
    
    h  = np.append(z_up,z_low)
    pf = np.append(pf_up,pf_low)   

    return H,h,pf 

def dynamic_biot_layer_matrix(rock_up,rock_low,fluid_up,fluid_low,perm,d1,d2,f,s,Pc,undrained,Pf_out):
    '''
        Author: xuesong hu
        Date: 2026-8-4    
        Description: solve the 1D quasi-static Biot equations for a two-layer porous medium using a matrix method.
                     both undrained and drained outer boundary conditions are considered.
                     implement continuity conditions at the layer interface.
    
        Parameters:
        rock_up   = [Ks_up,Gs_up,rhos_up,Kd_up,Gd_up,phi_up]
        fluid_up  = [Kf_up,rhof_up,vis_up]
        rock_low  = [Ks_low,Gs_low,rhos_low,Kd_low,Gd_low,phi_low]
        fluid_low = [Kf_low,rhof_low,vis_low]
        perm      = [perm_up,perm_low] [m^2]
        d1: thickness of upper layer [m]
        d2: thickness of lower layer [m]
        f:  frequency [Hz]
        Pc: harmonic oscillation stress amplitude on upper and lower surface [Pa]
        undrained: boundary-condition flag, (undrained = 1): undrained outer boundary; (undrained = 0): drained outer boundaries;
                   the program for open BC has not yet been written
        Pf_out: prescribed fluid pressure at the drained outer boundaries [Pa], only useful when undrained = 0
    
        Returns
        H: complex frequency-dependent effective P-wave modulus [Pa]
        h: spatial coordinates used for the fluid-pressure profile [m]
        pf: complex fluid pressure distribution for the last input frequency [Pa]
        
        Reference
        Vogelaar - 2007 - Extension of White's layered model to the full frequency range
        '''
    f = np.atleast_1d(np.asarray(f, dtype=float))
    a = d1/2
    b = d2/2
    # Upper layer
    Ks_up   = rock_up[0]
    Gs_up   = rock_up[1] 
    rhos_up = rock_up[2]
    Kd_up   = rock_up[3]
    Gd_up   = rock_up[4]
    phi_up  = rock_up[5]
    Kf_up   = fluid_up[0]
    rhof_up = fluid_up[1]
    vis_up  = fluid_up[2]
    perm_up = perm[0]
    s_up    = s[0]
    
    alpha_up = 1 - Kd_up/Ks_up
    m_up     = s_up*rhof_up/phi_up
    M_up     = 1 / ((alpha_up-phi_up)/Ks_up+phi_up/Kf_up)
    K_up     = Kd_up + alpha_up**2 * M_up
    H_up     = K_up + (4/3)*Gd_up
    rho_up   = (1-phi_up)*rhos_up + phi_up*rhof_up 

    # Lower Layer
    Ks_low   = rock_low[0]
    Gs_low   = rock_low[1] 
    rhos_low = rock_low[2]
    Kd_low   = rock_low[3]
    Gd_low   = rock_low[4]
    phi_low  = rock_low[5]
    Kf_low   = fluid_low[0]
    rhof_low = fluid_low[1]
    vis_low  = fluid_low[2]
    perm_low = perm[1]
    s_low    = s[1]

    alpha_low = 1 - Kd_low/Ks_low
    m_low     = s_low*rhof_low/phi_low
    M_low     = 1 / ((alpha_low-phi_low)/Ks_low+phi_low/Kf_low)
    K_low     = Kd_low + alpha_low**2 * M_low
    H_low     = K_low + (4/3)*Gd_low
    rho_low   = (1-phi_low)*rhos_low + phi_low*rhof_low 

    # main routine
    Y = np.array([-Pc,0,0,0,0,0,-Pc,0],dtype=np.complex128) 
    H = np.zeros(len(f),dtype=np.complex128)
    for i in range(len(f)):
        omega = 2*np.pi*f[i]
        A = np.zeros((8,8),dtype=np.complex128)
        # upper layer
        temp_a   = rhof_up*H_up - rho_up*alpha_up*M_up
        temp_b   = H_up*m_up-rho_up*M_up-1j*vis_up*H_up/(perm_up*omega)
        temp_c   = m_up*alpha_up*M_up-rhof_up*M_up-1j*vis_up*alpha_up*M_up/(perm_up*omega)
        delta    = temp_b*temp_b-4*temp_a*temp_c
        sigma_c1 = (-temp_b+np.sqrt(delta))/(2*temp_a)
        sigma_d1 = (-temp_b-np.sqrt(delta))/(2*temp_a)
        temp1    = (rhof_up*(H_up*sigma_c1+alpha_up*M_up)-
                    rho_up*(alpha_up*M_up*sigma_c1+M_up))/((alpha_up*M_up)**2-
                                                           H_up*M_up)
        kc_up    = np.sqrt(temp1)*omega
        temp2    = (rhof_up*(H_up*sigma_d1+alpha_up*M_up)-
                    rho_up*(alpha_up*M_up*sigma_d1+M_up))/((alpha_up*M_up)**2-
                                                           H_up*M_up)
        kd_up    = np.sqrt(temp2)*omega
        # lower layer
        temp_a   = rhof_low*H_low - rho_low*alpha_low*M_low
        temp_b   = H_low*m_low-rho_low*M_low-1j*vis_low*H_low/(perm_low*omega)
        temp_c   = m_low*alpha_low*M_low-rhof_low*M_low-1j*vis_low*alpha_low*M_low/(perm_low*omega)
        delta    = temp_b*temp_b-4*temp_a*temp_c
        sigma_c2 = (-temp_b+np.sqrt(delta))/(2*temp_a)
        sigma_d2 = (-temp_b-np.sqrt(delta))/(2*temp_a)
        temp1    = (rhof_low*(H_low*sigma_c2+alpha_low*M_low)-
                    rho_low*(alpha_low*M_low*sigma_c2+M_low))/((alpha_low*M_low)**2-
                                                           H_low*M_low)
        kc_low   = np.sqrt(temp1)*omega
        temp2    = (rhof_low*(H_low*sigma_d2+alpha_low*M_low)-
                    rho_low*(alpha_low*M_low*sigma_d2+M_low))/((alpha_low*M_low)**2-
                                                           H_low*M_low)
        kd_low   = np.sqrt(temp2)*omega  

        # assemble matrix A
        z = -a
        # sigma_up(-a) = -Pc 
        A[0,0] = (-H_up*sigma_d1*kc_up*np.sin(kc_up*z)
                  -alpha_up*M_up*kc_up*np.sin(kc_up*z))
        A[0,1] = (H_up*sigma_d1*kc_up*np.cos(kc_up*z)
                  +alpha_up*M_up*kc_up*np.cos(kc_up*z))
        A[0,2] = (-H_up*sigma_c1*kd_up*np.sin(kd_up*z)
                  -alpha_up*M_up*kd_up*np.sin(kd_up*z))
        A[0,3] = (H_up*sigma_c1*kd_up*np.cos(kd_up*z)
                  +alpha_up*M_up*kd_up*np.cos(kd_up*z))
        A[0,4] = 0
        A[0,5] = 0
        A[0,6] = 0
        A[0,7] = 0
        # w_up(-a) = 0
        A[1,0] = np.cos(kc_up*z)
        A[1,1] = np.sin(kc_up*z)
        A[1,2] = np.cos(kd_up*z)
        A[1,3] = np.sin(kd_up*z)
        A[1,4] = 0
        A[1,5] = 0
        A[1,6] = 0
        A[1,7] = 0

        z = 0
        # w_up(0) = w_low(0) -> w_up-w_low=0 
        A[2,0] = np.cos(kc_up*z)
        A[2,1] = np.sin(kc_up*z)
        A[2,2] = np.cos(kd_up*z)
        A[2,3] = np.sin(kd_up*z)
        A[2,4] = -np.cos(kc_low*z)
        A[2,5] = -np.sin(kc_low*z)
        A[2,6] = -np.cos(kd_low*z)
        A[2,7] = -np.sin(kd_low*z)  
        # u_up(0) = u_low(0) -> u_up-u_low=0 
        A[3,0] = sigma_d1*np.cos(kc_up*z)
        A[3,1] = sigma_d1*np.sin(kc_up*z)
        A[3,2] = sigma_c1*np.cos(kd_up*z)
        A[3,3] = sigma_c1*np.sin(kd_up*z)
        A[3,4] = -sigma_d2*np.cos(kc_low*z)
        A[3,5] = -sigma_d2*np.sin(kc_low*z)
        A[3,6] = -sigma_c2*np.cos(kd_low*z)
        A[3,7] = -sigma_c2*np.sin(kd_low*z)
        # sigma_up(0) = sigma_low(0) -> sigma_up-sigma_low=0
        A[4,0] = (-H_up*sigma_d1*kc_up*np.sin(kc_up*z)
                  -alpha_up*M_up*(kc_up*np.sin(kc_up*z)))
        A[4,1] = (H_up*sigma_d1*kc_up*np.cos(kc_up*z)
                  +alpha_up*M_up*kc_up*np.cos(kc_up*z))
        A[4,2] = (-H_up*sigma_c1*kd_up*np.sin(kd_up*z)
                  -alpha_up*M_up*(kd_up*np.sin(kd_up*z)))
        A[4,3] = (H_up*sigma_c1*kd_up*np.cos(kd_up*z)
                  +alpha_up*M_up*kd_up*np.cos(kd_up*z))
        A[4,4] = (-(-H_low*sigma_d2*kc_low*np.sin(kc_low*z)
                    -alpha_low*M_low*(kc_low*np.sin(kc_low*z))))
        A[4,5] = (-(H_low*sigma_d2*kc_low*np.cos(kc_low*z)
                    +alpha_low*M_low*kc_low*np.cos(kc_low*z)))
        A[4,6] = (-(-H_low*sigma_c2*kd_low*np.sin(kd_low*z)
                    -alpha_low*M_low*(kd_low*np.sin(kd_low*z))))
        A[4,7] = (-(H_low*sigma_c2*kd_low*np.cos(kd_low*z)
                    +alpha_low*M_low*kd_low*np.cos(kd_low*z)))
        # p_up(8) = p_low(0) -> p_up-p_low=0 
        A[5,0] = (alpha_up*M_up*sigma_d1*kc_up*np.sin(kc_up*z)
                  +M_up*kc_up*np.sin(kc_up*z))
        A[5,1] = (-alpha_up*M_up*sigma_d1*kc_up*np.cos(kc_up*z)
                  -M_up*kc_up*np.cos(kc_up*z))
        A[5,2] = (alpha_up*M_up*sigma_c1*kd_up*np.sin(kd_up*z)
                  +M_up*kd_up*np.sin(kd_up*z))
        A[5,3] = (-alpha_up*M_up*sigma_c1*kd_up*np.cos(kd_up*z)
                  -M_up*kd_up*np.cos(kd_up*z))
        A[5,4] = (-(alpha_low*M_low*sigma_d2*kc_low*np.sin(kc_low*z)
                    +M_low*kc_low*np.sin(kc_low*z)))
        A[5,5] = (-(-alpha_low*M_low*sigma_d2*kc_low*np.cos(kc_low*z)
                    -M_low*kc_low*np.cos(kc_low*z)))
        A[5,6] = (-(alpha_low*M_low*sigma_c2*kd_low*np.sin(kd_low*z)
                    +M_low*kd_low*np.sin(kd_low*z)))
        A[5,7] = (-(-alpha_low*M_low*sigma_c2*kd_low*np.cos(kd_low*z)
                    -M_low*kd_low*np.cos(kd_low*z)))

        z = b
        # sigma(b) = -Pc
        A[6,0] = 0
        A[6,1] = 0
        A[6,2] = 0
        A[6,3] = 0
        A[6,4] = (-H_low*sigma_d2*kc_low*np.sin(kc_low*z)
                  -alpha_low*M_low*kc_low*np.sin(kc_low*z))
        A[6,5] = (H_low*sigma_d2*kc_low*np.cos(kc_low*z)
                  +alpha_low*M_low*kc_low*np.cos(kc_low*z))
        A[6,6] = (-H_low*sigma_c2*kd_low*np.sin(kd_low*z)
                  -alpha_low*M_low*kd_low*np.sin(kd_low*z))
        A[6,7] = (H_low*sigma_c2*kd_low*np.cos(kd_low*z)
                  +alpha_low*M_low*kd_low*np.cos(kd_low*z))
        # w_low(b) = 0
        A[7,0] = 0
        A[7,1] = 0
        A[7,2] = 0
        A[7,3] = 0
        A[7,4] = np.cos(kc_low*z)
        A[7,5] = np.sin(kc_low*z)
        A[7,6] = np.cos(kd_low*z)
        A[7,7] = np.sin(kd_low*z)

        # solve AX=Y
        if not np.all(np.isfinite(A)):
            raise FloatingPointError(f"A contains NaN/Inf at f={f[i]} Hz")
        row_scale = np.max(np.abs(A), axis=1)
        if np.any(row_scale == 0):
            raise np.linalg.LinAlgError(f"A contains zero row at f={f[i]} Hz")
        A_scaled = A / row_scale[:, None]
        Y_scaled = Y / row_scale
        cond_scaled = np.linalg.cond(A_scaled)
        if cond_scaled > 1e10:
            print(f"Warning: A is ill-conditioned "f"at f={f[i]:.4e} Hz, "f"cond={cond_scaled:.3e}")
        X = np.linalg.solve(A_scaled,Y_scaled)
        # X = np.linalg.solve(A,Y)
        b1 = X[0] 
        b2 = X[1] 
        b3 = X[2] 
        b4 = X[3] 
        b5 = X[4] 
        b6 = X[5] 
        b7 = X[6]
        b8 = X[7] 

        # P wave modulus
        z    = -a
        u1   = (sigma_d1*(b1*np.cos(kc_up*z)+b2*np.sin(kc_up*z))
                +sigma_c1*(b3*np.cos(kd_up*z)+b4*np.sin(kd_up*z)))
        z    = b
        u2   = (sigma_d2*(b5*np.cos(kc_low*z)+b6*np.sin(kc_low*z))
                +sigma_c2*(b7*np.cos(kd_low*z)+b8*np.sin(kd_low*z)))
        H[i] = -Pc*(a+b)/(u2-u1)

    # fluid pressure, p=-αMdu/dz-Mdw/dz
    # lower layer
    z_low  = np.linspace(b/100,b,100)
    du2_dx = (-b5*sigma_d2*kc_low*np.sin(kc_low*z_low)
              +b6*sigma_d2*kc_low*np.cos(kc_low*z_low)
              -b7*sigma_c2*kd_low*np.sin(kd_low*z_low)
              +b8*sigma_c2*kd_low*np.cos(kd_low*z_low))
    dw2_dx = (-b5*kc_low*np.sin(kc_low*z_low)
              +b6*kc_low*np.cos(kc_low*z_low)
              -b7*kd_low*np.sin(kd_low*z_low)
              +b8*kd_low*np.cos(kd_low*z_low))
    pf_low = (-alpha_low*M_low*du2_dx
              -M_low*dw2_dx)
    # upper layer
    z_up  = np.linspace(-a,0,100)
    du2_dx = (-b1*sigma_d1*kc_up*np.sin(kc_up*z_up)
              +b2*sigma_d1*kc_up*np.cos(kc_up*z_up)
              -b3*sigma_c1*kd_up*np.sin(kd_up*z_up)
              +b4*sigma_c1*kd_up*np.cos(kd_up*z_up))
    dw2_dx = (-b1*kc_up*np.sin(kc_up*z_up)
              +b2*kc_up*np.cos(kc_up*z_up)
              -b3*kd_up*np.sin(kd_up*z_up)
              +b4*kd_up*np.cos(kd_up*z_up))
    pf_up = (-alpha_up*M_up*du2_dx
              -M_up*dw2_dx)  

    h  = np.append(z_up,z_low)
    pf = np.append(pf_up,pf_low)   
    
    return H,h,pf 