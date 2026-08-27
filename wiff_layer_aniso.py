from utils import coth
import numpy as np
'''
==================================
Function Index:
==================================
biot_layer_aniso(rock_up,rock_low,fluid_up,fluid_low,perm,d1,d2,f,Pc,strain0)
    -> C11,C13,C33,C44,C66,h,pf_C33,pf_C11

biot_aniso_background_layer(Cd_up,fluid_up,perm_up,phi_up,d1,
                        Cd_low,fluid_low,perm_low,phi_low,d2,
                        grain,f):
    -> C11,C13,C33,C44,C66
'''
def biot_layer_aniso(rock_up,rock_low,fluid_up,fluid_low,perm,d1,d2,f,Pc,strain0):
    '''
    Author: xuesong hu
    Date: 2026-8-8 
    Description: calculate the frequency-dependent anisotropic elastic stiffnesses (VTI) ...
                 of a two-layer fluid-saturated porous medium using the quasi-static Biot equations.

    Parameters
    rock_up   = [Ks_up,Gs_up,rhos_up,Kd_up,Gd_up,phi_up]
    fluid_up  = [Kf_up,rhof_up,vis_up]
    rock_low  = [Ks_low,Gs_low,rhos_low,Kd_low,Gd_low,phi_low]
    fluid_low = [Kf_low,rhof_low,vis_low]
    perm      = [perm_up,perm_low] [m^2]
    d1: thickness of upper layer [m]
    d2: thickness of lower layer [m]
    f:  frequency [Hz]      
    Pc: harmonic oscillation stress amplitude on upper and lower surface [Pa]
    strain0: harmonic oscillation strain amplitude on upper and lower surface [-]

    Returns
    C11,C13,C33,C44,C66
    h: spatial coordinates across the layered medium [m]
    pf_C33: fluid-pressure distribution associated with the C33 calculation at the last input frequency [Pa]

    Reference
    Jianping Liao et al. - 2023 - Seismic dispersion, attenuation and frequency-dependent anisotropy in a fluid-saturated porous periodically layered medium
    '''
    f = np.atleast_1d(np.asarray(f, dtype=float))
    a = d1/2
    b = d2/2
    # upper layer
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

    rho_up  = rhof_up*phi_up+rhos_up*(1-phi_up)
    rho_low = rhof_low*phi_low+rhos_low*(1-phi_low)
    rho     = rho_up*a/(a+b)+rho_low*b/(a+b)

    # define D and gamma
    # upper layer
    alpha_up = 1-Kd_up/Ks_up
    M_up     = 1 / ((alpha_up-phi_up)/Ks_up+phi_up/Kf_up)
    C11a     = Kd_up + 4/3*Gd_up
    C11a_sat = C11a + alpha_up**2 * M_up
    D_up     = perm_up/vis_up*M_up*C11a/C11a_sat
    gamma_up = alpha_up*M_up/C11a_sat
    C12a     = Kd_up - 2/3*Gd_up
    C44a     = Gd_up
    kappa_up = 2*alpha_up*C44a*M_up/C11a_sat
    # lower layer
    alpha_low = 1-Kd_low/Ks_low
    M_low     = 1 / ((alpha_low-phi_low)/Ks_low+phi_low/Kf_low)
    C11b      = Kd_low + 4/3*Gd_low
    C11b_sat  = C11b + alpha_low**2 * M_low
    D_low     = perm_low/vis_low*M_low*C11b/C11b_sat
    gamma_low = alpha_low*M_low/C11b_sat
    C12b      = Kd_low - 2/3*Gd_low
    C44b      = Gd_low
    kappa_low = 2*alpha_low*C44b*M_low/C11b_sat

    # C33,C13
    A = np.zeros((4,4),dtype=np.complex128)
    Y = np.zeros(4,dtype=np.complex128)
    C33 = np.zeros(len(f),dtype=np.complex128)
    C13 = np.zeros(len(f),dtype=np.complex128)
    for i in range(len(f)):
        omega = 2*np.pi*f[i]
        q1    = np.sqrt(1j*omega/D_up)
        q2    = np.sqrt(1j*omega/D_low)

        z = -a
        A[0,0] = q1*np.exp(q1*z)
        A[0,1] = -q1*np.exp(-q1*z)
        A[0,2] = 0
        A[0,3] = 0
        Y[0]   = 0

        z = 0
        A[1,0] = perm_up/vis_up*q1*np.exp(q1*z)
        A[1,1] = -perm_up/vis_up*q1*np.exp(-q1*z)
        A[1,2] = -perm_low/vis_low*q2*np.exp(q2*z)
        A[1,3] = perm_low/vis_low*q2*np.exp(-q2*z)
        Y[1]   = 0

        A[2,0] = np.exp(q1*z)
        A[2,1] = np.exp(-q1*z)
        A[2,2] = -np.exp(q2*z)
        A[2,3] = -np.exp(-q2*z)
        Y[2]   = (gamma_low-gamma_up)*Pc

        z = b
        A[3,0] = 0
        A[3,1] = 0
        A[3,2] = q2*np.exp(q2*z)
        A[3,3] = -q2*np.exp(-q2*z)
        Y[3]   = 0

        # solve AX=Y
        X  = np.linalg.solve(A,Y)
        E1 = X[0]
        F1 = X[1] 
        E2 = X[2] 
        F2 = X[3]

        int_strainA_33 = (alpha_up*E1*(1-np.exp(-q1*a))/q1 + alpha_up*F1*(np.exp(q1*a)-1)/q1 + (alpha_up*gamma_up-1)*Pc*a) / C11a
        int_strainB_33 = (alpha_low*E2*(np.exp(q2*b)-1)/q2 + alpha_low*F2*(1-np.exp(-q2*b))/q2 + (alpha_low*gamma_low-1)*Pc*b) / C11b
        e33 = (int_strainA_33 + int_strainB_33)/(a+b)
        C33[i] = -Pc/e33

        ratio_a = C12a / C11a
        ratio_b = C12b / C11b
        int_sigmaA_11 = (alpha_up*E1*(ratio_a-1)*(1-np.exp(-q1*a))/q1 
                         +alpha_up*F1*(ratio_a-1)*(np.exp(q1*a)-1)/q1
                         +Pc*(ratio_a*(alpha_up*gamma_up-1)-alpha_up*gamma_up)*a)
        int_sigmaB_11 = (alpha_low*E2*(ratio_b-1)*(np.exp(q2*b)-1)/q2
                         +alpha_low*F2*(ratio_b-1)*(1-np.exp(-q2*b))/q2
                         +Pc*(ratio_b*(alpha_low*gamma_low-1)-alpha_low*gamma_low)*b)
        sigma11 = (int_sigmaA_11 + int_sigmaB_11) / (a + b)
        C13[i] = sigma11 / e33
    # fluid pressure
    z_up  = np.linspace(-a,-a/100,100)
    pf_up = E1*np.exp(q1*z_up)+F1*np.exp(-q1*z_up)+gamma_up*Pc
    z_low  = np.linspace(0,b,100)
    pf_low = E2*np.exp(q2*z_low)+F2*np.exp(-q2*z_low)+gamma_low*Pc
    h      = np.append(z_up,z_low)
    pf_C33 = np.append(pf_up,pf_low)

    # C11
    A = np.zeros((4,4),dtype=np.complex128)
    Y = np.zeros(4,dtype=np.complex128)
    C11 = np.zeros(len(f),dtype=np.complex128)
    for i in range(len(f)):
        omega = 2*np.pi*f[i]
        q1    = np.sqrt(1j*omega/D_up)
        q2    = np.sqrt(1j*omega/D_low)

        z = -a
        A[0,0] = q1*np.exp(q1*z)
        A[0,1] = -q1*np.exp(-q1*z)
        A[0,2] = 0
        A[0,3] = 0
        Y[0]   = 0

        z = 0
        A[1,0] = perm_up/vis_up*q1*np.exp(q1*z)
        A[1,1] = -perm_up/vis_up*q1*np.exp(-q1*z)
        A[1,2] = -perm_low/vis_low*q2*np.exp(q2*z)
        A[1,3] = perm_low/vis_low*q2*np.exp(-q2*z)
        Y[1]   = 0

        A[2,0] = np.exp(q1*z)
        A[2,1] = np.exp(-q1*z)
        A[2,2] = -np.exp(q2*z)
        A[2,3] = -np.exp(-q2*z)
        Y[2]   = (kappa_low-kappa_up)*strain0

        z = b
        A[3,0] = 0
        A[3,1] = 0
        A[3,2] = q2*np.exp(q2*z)
        A[3,3] = -q2*np.exp(-q2*z)
        Y[3]   = 0   
        # solve AX=Y
        X  = np.linalg.solve(A,Y)
        E1 = X[0]
        F1 = X[1] 
        E2 = X[2] 
        F2 = X[3]    

        int_sigmaA_11 = (-(C11a-C12a**2/C11a)*strain0*a
                         -2*alpha_up*C44a/C11a*(E1*(1-np.exp(-q1*a))/q1
                                                + F1*(np.exp(q1*a)-1)/q1
                                                +kappa_up*strain0*a))      
        int_sigmaB_11 = (-(C11b-C12b**2/C11b)*strain0*b
                         -2*alpha_low*C44b/C11b*(E2*(np.exp(q2*b)-1)/q2
                                                     + F2*(1-np.exp(-q2*b))/q2
                                                     +kappa_low*strain0*b))
        sigma11 = (int_sigmaA_11 + int_sigmaB_11) / (a + b)
        C11[i] = (-sigma11/strain0+C13[i]**2/C33[i])
    # fluid pressure 
    pf_up  = E1*np.exp(q1*z_up)+F1*np.exp(-q1*z_up)+kappa_up*strain0
    pf_low = E2*np.exp(q2*z_low)+F2*np.exp(-q2*z_low)+kappa_low*strain0
    pf_C11 = np.append(pf_up,pf_low)

    # C44,C66
    C44 = np.zeros(len(f),dtype=np.complex128)
    C66 = np.zeros(len(f),dtype=np.complex128)
    for i in range(len(f)):
        C44[i] = (a/(a+b)*(1/C44a)+b/(a+b)*(1/C44b))**(-1)
        C66[i] = (a/(a+b)*(C44a)+b/(a+b)*(C44b))

    return C11,C13,C33,C44,C66,h,pf_C33,pf_C11

def biot_aniso_background_layer(Cd_up,fluid_up,perm_up,phi_up,d1,
                          Cd_low,fluid_low,perm_low,phi_low,d2,
                          grain,f):
    '''
    Author: xuesong hu
    Date: 2026-8-8 
    Description: calculate the frequency-dependent effective VTI stiffnesses of a
                 two-layer anisotropic porous medium using anisotropic Biot theory.
    
    Parameters
    -----------------------------------------------------------
    upper layer:
    Cd_up      = [C11_up, C13_up, C33_up, C44_up, C66_up] [Pa]
    fluid_up  = [Kf_up, vis_up, rhof_up]
    perm_up: permeability of the upper layer [m^2]
    phi_up: porosity of the upper layer [-]
    d1: thickness of the upper layer [m]
    -----------------------------------------------------------
    lower layer:
    Cd_low    = [C11_low, C13_low, C33_low, C44_low, C66_low] [Pa]
    fluid_low = [Kf_low, vis_low, rhof_low]
    perm_low: permeability of the lower layer [m^2]
    phi_low: porosity of the lower layer [-]
    d2: thickness of the lower layer [m]
    -----------------------------------------------------------
    grain = [Ks, Gs, rhos]
    f: Frequency [Hz]

    Returns
    C11, C13, C33, C44, C66
    
    Reference
    Dan He and Junxin Guo - 2024 - Dynamic seismic signatures in a fluid saturated porous periodically layered medium considering effects of intrinsic anisotropy 
    '''
    a = d1/2
    b = d2/2
    # upper layer
    C11_up = Cd_up[0]
    C13_up = Cd_up[1]
    C33_up = Cd_up[2]
    C44_up = Cd_up[3]
    C66_up = Cd_up[4]
    C12_up = C11_up - 2*C66_up
    Kf_up   = fluid_up[0]
    vis_up  = fluid_up[1]
    rhof_up = fluid_up[2]
    # lower layer
    C11_low  = Cd_low[0]
    C13_low  = Cd_low[1]
    C33_low  = Cd_low[2]
    C44_low  = Cd_low[3]
    C66_low  = Cd_low[4]
    C12_low  = C11_low - 2*C66_low
    Kf_low   = fluid_low[0]
    vis_low  = fluid_low[1]
    rhof_low = fluid_low[2] 
    # grain
    Ks   = grain[0]
    Gs   = grain[1]
    rhos = grain[2]
    # frequency
    omega = 2*np.pi*f
    # fractiona   
    L  = a + b
    f1 = a / L
    f2 = b / L

    # main routine
    # alpha
    alpha1_up = 1 - (C11_up+C12_up+C13_up)/(3*Ks)
    alpha3_up = 1 - (2*C13_up+C33_up)/(3*Ks)
    alpha1_low = 1 - (C11_low+C12_low+C13_low)/(3*Ks)
    alpha3_low = 1 - (2*C13_low+C33_low)/(3*Ks)
    # M
    M_up  = Ks/(1-phi_up*(1-Ks/Kf_up)-(2*C11_up+2*C12_up+4*C13_up+C33_up)/(9*Ks))
    M_low = Ks/(1-phi_low*(1-Ks/Kf_low)-(2*C11_low+2*C12_low+4*C13_low+C33_low)/(9*Ks))
    # C33sat_low_fre
    C33sat_up  = C33_up+alpha3_up**2 *M_up
    C33sat_low = C33_low+alpha3_low**2 *M_low 
    # D
    D_up  = (perm_up*M_up*C33_up)/(vis_up*C33sat_up)
    D_low = (perm_low*M_low*C33_low)/(vis_low*C33sat_low)
    # lambda
    lam_up  = np.sqrt((1j*omega)/D_up)
    lam_low = np.sqrt((1j*omega)/D_low)
    # gamma
    gamma_up  = (alpha3_up*M_up)/C33sat_up
    gamma_low = (alpha3_low*M_low)/C33sat_low

    # C33
    C33_hf   = 1/(f1/C33sat_up+f2/C33sat_low)
    A        = (vis_up/(lam_up*perm_up))*coth(lam_up*a)
    B        = (vis_low/(lam_low*perm_low))*coth(lam_low*b)
    gama2    = (gamma_up-gamma_low)**2
    C33 = C33_hf * 1/((C33_hf*gama2)/(1j*omega*L*(A+B))+1)

    # C13
    C1       = C13_up*gamma_up-C13_low*gamma_low-(gamma_up*C33_up*alpha1_up)/alpha3_up+(gamma_low*C33_low*alpha1_low)/alpha3_low
    C2       = (f1*(gamma_up*((C13_up*alpha3_up)/C33_up-alpha1_up)-C13_up/C33_up)
                +f2*(gamma_low*((C13_low*alpha3_low)/C33_low-alpha1_low)-C13_low/C33_low))
    C13_var1 = ((gamma_up-gamma_low)*C1)/(1j*omega*L*(A+B))-C2
    C13_var2 = f1/C33sat_up+f2/C33sat_low+gama2/(1j*omega*L*(A+B))
    C13      = C13_var1/C13_var2

    # C11
    x_up  = (M_up*(alpha1_up*C33_up-alpha3_up*C13_up))/C33sat_up
    x_low = (M_low*(alpha1_low*C33_low-alpha3_low*C13_low))/C33sat_low
    D1   = f1*(C11_up*C33_up-C13_up**2)/(C33_up)+f2*(C11_low*C33_low-C13_low**2)/(C33_low)
    D2   = f1*((alpha3_up*C13_up-alpha1_up*C33_up)*x_up/C33_up)+f2*((alpha3_low*C13_low-alpha1_low*C33_low)*x_low/C33_low)
    D3   = (x_up-x_low)**2 / (1j*omega*L*(A+B))
    C11  = D1-D2-D3+C13**2/C33

    # C44
    C44  = 1/(f1/C44_up+f2/C44_low)

    # C66
    C66  = f1*C66_up+f2*C66_low

    return C11,C13,C33,np.linspace(C44,C44,len(f)),np.linspace(C66,C66,len(f))