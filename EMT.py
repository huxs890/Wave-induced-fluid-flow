import numpy as np
'''
Effective Medium Theory (EMT)
==================================
Function Index:
==================================
hudson(K1,G1,K2,G2,asp,epsilon)
    -> C1_aniso,C2_aniso

dem(Km,Gm,Kf,Gf,asp,phi,dphi)
    -> K_dem,G_dem
'''
def hudson(K1,G1,K2,G2,asp,epsilon):
    '''
    Author: xuesong hu
    Date: 2026-7-30
    
    Description
    Calculate the effective anisotropic elastic stiffnesses of a cracked ...
    medium using Hudson's first- and second-order effective medium theory.

    Parameters
    K1: bulk modulus of the isotropic background medium [Pa]
    G1: shear modulus of the isotropic background medium [Pa]
    K2: bulk modulus of the crack-filling material [Pa]
    G2: shear modulus of the crack-filling material [Pa]
    asp: aspect ratio of the penny-shaped cracks [-]
    epsilon: crack density parameter [-], small crack densities due to Hudson's perturbation approximation

    Returns:
    C1_aniso: effective stiffness coefficients obtained using Hudson's first-order approximation
            np.array([C11_ani1,C13_ani1,C33_ani1,C44_ani1,C66_ani1]), [Pa]
    C2_aniso: effective stiffness coefficients obtained using Hudson's second-order approximation
            np.array([C11_ani2,C13_ani2,C33_ani2,C44_ani2,C66_ani2]), [Pa]

    Reference
    Gary Mavko, Tapan Mukerji, Jack Dvorkin. - 2020 - The Rock Physics Handbook, Third Edition
    '''
    lamda1 = K1-2/3*G1
    lamda2 = K2-2/3*G2
    q = 15*lamda1**2/(G1**2)+28*lamda1/G1+28

    # inclusion medium
    M = 4*G2/(np.pi*asp*G1)*(lamda1+2*G1)/(3*lamda1+4*G1)
    kappa = (K2+4/3*G2)*(lamda1+2*G1)/((np.pi*asp*G1)*(lamda1+G1))
    U1 = (16/3)*(lamda1+2*G1)/(3*lamda1+4*G1)*(1/(1+M))
    U3 = (4/3)*(lamda1+2*G1)/(lamda1+G1)*(1/(1+kappa))

    # background medium
    C11_back = lamda1+2*G1
    C13_back = lamda1
    C33_back = lamda1+2*G1
    C44_back = G1
    C66_back = G1

    # first-order approximation
    C11_1 = -lamda1**2/G1*epsilon*U3
    C13_1 = -lamda1*(lamda1+2*G1)/G1*epsilon*U3
    C33_1 = -(lamda1+2*G1)**2/G1*epsilon*U3
    C44_1 = -G1*epsilon*U1
    C66_1 = 0

    C11_ani1 = C11_back+C11_1
    C13_ani1 = C13_back+C13_1
    C33_ani1 = C33_back+C33_1
    C44_ani1 = C44_back+C44_1
    C66_ani1 = C66_back+C66_1
    C1_aniso = np.array([C11_ani1,C13_ani1,C33_ani1,C44_ani1,C66_ani1])

    # second-order approximation
    C11_2 = q/15*lamda1**2*((epsilon*U3)**2)/(lamda1+2*G1)
    C13_2 = q/15*lamda1*(epsilon*U3)**2
    C33_2 = q/15*(lamda1+2*G1)*(epsilon*U3)**2
    C44_2 = 2/15*G1*(3*lamda1+8*G1)*((epsilon*U1)**2)/(lamda1+2*G1)
    C66_2 = 0

    C11_ani2 = C11_back+C11_1+C11_2
    C13_ani2 = C13_back+C13_1+C13_2
    C33_ani2 = C33_back+C33_1+C33_2
    C44_ani2 = C44_back+C44_1+C44_2
    C66_ani2 = C66_back+C66_1+C66_2
    C2_aniso = np.array([C11_ani2,C13_ani2,C33_ani2,C44_ani2,C66_ani2])

    return C1_aniso,C2_aniso

def dem(Km,Gm,Kf,Gf,asp,phi,dphi):
    '''
    Author: xuesong hu
    Date: 2026-7-30
        
    Description
    Calculate the effective bulk and shear moduli of a composite medium ...
    using the Differential Effective Medium (DEM) theory.

    Parameters:
    Km: bulk modulus of the initial matrix material [Pa]
    Gm: shear modulus of the initial matrix material [Pa]
    Kf: bulk modulus of the inclusion material [Pa]
    Gf: shear modulus of the inclusion material [Pa]
    asp: aspect ratio of the spheroidal inclusions [-]
        asp < 1 : Oblate spheroidal inclusions, commonly used to represent cracks or pores
        asp = 1 : Spherical inclusions
        asp > 1 : Prolate spheroidal inclusions
    phi: final volume fraction of inclusions [-]
    dphi: increment of inclusion volume fraction used in each DEM iteration [-], ...
        a smaller value generally provides a more accurate numerical approximation

    Returns
    K_dem: effective bulk modulus of the composite medium predicted by the DEM model [Pa]
    G_dem: effective shear modulus of the composite medium predicted by the DEM model [Pa]

    Reference
    Gary Mavko, Tapan Mukerji, Jack Dvorkin. - 2020 - The Rock Physics Handbook, Third Edition
    '''
    if dphi > phi:
        K_dem = Km
        G_dem = Gm
    else:
        num      = phi/dphi
        phi_init = 0

        for i in range(num):
            A = Gf/Gm-1
            B = 1/3*(Kf/Km-Gf/Gm)
            R = (3*Gm)/(3*Km+4*Gm) 

            if asp > 1:
                theta = (asp*(asp*np.sqrt((asp*asp-1))-np.acosh(asp)))/((asp*asp-1)**(3/2))
            elif asp < 1:
                theta = (asp*(np.acos(asp)-asp*np.sqrt((1-asp*asp))))/((1-asp*asp)**(3/2))
            f=(asp*asp*(3*theta-2))/(1-asp*asp)

            F1 = 1+A*(1.5*(f+theta)-R*(1.5*f+2.5*theta-4/3))
            F2 = 1+A*(1+1.5*(f+theta)-(R/2)*(3*f+5*theta))+B*(3-4*R)+0.5*A*(A+3*B)*(3-4*R)*(f+theta-R*(f-theta+2*theta*theta))
            F3 = 1+A*(1-(f+1.5*theta)+R*(f+theta))
            F4 = 1+(A/4)*(f+3*theta-R*(f-theta))
            F5 = A*(-f+R*(f+theta-4/3))+B*theta*(3-4*R)
            F6 = 1+A*(1+f-R*(f+theta))+B*(1-theta)*(3-4*R)
            F7 = 2+(A/4)*(3*f+9*theta-R*(3*f+5*theta))+B*theta*(3-4*R)
            F8 = A*(1-2*R+(f/2)*(R-1)+(theta/2)*(5*R-3))+B*(1-theta)*(3-4*R)
            F9 = A*((R-1)*f-R*theta)+B*theta*(3-4*R)

            P = F1/F2
            Q = 1/5*(2/F3+1/F4+(F4*F5+F6*F7-F8*F9)/(F2*F4))

            K_dem = Km+((Kf-Km)*dphi)/((1-phi_init))*P
            G_dem = Gm+((Gf-Gm)*dphi)/((1-phi_init))*Q
            phi_init = phi+dphi
            Km    = K_dem
            Km    = K_dem

    return K_dem,G_dem