import numpy as np
'''
Date: 2026-8-4
Written by xuesong hu
==================================
Function Index:
==================================
y = coth(x)

modulus_to_velocity(K,G,rho)
    -> Vp,Vs

velocity_to_modulus(Vp,Vs,rho)
    -> K,G

phase_velocity_vti(C11,C13,C33,C44,C66,theta,rho)   
    -> Vp,Vsv,Vsh
'''
def coth(x):
    return 1 / np.tanh(x)

def modulus_to_velocity(K,G,rho):
    '''
    Bulk and shear modulus to Vp and Vs
    '''
    Vp = np.sqrt((K+4/3*G)/rho)
    Vs = np.sqrt(G/rho)
    return Vp,Vs

def velocity_to_modulus(Vp,Vs,rho):
    '''
    Vp and Vs to bulk and shear modulus
    '''
    G = rho * Vs**2
    K = rho * (Vp**2 - 4/3* Vs**2)
    return K,G

def phase_velocity_vti(C11,C13,C33,C44,C66,theta,rho):
    M=((C11-C44)*(np.sin(theta))**2-(C33-C44)*(np.cos(theta))**2)**2+((C13+C44)**2)*(np.sin(2*theta))**2
    Vp = np.sqrt(C11*(np.sin(theta))**2+C33*(np.cos(theta))**2+C44+np.sqrt(M))/(np.sqrt(2*rho))
    Vsv = np.sqrt(C11*(np.sin(theta))**2+C33*(np.cos(theta))**2+C44-np.sqrt(M))/(np.sqrt(2*rho))
    Vsh = np.sqrt((C66*(np.sin(theta))**2+C44*(np.cos(theta))**2)/rho)
    return Vp,Vsv,Vsh