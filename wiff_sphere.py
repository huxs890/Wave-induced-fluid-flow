import numpy as np
'''
==================================
Function Index:
==================================
white_sphere_1975(skeleton,fluid_in,fluid_out,perm,a,b,f)
    -> K,Vp,inv_Qp

biot_sphere_matrix(rock_in,rock_out,fluid_in,fluid_out,perm,a,b,f,Pc,undrained,Pf_out)
    -> K,r,pf

dynamic_biot_sphere_matrix(rock_in,rock_out,fluid_in,fluid_out,perm,a,b,f,s,Pc,undrained,Pf_out)
    -> K
'''
def white_sphere_1975(skeleton,fluid_in,fluid_out,perm,a,b,f):
    '''
    Author: xuesong hu
    Date: 2026-8-5   
    
    Description: calculate the frequency-dependent bulk modulus, P-wave velocity,
                 and attenuation of a partially saturated porous rock using the White(1975) concentric-sphere model.
                 Wave-induced fluid pressure diffusion between the two regions produces frequency-dependent dispersion and attenuation.
    
    Parameters
    skeleton  = [Ks,Gs,rhos,Kd,Gd,phi]
    fluid_in  = [Kf_in,rhof_in,vis_in]
    fluid_out = [Kf_out,rhof_out,vis_out]
    perm: permeability, 1*1 [m^2]
    a: radius of inner sphere [m]
    b: outer radius of the spherical model [m]
    f: frequency [Hz]

    Returns
    K: Frequency-dependent effective bulk modulus [Pa]
    Vp: P-wave phase velocity [m/s]
    inv_Qp: Inverse P-wave quality factor

    Reference
    White - 1975 - Computed seismic speeds and attenuation in rocks with partial gas saturation
    Dutta and Seriff - 1979 - On White's model of attenuation in rocks with partial gas saturation
    Gary Mavko, Tapan Mukerji, Jack Dvorkin. - 2020 - The Rock Physics Handbook, Third Edition
    '''
    Sg = (a/b)**3
    Sw = 1-Sg
    Ks   = skeleton[0]
    Gs   = skeleton[1] 
    rhos = skeleton[2]
    Kd   = skeleton[3]
    Gd   = skeleton[4]
    phi  = skeleton[5]
    omega = 2*np.pi*f
    alpha = 1 - Kd/Ks
    # inner sphere
    Kf_in   = fluid_in[0]
    rhof_in = fluid_in[1]
    vis_in  = fluid_in[2]

    M_in  = 1/((alpha-phi)/Ks+phi/Kf_in)
    Kg_in = Kd + alpha**2 *M_in
    Eg_in = Kg_in + 4/3*Gd

    # outter shell
    Kf_out   = fluid_out[0]
    rhof_out = fluid_out[1]
    vis_out  = fluid_out[2]

    M_out  = 1/((alpha-phi)/Ks+phi/Kf_out)
    Kg_out = Kd + alpha**2 *M_out
    Eg_out = Kg_out + 4/3*Gd

    # main routine
    KE_in   = Kd/Kg_in*M_in
    gama_in = np.sqrt(1j*omega*vis_in/(perm*KE_in))
    Z_in    = (1-np.exp(-2*gama_in*a))/((gama_in*a-1)+(gama_in*a+1)*np.exp(-2*gama_in*a))
    R_in   = (Kg_in-Kd)*(3*Kg_out+4*Gd)/(Kg_out*(3*Kg_in+4*Gd)+4*Gd*(Kg_in-Kg_out)*Sg)

    KE_out  = Kd/Kg_out*M_out
    gama_out= np.sqrt(1j*omega*vis_out/(perm*KE_out))
    m1    = (gama_out*b+1)*np.exp(-2*gama_out*(b-a))+(gama_out*b-1)
    m2   = (gama_out*b+1)*(gama_out*a-1)*np.exp(-2*gama_out*(b-a))-(gama_out*b-1)*(gama_out*a+1)
    Z_out    = m1/m2
    R_out   = (Kg_out-Kd)*(3*Kg_in+4*Gd)/(Kg_out*(3*Kg_in+4*Gd)+4*Gd*(Kg_in-Kg_out)*Sg)

    c1   = 3j*a*perm*(R_in-R_out)*(M_in/Kg_in-M_out/Kg_out)
    c2   = b**3 * omega*(vis_in*Z_in-vis_out*Z_out)
    W    = c1/c2
    Kinf = 1/(Sg/Eg_in+Sw/Eg_out)-(4/3)*Gd
    K    = Kinf/(1-W*Kinf)
    H    = K+(4/3)*Gd

    rhof=Sg*rhof_in+Sw*rhof_out
    rho=(1-phi)*rhos+phi*rhof
    Vp   = np.sqrt(np.real(H)/rho)
    inv_Qp = np.imag(H)/np.real(H)

    return K,Vp,inv_Qp


def biot_sphere_matrix(rock_in,rock_out,fluid_in,fluid_out,perm,a,b,f,Pc,undrained,Pf_out):
    '''
    Author: xuesong hu
    Date: 2026-8-5   
    Description: solve the quasi-static Biot equations for a concentric spherical porous medium using a matrix method.
                 both drained and undrained outer boundary conditions are supported

    Parameters
    rock_in   = [Ks_in,Gs_in,rhos_in,Kd_in,Gd_in,phi_in]      
    rock_out  = [Ks_out,Gs_out,rhos_out,Kd_out,Gd_out,phi_out]
    fluid_in  = [Kf_in,rhof_in,vis_in]
    fluid_out = [Kf_out,rhof_out,vis_out]
    perm      = [perm_in, perm_out] [m^2]
    a: radius of the inner sphere [m]
    b: outer radius of the spherical model [m]
    f: Frequency [Hz]
    Pc: harmonic oscillation stress amplitude on outer surface [Pa]
    undrained: boundary-condition flag, (undrained = 1): undrained outer boundary; (undrained = 0): drained outer boundaries;
    Pf_out: prescribed fluid pressure at the drained outer boundaries [Pa], only useful when undrained = 0

    Returns
    K: complex frequency-dependent effective bulk modulus [Pa]
    r: radial coordinates used for the pore-pressure profile [m]
    pf: complex fluid-pressure distribution at the last input frequency [Pa]

    Reference
    Vogelaar et al. - 2010 - Exact expression for the effective acoustics of patchy-saturated rocks
    '''
    f = np.atleast_1d(np.asarray(f, dtype=float))
    # inner shpere
    Ks_in   = rock_in[0]
    Gs_in   = rock_in[1]
    rhos_in = rock_in[2]
    Kd_in   = rock_in[3]
    Gd_in   = rock_in[4]
    phi_in  = rock_in[5] 
    Kf_in   = fluid_in[0]
    rhof_in = fluid_in[1]
    vis_in  = fluid_in[2]
    perm_in = perm[0]

    alpha_in = 1 - Kd_in/Ks_in
    M_in     = 1/((alpha_in-phi_in)/Ks_in+phi_in/Kf_in)
    Ksat_in  = Kd_in + alpha_in**2 *M_in
    L_in     = Kd_in + 4/3*Gd_in
    H_in     = Ksat_in + 4/3*Gd_in
    B_in     = alpha_in*M_in/H_in

    # outer shell
    Ks_out   = rock_out[0]
    Gs_out   = rock_out[1]
    rhos_out = rock_out[2]
    Kd_out   = rock_out[3]
    Gd_out   = rock_out[4]
    phi_out  = rock_out[5] 
    Kf_out   = fluid_out[0]
    rhof_out = fluid_out[1]
    vis_out  = fluid_out[2]
    perm_out = perm[1]

    alpha_out = 1 - Kd_out/Ks_out
    M_out     = 1/((alpha_out-phi_out)/Ks_out+phi_out/Kf_out)
    Ksat_out  = Kd_out + alpha_out**2 *M_out
    L_out     = Kd_out + 4/3*Gd_out
    H_out     = Ksat_out + 4/3*Gd_out
    B_out     = alpha_out*M_out/H_out

    # main routin
    A = np.zeros((6,6),dtype=np.complex128)
    K = np.zeros(len(f),dtype=np.complex128)
    for i in range(len(f)):
        omega = 2*np.pi*f[i]

        # define bessel function
        # inner sphere
        r = a
        D_in    = -perm_in*L_in*M_in/(vis_in*H_in)
        k_in    = np.sqrt(1j*omega/D_in)
        j0_in_a = np.sin(k_in*r)/(k_in*r)
        n0_in_a = -np.cos(k_in*r)/(k_in*r)
        j1_in_a = np.sin(k_in*r)/((k_in*r)**2)-np.cos(k_in*r)/(k_in*r)
        n1_in_a = -np.cos(k_in*r)/((k_in*r)**2)-np.sin(k_in*r)/(k_in*r)
        # outer shell
        r = a
        D_out    = -perm_out*L_out*M_out/(vis_out*H_out)
        k_out    = np.sqrt(1j*omega/D_out)
        j0_out_a = np.sin(k_out*r)/(k_out*r)
        n0_out_a = -np.cos(k_out*r)/(k_out*r)
        j1_out_a = np.sin(k_out*r)/((k_out*r)**2)-np.cos(k_out*r)/(k_out*r)
        n1_out_a = -np.cos(k_out*r)/((k_out*r)**2)-np.sin(k_out*r)/(k_out*r)    
        r = b
        j0_out_b = np.sin(k_out*r)/(k_out*r)
        n0_out_b = -np.cos(k_out*r)/(k_out*r)
        j1_out_b = np.sin(k_out*r)/((k_out*r)**2)-np.cos(k_out*r)/(k_out*r)
        n1_out_b = -np.cos(k_out*r)/((k_out*r)**2)-np.sin(k_out*r)/(k_out*r)
        # assemble matrix A
        # w_in(a)=w_out(a)
        A[0,0] = j1_in_a
        A[0,1] = 0
        A[0,2] = -j1_out_a
        A[0,3] = -n1_out_a
        A[0,4] = 0
        A[0,5] = 0
        # u_in(a)=u_out(a)
        A[1,0] = -B_in*j1_in_a
        A[1,1] = a/H_in
        A[1,2] = B_out*j1_out_a
        A[1,3] = B_out*n1_out_a
        A[1,4] = -a/H_out
        A[1,5] = -1/(H_out* a**2)
        # p_in(a) = p_out(a)
        A[2,0] = -M_in*L_in*k_in/H_in*j0_in_a
        A[2,1] = -3*B_in
        A[2,2] = M_out*L_out*k_out/H_out*j0_out_a
        A[2,3] = M_out*L_out*k_out/H_out*n0_out_a
        A[2,4] = 3*B_out
        A[2,5] = 0
        # sigma_in(a) = sigma_out(a)
        A[3,0] = 4*Gd_in*B_in/a*j1_in_a
        A[3,1] = 3*Ksat_in/H_in
        A[3,2] = -4*Gd_out*B_out/a*j1_out_a
        A[3,3] = -4*Gd_out*B_out/a*n1_out_a
        A[3,4] = -3*Ksat_out/H_out
        A[3,5] = 4*Gd_out/(H_out* a**3)
        # sigma_out(b) = -Pc
        A[4,0] = 0
        A[4,1] = 0
        A[4,2] = 4*Gd_out*B_out/b*j1_out_b
        A[4,3] = 4*Gd_out*B_out/b*n1_out_b
        A[4,4] = 3*Ksat_out/H_out
        A[4,5] = -4*Gd_out/(H_out* b**3)   

        # undrained == 1 : undrained; undrained == 0 : drained
        if undrained == 1:   
            A[5,0] = 0
            A[5,1] = 0
            A[5,2] = j1_out_b
            A[5,3] = n1_out_b
            A[5,4] = 0
            A[5,5] = 0
            Y = np.array([0,0,0,0,-Pc,0],dtype=np.complex128)
        elif undrained == 0:
            A[5,0] = 0
            A[5,1] = 0
            A[5,2] = -M_out*L_out*k_out/H_out*j0_out_b
            A[5,3] = -M_out*L_out*k_out/H_out*n0_out_b
            A[5,4] = -3*B_out
            A[5,5] = 0            
            Y = np.array([0,0,0,0,-Pc,Pf_out],dtype=np.complex128)
        else:
            raise ValueError("undrained must be 0 or 1")
        # Solving algebraic equations —— AX=Y
        X = np.linalg.solve(A,Y)
        c1=X[0]
        c2=0
        c3=X[1]
        c4=0
        c5=X[2]
        c6=X[3]
        c7=X[4]
        c8=X[5]  
        # Outer boundary displacemen and bulk modulus 
        r = b 
        u2_b = c7*r/H_out+c8/(H_out*r**2)-c5*B_out*j1_out_b-c6*B_out*n1_out_b
        K[i] = -b*Pc/(3*u2_b)
    # fluid pressure, the last frequency
    # inner sphere
    r_in  = np.linspace(a/100,a,100)
    j0_in = np.sin(k_in*r_in)/(k_in*r_in)
    n0_in = -np.cos(k_in*r_in)/(k_in*r_in)
    p_in  = -M_in*L_in*k_in/H_in*(c1*j0_in+c2*n0_in)-c3*3*B_in
    # outer shell
    r_out  = np.linspace(a+a/100,b,100)
    j0_out = np.sin(k_out*r_out)/(k_out*r_out)
    n0_out = -np.cos(k_out*r_out)/(k_out*r_out)
    p_out  = -M_out*L_out*k_out/H_out*(c5*j0_out+c6*n0_out)-c7*3*B_out
    r  = np.append(r_in,r_out)
    pf = np.append(p_in,p_out)   

    return K,r,pf

def dynamic_biot_sphere_matrix(rock_in,rock_out,fluid_in,fluid_out,perm,a,b,f,s,Pc,undrained,Pf_out):
    '''
    Author: xuesong hu
    Date: 2026-8-5   
    Description: solve the dynamic Biot equations for a concentric spherical porous medium using a matrix method.
                 now this function only supports undrained outer boundary condition.

    Parameters
    rock_in   = [Ks_in,Gs_in,rhos_in,Kd_in,Gd_in,phi_in]      
    rock_out  = [Ks_out,Gs_out,rhos_out,Kd_out,Gd_out,phi_out]
    fluid_in  = [Kf_in,rhof_in,vis_in]
    fluid_out = [Kf_out,rhof_out,vis_out]
    perm      = [perm_in, perm_out] [m^2]
    a: radius of the inner sphere [m]
    b: outer radius of the spherical model [m]
    f: Frequency [Hz]
    s = [s_in,s_out]: tortuosity parameters of the inner and outer regions [-]
    Pc: harmonic oscillation stress amplitude on outer surface [Pa]
    undrained: boundary-condition flag, (undrained = 1): undrained outer boundary; (undrained = 0): drained outer boundaries;
               currently, only the undrained outer boundary is implemented
    Pf_out: prescribed fluid pressure at the drained outer boundaries [Pa], only useful when undrained = 0

    Returns
    K: complex frequency-dependent effective bulk modulus [Pa]

    Reference
    Dutta and Odé - 1979 - Attenuation and dispersion of compressional waves in fluid-filled porous rocks
                           with partial gas saturation (White model)—Part I: Biot theory
    Dutta and Odé - 1979 - Attenuation and dispersion of compressional waves in fluid-filled porous rocks 
                           with partial gas saturation (White model)—Part Ⅱ: Results   
    Yirong Wang et al. - 2023 - Wave-induced fluid pressure diffusion and anelasticity in partially saturated rocks: 
                                The influences of boundary conditions                  
    '''
    Sg = (a/b)**3
    Sw = 1 - Sg
    f = np.atleast_1d(np.asarray(f, dtype=float))
    # inner shpere
    Ks_in   = rock_in[0]
    Gs_in   = rock_in[1]
    rhos_in = rock_in[2]
    Kd_in   = rock_in[3]
    Gd_in   = rock_in[4]
    phi_in  = rock_in[5] 
    Kf_in   = fluid_in[0]
    rhof_in = fluid_in[1]
    vis_in  = fluid_in[2]
    perm_in = perm[0]
    s_in    = s[0] 

    # outer shell
    Ks_out   = rock_out[0]
    Gs_out   = rock_out[1]
    rhos_out = rock_out[2]
    Kd_out   = rock_out[3]
    Gd_out   = rock_out[4]
    phi_out  = rock_out[5] 
    Kf_out   = fluid_out[0]
    rhof_out = fluid_out[1]
    vis_out  = fluid_out[2]
    perm_out = perm[1]
    s_out    = s[1]

    rho_in  = rhof_in*phi_in + rhos_in*(1-phi_in)
    rho_out = rhof_out*phi_out + rhos_out*(1-phi_out)

    # main routine
    KC1 = np.zeros(len(f),dtype=np.complex128)
    KD1 = np.zeros(len(f),dtype=np.complex128)
    KC2 = np.zeros(len(f),dtype=np.complex128)
    KD2 = np.zeros(len(f),dtype=np.complex128)
    Y   = np.array([0,0,0,0,-Pc,0],dtype=np.complex128)
    K   = np.zeros(len(f),dtype=np.complex128)

    for i in range(len(f)):
        omega = 2*np.pi*f[i]
        # inner sphere
        Q_in     = Kf_in*(Ks_in-Kd_in)/phi_in/(Ks_in-Kf_in)
        K_in     = Ks_in*(Kd_in+Q_in)/(Ks_in+Q_in)

        H_in     = K_in+4*Gd_in/3
        alpha_in = 1 - Kd_in/Ks_in
        D_in     = Ks_in/2/(alpha_in+phi_in/Kf_in*(Ks_in-Kf_in))
        L_in     = Kd_in - 2/3*Gd_in + 2* alpha_in**2 *D_in
        K_in     = Kd_in + 2* alpha_in**2 *D_in
        H_in     = K_in + 4/3*Gd_in
        m_in     = s_in*rhof_in/phi_in

        temp_a = rhof_in*H_in - 2*rho_in*alpha_in*D_in
        temp_b = H_in*m_in-2*rho_in*D_in-1j*vis_in*H_in/(perm_in*omega)
        temp_c = 2*m_in*alpha_in*D_in-2*rhof_in*D_in-2*1j*vis_in*alpha_in*D_in/(perm_in*omega)
        temp_d = temp_b*temp_b-4*temp_a*temp_c
        Oc1    = (-temp_b+np.sqrt(temp_d))/(2*temp_a)
        Od1    = (-temp_b-np.sqrt(temp_d))/(2*temp_a)
        temp = (rhof_in*(H_in*Oc1+2*alpha_in*D_in)-rho_in*(2*alpha_in*D_in*Oc1+2*D_in))/(4*alpha_in**2 *D_in**2 -2*D_in*H_in)
        kc1  = np.sqrt(temp)*omega
        temp = (rhof_in*(H_in*Od1+2*alpha_in*D_in)-rho_in*(2*alpha_in*D_in*Od1+2*D_in))/(4*alpha_in**2 *D_in**2 -2*D_in*H_in)
        kd1  = np.sqrt(temp)*omega

        # outer shell
        Q_out     = Kf_out*(Ks_out-Kd_out)/phi_out/(Ks_out-Kf_out)
        K_out     = Ks_out*(Kd_out+Q_out)/(Ks_out+Q_out)

        H_out     = K_out+4*Gd_out/3
        alpha_out = 1 - Kd_out/Ks_out
        D_out     = Ks_out/2/(alpha_out+phi_out/Kf_out*(Ks_out-Kf_out))
        L_out     = Kd_out - 2/3*Gd_out + 2* alpha_out**2 *D_out
        K_out     = Kd_out + 2* alpha_out**2 *D_out
        H_out     = K_out + 4/3*Gd_out
        m_out     = s_out*rhof_out/phi_out

        temp_a = rhof_out*H_out - 2*rho_out*alpha_out*D_out
        temp_b = H_out*m_out-2*rho_out*D_out-1j*vis_out*H_out/(perm_out*omega)
        temp_c = 2*m_out*alpha_out*D_out-2*rhof_out*D_out-2*1j*vis_out*alpha_out*D_out/(perm_out*omega)
        temp_d = temp_b*temp_b-4*temp_a*temp_c
        Oc2    = (-temp_b+np.sqrt(temp_d))/(2*temp_a)
        Od2    = (-temp_b-np.sqrt(temp_d))/(2*temp_a)
        temp = (rhof_out*(H_out*Oc2+2*alpha_out*D_out)-rho_out*(2*alpha_out*D_out*Oc2+2*D_out))/(4*alpha_out**2 *D_out**2 -2*D_out*H_out)
        kc2  = np.sqrt(temp)*omega
        temp = (rhof_out*(H_out*Od2+2*alpha_out*D_out)-rho_out*(2*alpha_out*D_out*Od2+2*D_out))/(4*alpha_out**2 *D_out**2 -2*D_out*H_out)
        kd2  = np.sqrt(temp)*omega   

        r = a
        zc1=kc1*r    
        zd1=kd1*r
        zc2=kc2*r    
        zd2=kd2*r
        KC1[i]=kc1
        KD1[i]=kd1
        KC2[i]=kc2
        KD2[i]=kd2             
        # bessel function, number 1: inner, number 2: outer
        j1_zc1     = np.sin(zc1)/zc1**2-np.cos(zc1)/zc1
        dj1_zc1_dr = kc1*((np.cos(zc1)*zc1**2
                           -np.sin(zc1)*2*zc1)/zc1**4
                           -(-np.sin(zc1)*zc1-np.cos(zc1))/zc1**2)
        j1_zd1     = np.sin(zd1)/zd1**2-np.cos(zd1)/zd1
        dj1_zd1_dr = kd1*((np.cos(zd1)*zd1**2-np.sin(zd1)*2*zd1)/zd1**4
                          -(-np.sin(zd1)*zd1-np.cos(zd1))/zd1**2)
        j1_zc2     = np.sin(zc2)/zc2**2-np.cos(zc2)/zc2
        dj1_zc2_dr = kc2*((np.cos(zc2)*zc2**2-np.sin(zc2)*2*zc2)/zc2**4
                          -(-np.sin(zc2)*zc2-np.cos(zc2))/zc2**2)
        j1_zd2     = np.sin(zd2)/zd2**2-np.cos(zd2)/zd2
        dj1_zd2_dr = kd2*((np.cos(zd2)*zd2**2-np.sin(zd2)*2*zd2)/zd2**4
                          -(-np.sin(zd2)*zd2-np.cos(zd2))/zd2**2)
        n1_zc2     = -np.cos(zc2)/zc2**2-np.sin(zc2)/zc2
        dn1_zc2_dr = kc2*((np.sin(zc2)*zc2**2+np.cos(zc2)*2*zc2)/zc2**4
                          -(np.cos(zc2)*zc2-np.sin(zc2))/zc2**2)
        n1_zd2     = -np.cos(zd2)/zd2**2-np.sin(zd2)/zd2
        dn1_zd2_dr = kd2*((np.sin(zd2)*zd2**2+np.cos(zd2)*2*zd2)/zd2**4
                          -(np.cos(zd2)*zd2-np.sin(zd2))/zd2**2)
        # assemble matrix A
        A      = np.zeros((6,6),dtype=np.complex128)

        A[0,0] = Od1*(np.sin(zc1)/zc1**2-np.cos(zc1)/zc1)
        A[0,1] = Oc1*(np.sin(zd1)/zd1**2-np.cos(zd1)/zd1)
        A[0,2] = -Od2*(np.sin(zc2)/zc2**2-np.cos(zc2)/zc2)
        A[0,3] = -Od2*(-np.cos(zc2)/zc2**2-np.sin(zc2)/zc2)
        A[0,4] = -Oc2*(np.sin(zd2)/zd2**2-np.cos(zd2)/zd2)
        A[0,5] = -Oc2*(-np.cos(zd2)/zd2**2-np.sin(zd2)/zd2)

        A[1,0] = np.sin(zc1)/zc1**2-np.cos(zc1)/zc1
        A[1,1] = np.sin(zd1)/zd1**2-np.cos(zd1)/zd1
        A[1,2] = -(np.sin(zc2)/zc2**2-np.cos(zc2)/zc2)
        A[1,3] = -(-np.cos(zc2)/zc2**2-np.sin(zc2)/zc2)
        A[1,4] = -(np.sin(zd2)/zd2**2-np.cos(zd2)/zd2)
        A[1,5] = -(-np.cos(zd2)/zd2**2-np.sin(zd2)/zd2)

        A[2,0] = H_in*Od1*dj1_zc1_dr+2*L_in*Od1*j1_zc1/r+2*alpha_in*D_in*(dj1_zc1_dr+2/r*j1_zc1)
        A[2,1] = H_in*Oc1*dj1_zd1_dr+2*L_in*Oc1*j1_zd1/r+2*alpha_in*D_in*(dj1_zd1_dr+2/r*j1_zd1)
        A[2,2] = -(H_out*Od2*dj1_zc2_dr+2*L_out*Od2*j1_zc2/r+2*alpha_out*D_out*(dj1_zc2_dr+2/r*j1_zc2))
        A[2,3] = -(H_out*Od2*dn1_zc2_dr+2*L_out*Od2*n1_zc2/r+2*alpha_out*D_out*(dn1_zc2_dr+2/r*n1_zc2))
        A[2,4] = -(H_out*Oc2*dj1_zd2_dr+2*L_out*Oc2*j1_zd2/r+2*alpha_out*D_out*(dj1_zd2_dr+2/r*j1_zd2))
        A[2,5] = -(H_out*Oc2*dn1_zd2_dr+2*L_out*Oc2*n1_zd2/r+2*alpha_out*D_out*(dn1_zd2_dr+2/r*n1_zd2))

        A[3,0] = -2*alpha_in*D_in*(Od1*dj1_zc1_dr+2/r*Od1*j1_zc1)-2*D_in*(dj1_zc1_dr+2/r*j1_zc1)
        A[3,1] = -2*alpha_in*D_in*(Oc1*dj1_zd1_dr+2/r*Oc1*j1_zd1)-2*D_in*(dj1_zd1_dr+2/r*j1_zd1)
        A[3,2] = -(-2*alpha_out*D_out*(Od2*dj1_zc2_dr+2/r*Od2*j1_zc2)-2*D_out*(dj1_zc2_dr+2/r*j1_zc2))
        A[3,3] = -(-2*alpha_out*D_out*(Od2*dn1_zc2_dr+2/r*Od2*n1_zc2)-2*D_out*(dn1_zc2_dr+2/r*n1_zc2))
        A[3,4] = -(-2*alpha_out*D_out*(Oc2*dj1_zd2_dr+2/r*Oc2*j1_zd2)-2*D_out*(dj1_zd2_dr+2/r*j1_zd2))
        A[3,5] = -(-2*alpha_out*D_out*(Oc2*dn1_zd2_dr+2/r*Oc2*n1_zd2)-2*D_out*(dn1_zd2_dr+2/r*n1_zd2))

        r = b
        zc2=kc2*r
        zd2=kd2*r

        j1_zc2     = np.sin(zc2)/zc2**2-np.cos(zc2)/zc2
        dj1_zc2_dr = kc2*((np.cos(zc2)*zc2**2-np.sin(zc2)*2*zc2)/zc2**4-(-np.sin(zc2)*zc2-np.cos(zc2))/zc2**2)
        j1_zd2     = np.sin(zd2)/zd2**2-np.cos(zd2)/zd2
        dj1_zd2_dr = kd2*((np.cos(zd2)*zd2**2-np.sin(zd2)*2*zd2)/zd2**4-(-np.sin(zd2)*zd2-np.cos(zd2))/zd2**2)
        n1_zc2     = -np.cos(zc2)/zc2**2-np.sin(zc2)/zc2
        dn1_zc2_dr = kc2*((np.sin(zc2)*zc2**2+np.cos(zc2)*2*zc2)/zc2**4-(np.cos(zc2)*zc2-np.sin(zc2))/zc2**2)
        n1_zd2     = -np.cos(zd2)/zd2**2-np.sin(zd2)/zd2
        dn1_zd2_dr = kd2*((np.sin(zd2)*zd2**2+np.cos(zd2)*2*zd2)/zd2**4-(np.cos(zd2)*zd2-np.sin(zd2))/zd2**2)

        A[4,0] = 0
        A[4,1] = 0
        A[4,2] = H_out*Od2*dj1_zc2_dr+2*L_out*Od2*j1_zc2/r+2*alpha_out*D_out*(dj1_zc2_dr+2/r*j1_zc2)
        A[4,3] = H_out*Od2*dn1_zc2_dr+2*L_out*Od2*n1_zc2/r+2*alpha_out*D_out*(dn1_zc2_dr+2/r*n1_zc2)
        A[4,4] = H_out*Oc2*dj1_zd2_dr+2*L_out*Oc2*j1_zd2/r+2*alpha_out*D_out*(dj1_zd2_dr+2/r*j1_zd2)
        A[4,5] = H_out*Oc2*dn1_zd2_dr+2*L_out*Oc2*n1_zd2/r+2*alpha_out*D_out*(dn1_zd2_dr+2/r*n1_zd2)

        # closed boundary
        A[5,0] = 0
        A[5,1] = 0
        A[5,2] = np.sin(zc2)/zc2**2-np.cos(zc2)/zc2
        A[5,3] = -np.cos(zc2)/zc2**2-np.sin(zc2)/zc2
        A[5,4] = np.sin(zd2)/zd2**2-np.cos(zd2)/zd2
        A[5,5] = -np.cos(zd2)/zd2**2-np.sin(zd2)/zd2

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
        b3 = X[1] 
        b5 = X[2] 
        b6 = X[3] 
        b7 = X[4] 
        b8 = X[5] 

        # Bulk modulus
        r   = b
        zc2 = kc2*r    
        zd2 = kd2*r
        wc2 = b5*(np.sin(zc2)/zc2**2-np.cos(zc2)/zc2)+b6*(-np.cos(zc2)/zc2**2-np.sin(zc2)/zc2)
        wd2 = b7*(np.sin(zd2)/zd2**2-np.cos(zd2)/zd2)+b8*(-np.cos(zd2)/zd2**2-np.sin(zd2)/zd2)
        u2  = Od2*wc2+Oc2*wd2
        K[i]= -Pc*b/(3*u2)

    # fluid pressure
    '''
    
    '''

    return K
