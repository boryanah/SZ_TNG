import numpy as np

from classy_sz import Class


class Theory(object):
    """
    Dummy object for calculating the theory
    """
    def __init__(self, cosmo_params, common_params, x_200c, m_bins, s_bins, fixed_profile=False):
        """
        Constructor of the power spectrum data
        """

        # fix input (yaml cannot take spaces in the keys)
        new_keys = []
        old_keys = []
        for key in common_params.keys():
            old_keys.append(key)
            new_keys.append(" ".join(key.split("__")))
        for i, old_key in enumerate(old_keys):
            common_params[new_keys[i]] = common_params.pop(old_key)
        extra_params = {
            'output': 'pressure_profile_2h,gas_density_profile_2h',
            'pressure profile':'B12',
            'gas profile':'B16',
            'gas profile mode' : 'custom', # important to read values of parameters
            'use_xout_in_density_profile_from_enclosed_mass' : 1,
            'n_z_m_to_xout' : 30,
            'n_mass_m_to_xout' : 30,
            'M_min' : 1.0e10, 
            'M_max' : 5e15,
            'n_k_pressure_profile' :50, # this is l/ls # default 80
            'n_m_pressure_profile' :30, # default: 100, decrease for faster
            'n_z_pressure_profile' :30, # default: 100, decrease for faster
            'k_min_gas_pressure_profile' : 1e-3, # l/ls hence no need for very extreme values...
            'k_max_gas_pressure_profile' : 1e2, 
            'k_min_gas_pressure_profile_2h' : 1e-3, # l/ls hence no need for very extreme values...
            'k_max_gas_pressure_profile_2h' : 1e2, 
            'n_k_density_profile' :50, # default 80
            'n_m_density_profile' :30, # default: 100, decrease for faster
            'n_z_density_profile' :30, # default: 100, decrease for faster
            'k_min_gas_density_profile' : 1e-3,
            'k_max_gas_density_profile' : 1e2, 
            'k_min_samp_fftw' : 1e-3,
            'k_max_samp_fftw' : 1e3,
            'N_samp_fftw' : 1024,
            'hm_consistency' : 0, #1,  #0, # TESTING!!!!!!!!!!!
            'use_fft_for_profiles_transform' : 1,
            'x_min_gas_density_fftw' : 1e-5,
            'x_max_gas_density_fftw' : 1e2,
            'x_min_gas_pressure_fftw' : 1e-5,
            'x_max_gas_pressure_fftw' : 1e2,
        }
        
        # set up constants
        self.X_H = 0.76
        self.m_amu = 1.66e-24 # g
        self.k_B = 1.3807e-16 # erg/T
        self.Mpc_to_cm = 3.086e24 # cm
        self.Msun = 1.989e33 # g
        self.eV_to_erg = 1.60218e-12 # erg
        self.rho_crit_unit = (cosmo_params['H0']/100.)**2*self.Msun/self.Mpc_to_cm**3 # g/cm^3
        self.fixed_profile = fixed_profile
        if self.fixed_profile:
            self.fixed_profs = {}
        
        # initialize class_sz
        M = Class()
        M.set(common_params)
        M.set(cosmo_params)
        M.set(extra_params)
        M.compute_class_szfast()
        self.M = M

        """
        print("# common")
        for key in common_params.keys():
            print(f"{key} = {common_params[key]}")
        print("# cosmo")
        for key in cosmo_params.keys():
            print(f"{key} = {cosmo_params[key]}")
        print("# extra")
        for key in extra_params.keys():
            print(f"{key} = {extra_params[key]}")
        """
        
        self.pressure_dict = {
            'A_P0': 'P0_B12',
            'A_xc': 'xc_B12',
            'A_beta': 'beta_B12',
            
            'alpha_m_P0': 'alpha_m_P0_B12',
            'alpha_m_xc': 'alpha_m_xc_B12',
            'alpha_m_beta': 'alpha_m_beta_B12',

            'alphap_m_P0': 'alphap_m_P0_B12',
            'alphap_m_xc': 'alphap_m_xc_B12',
            'alphap_m_beta': 'alphap_m_beta_B12',

            'alpha_z_P0': 'alpha_z_P0_B12',
            'alpha_z_xc': 'alpha_z_xc_B12',
            'alpha_z_beta': 'alpha_z_beta_B12',

            'alpha_c_P0': 'alpha_c_P0_B12',
            'alpha_c_xc': 'alpha_c_xc_B12',
            'alpha_c_beta': 'alpha_c_beta_B12',

            'alpha': 'alpha_B12',
            'gamma': 'gamma_B12',
            'mcut': 'mcut_B12'
        }
        
        # store the radial profiles of the gas
        self.rho_gas_nfw = np.vectorize(M.get_gas_profile_at_x_M_z_nfw_200c)
        self.rho_gas_b16 = np.vectorize(M.get_gas_profile_at_x_M_z_b16_200c)
        self.rho2h_gas_b16 = np.vectorize(M.get_rho_2h_at_r_and_m_and_z)
        self.p_e_gas_b12 = np.vectorize(M.get_pressure_P_over_P_delta_at_x_M_z_b12_200c)
        self.p_e2h_gas_b12 = np.vectorize(M.get_gas_pressure_2h_at_r_and_m_and_z)
        self.p_e200 = np.vectorize(M.get_P_delta_at_m_and_z_b12)
        
        # normalized radial array for b16
        self.x_200c = x_200c # ratio

        # center of mass bins
        self.m200c = (m_bins[1:] + m_bins[:-1])*.5 # Msun
        self.sbinc = (s_bins[1:] + s_bins[:-1])*.5 
        
    def get_A_param(self, z, M, A0, alpha_z, alpha_m, c=0., alpha_c=0., alphap_m=0., Mcut_break=1.e14):
        if np.isclose(alphap_m, 0.):
            A = A0 * (1+z)**alpha_z * (1. + c)**alpha_c * (M/1.e14)**alpha_m
        else:
            if M > Mcut_break:
                A = A0 * (1+z)**alpha_z * (1. + c)**alpha_c * (M/Mcut_break)**alpha_m
            else:
                A = A0 * (1+z)**alpha_z * (1. + c)**alpha_c * (M/Mcut_break)**alphap_m
        return A

    def get_P_e_B12(self, x, M, z, A_P0=0., A_beta=0., A_xc=0., alpha_m_P0=0., alpha_m_xc=0., alpha_m_beta=0., alpha_z_P0=0., alpha_z_xc=0., alpha_z_beta=0., alpha=1, gamma=-0.3):
        P0 = self.get_A_param(z, M, A_P0, alpha_z_P0, alpha_m_P0)
        xc = self.get_A_param(z, M, A_xc, alpha_z_xc, alpha_m_xc)
        beta = self.get_A_param(z, M, A_beta, alpha_z_beta, alpha_m_beta)
        P_e = P0*(x/xc)**gamma*(1.+(x/xc)**alpha)**(-beta)
        return P_e

    def get_n_e_B12(self, x, M, z, A_rho0=0., A_alpha=0., A_beta=0., alpha_m_rho0=0., alpha_m_alpha=0., alpha_m_beta=0., alpha_z_rho0=0., alpha_z_alpha=0., alpha_z_beta=0., xc=0.5, gamma=-0.2):
        rho0 = self.get_A_param(z, M, A_rho0, alpha_z_rho0, alpha_m_rho0)
        alpha = self.get_A_param(z, M, A_alpha, alpha_z_alpha, alpha_m_alpha)
        beta = self.get_A_param(z, M, A_beta, alpha_z_beta, alpha_m_beta)
        n_e = rho0*(x/xc)**gamma*(1.+(x/xc)**alpha)**(-beta)
        return n_e

    def get_nfw(self, x, rho0=0., xc=0., alpha=0., beta=0.):
        if np.isclose(beta, 0): beta = -alpha
        rho = rho0/((x/xc)**(1.+alpha)*(1. + x/xc)**(2.+beta))
        return rho
    
    def get_n_e(self, redshift_snap, param_dict):
        # object attributes
        M = self.M
        m200c = self.m200c*M.h() # Msun/h
        x_200c = self.x_200c
        sbinc = self.sbinc
        rho_gas_b16 = self.rho_gas_b16
        rho2h_gas_b16 = self.rho2h_gas_b16
        
        # let's get rid of parameters that are not part of the 1h call
        A_2h = param_dict.pop('A_A2h', 1.) 
        alpha_m_A2h = param_dict.pop('alpha_m_A2h', 0.)
        alpha_z_A2h = param_dict.pop('alpha_z_A2h', 0.)
        alpha_c_A2h = param_dict.pop('alpha_c_A2h', 0.)
        
        # change name of parameter
        param_dict['mcut'] = 10.**param_dict.pop('Mcut_break', 14.)

        # store as dictionary
        n_e_snap = {}
        if self.fixed_profile:
            n_e_snap_1h = {}
            n_e_snap_2h = {}
        for snap in redshift_snap.keys():
            # initialize array
            n_e_snap[snap] = np.zeros(len(m200c)*len(x_200c)*len(sbinc))
            if self.fixed_profile:
                n_e_snap_1h[snap] = np.zeros(len(m200c)*len(x_200c)*len(sbinc))
                n_e_snap_2h[snap] = np.zeros(len(m200c)*len(x_200c)*len(sbinc))

        # initialize radial array
        self.r200c = np.zeros_like(m200c)
            
        # loop over concentration values
        for j in range(len(sbinc)):
            # compute
            param_dict['cp_B16'] = sbinc[j]
            if self.fixed_profile:
                M.compute_class_sz(param_dict) # definitely compute in this case
            else:
                pass #M.compute_class_sz(param_dict) # don't need if not fitting
            param_dict['c_asked'] = param_dict.pop('cp_B16', 0.)
            for snap in redshift_snap.keys():
                # redshift
                z = redshift_snap[snap]

                # loop over each mass bin
                for i in range(len(m200c)):
                    # radius
                    self.r200c[i] = M.get_r_delta_of_m_delta_at_z(200., m200c[i], z) # Mpc/h
                    r = x_200c*self.r200c[i]

                    # call one halo term
                    rho_1h_gas = rho_gas_b16(x_200c, m200c[i], z, **param_dict)
                    # already multiplied by (M.get_rho_crit_at_z(z) * M.get_f_b()) 
                    # equivalent
                    #rho_1h_gas = self.get_n_e_B12(x_200c, m200c[i]/M.h(), z, **param_dict)
                    #rho_1h_gas *= (M.get_rho_crit_at_z(z) * M.get_f_b())  # needs

                    # call two halo term
                    rho_2h_gas = rho2h_gas_b16(r, m200c[i], z)

                    # multiply by fictitious factor
                    rho_2h_gas *= A_2h * (1.+sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/param_dict['mcut'])**alpha_m_A2h * (1.+z)**alpha_z_A2h
                    
                    # combine one and two halo
                    rho_1p2h = rho_1h_gas + rho_2h_gas
                    rho_1p2h *= self.rho_crit_unit # g/cm^3
                    rho_1p2h *= 0.5*(self.X_H + 1.)/self.m_amu # ask; could do 200./X_H/self.m_amu # cm^-3
                    
                    # add to array
                    n_e_snap[snap][(i*len(sbinc)+j)*len(x_200c): (i*len(sbinc)+j+1)*len(x_200c)] = rho_1p2h
                    if self.fixed_profile:
                        n_e_snap_1h[snap][(i*len(sbinc)+j)*len(x_200c): (i*len(sbinc)+j+1)*len(x_200c)] = rho_1h_gas
                        n_e_snap_2h[snap][(i*len(sbinc)+j)*len(x_200c): (i*len(sbinc)+j+1)*len(x_200c)] = rho_2h_gas
        if self.fixed_profile:
            self.fixed_profs['n_e_1h'] = n_e_snap_1h
            self.fixed_profs['n_e_2h'] = n_e_snap_2h
        return n_e_snap


    def get_n_e_fixed(self, redshift_snap, param_dict, fixed_profs):
        # object attributes
        M = self.M
        m200c = self.m200c*M.h() # Msun/h
        x_200c = self.x_200c
        sbinc = self.sbinc
        rho_gas_b16 = self.rho_gas_b16
        rho2h_gas_b16 = self.rho2h_gas_b16
        
        # let's get rid of parameters that are not part of the 1h call
        A_2h = param_dict.pop('A_A2h', 1.) 
        alpha_m_A2h = param_dict.pop('alpha_m_A2h', 0.)
        alphap_m_A2h = param_dict.pop('alphap_m_A2h', 0.)
        alpha_z_A2h = param_dict.pop('alpha_z_A2h', 0.)
        alpha_c_A2h = param_dict.pop('alpha_c_A2h', 0.)
        
        # change name of parameter
        param_dict['mcut'] = 10.**param_dict.pop('Mcut_break', 14.)

        # store as dictionary
        n_e_snap = {}
        for snap in redshift_snap.keys():
            # initialize array
            n_e_snap[snap] = np.zeros(len(m200c)*len(x_200c)*len(sbinc))
            
        # loop over concentration values
        for j in range(len(sbinc)):
            for snap in redshift_snap.keys():
                # redshift
                z = redshift_snap[snap]

                # loop over each mass bin
                for i in range(len(m200c)):
                    # call one halo term
                    rho_1h_gas = fixed_profs['n_e_1h'][snap][(i*len(sbinc)+j)*len(x_200c): (i*len(sbinc)+j+1)*len(x_200c)]
                    
                    # call two halo term 
                    rho_2h_gas = fixed_profs['n_e_2h'][snap][(i*len(sbinc)+j)*len(x_200c): (i*len(sbinc)+j+1)*len(x_200c)]

                    # multiply by fictitious factor
                    #rho_2h_gas *= A_2h * (1.+sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/param_dict['mcut'])**alpha_m_A2h * (1.+z)**alpha_z_A2h
                    if np.isclose(alphap_m_A2h, 0.):
                        #f = A_2h * (1.+sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/1.e14)**alpha_m_A2h * (1.+z)**alpha_z_A2h # og
                        #f = A_2h * (1.+sbinc[j]*alpha_c_A2h) * (m200c[i]/M.h()/1.e14)**alpha_m_A2h * (1.+z)**alpha_z_A2h # TESTING
                        f = A_2h * (1.-sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/1.e14)**alpha_m_A2h * (1.+z)**alpha_z_A2h # minus high is bigger gap than low
                    else:
                        if m200c[i]/M.h() > param_dict['mcut']:
                            #f = A_2h * (1.+sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/param_dict['mcut'])**alpha_m_A2h * (1.+z)**alpha_z_A2h # og
                            #f = A_2h * (1.+sbinc[j]*alpha_c_A2h) * (m200c[i]/M.h()/param_dict['mcut'])**alpha_m_A2h * (1.+z)**alpha_z_A2h # TESTING
                            f = A_2h * (1.-sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/param_dict['mcut'])**alpha_m_A2h * (1.+z)**alpha_z_A2h 
                        else:
                            #f = A_2h * (1.+sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/param_dict['mcut'])**alphap_m_A2h * (1.+z)**alpha_z_A2h # og
                            #f = A_2h * (1.+sbinc[j]*alpha_c_A2h) * (m200c[i]/M.h()/param_dict['mcut'])**alphap_m_A2h * (1.+z)**alpha_z_A2h # TESTING
                            f = A_2h * (1.-sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/param_dict['mcut'])**alphap_m_A2h * (1.+z)**alpha_z_A2h
                    
                    # combine one and two halo
                    rho_1p2h = rho_1h_gas + rho_2h_gas*f
                    rho_1p2h *= self.rho_crit_unit # g/cm^3
                    rho_1p2h *= 0.5*(self.X_H + 1.)/self.m_amu # ask; could do 200./X_H/self.m_amu # cm^-3
                    
                    # add to array
                    n_e_snap[snap][(i*len(sbinc)+j)*len(x_200c): (i*len(sbinc)+j+1)*len(x_200c)] = rho_1p2h
        return n_e_snap
    
    def get_P_e(self, redshift_snap, param_dict):
        # object attributes
        M = self.M
        m200c = self.m200c*M.h() # Msun/h
        x_200c = self.x_200c
        sbinc = self.sbinc
        p_e_gas_b12 = self.p_e_gas_b12
        p_e2h_gas_b12 = self.p_e2h_gas_b12

        # let's get rid of parameters that are not part of the 1h call
        A_2h = param_dict.pop('A2h0', 1.)
        alpha_m_A2h = param_dict.pop('alpha_m_A2h', 0.)
        alpha_c_A2h = param_dict.pop('alpha_c_A2h', 0.)
        alpha_z_A2h = param_dict.pop('alpha_z_A2h', 0.)
        
        # change name of parameter
        param_dict['mcut'] = 10.**param_dict.pop('Mcut_break', 14.)
        
        # store as dictionary
        P_e_snap = {}
        if self.fixed_profile:
            P_e_snap_1h = {}
            P_e_snap_2h = {}
            
        for snap in redshift_snap.keys():
            # initialize array
            P_e_snap[snap] = np.zeros(len(m200c)*len(x_200c)*len(sbinc))
            if self.fixed_profile:
                P_e_snap_1h[snap] = np.zeros(len(m200c)*len(x_200c)*len(sbinc))
                P_e_snap_2h[snap] = np.zeros(len(m200c)*len(x_200c)*len(sbinc))
            
        # initialize radial array
        self.r200c = np.zeros_like(m200c)

        class_dict = {}
        for key in param_dict.keys():
            class_dict[self.pressure_dict[key]] = param_dict[key]
        
        # loop over concentration values
        for j in range(len(sbinc)):
            # compute
            class_dict['cp_B12'] = sbinc[j]
            if self.fixed_profile:
                M.compute_class_sz(class_dict) # definitely compute in this case
            else:
                pass #M.compute_class_sz(class_dict) # don't need if not fitting
            param_dict['c_asked'] = class_dict.pop('cp_B12', 0.)
            
            for snap in redshift_snap.keys():
                # redshift
                z = redshift_snap[snap]
                
                # loop over each mass bin
                for i in range(len(m200c)):
                    # radius
                    self.r200c[i] = M.get_r_delta_of_m_delta_at_z(200., m200c[i], z) # Mpc/h
                    r = x_200c*self.r200c[i]

                    # normalization
                    pe200 = M.get_P_delta_at_m_and_z_b12(m200c[i], z)
                    
                    # call one halo term # og
                    p_e_1h_gas = p_e_gas_b12(x_200c, m200c[i], z, **param_dict) # eV/cm^3
                    # equivalent
                    #p_e_1h_gas = self.get_P_e_B12(x_200c, m200c[i]/M.h(), z, **param_dict)
                    p_e_1h_gas *= pe200 # eV/cm^3 (important for both)
                    
                    # call two halo term
                    p_e_2h_gas = p_e2h_gas_b12(r, m200c[i], z) # eV/cm^3
                                        
                    # multiply by fictitious factor
                    p_e_2h_gas *= A_2h * (1.+sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/param_dict['mcut'])**alpha_m_A2h * (1.+z)**alpha_z_A2h
                    
                    # combine one and two halo 
                    p_e_1p2h = p_e_1h_gas + p_e_2h_gas
                    p_e_1p2h *= self.eV_to_erg # erg/cm^3
                    p_e_1p2h *= ((2. + 2.*self.X_H) / (3. + 5.*self.X_H)) # can remove
                    
                    # add to array
                    P_e_snap[snap][(i*len(sbinc)+j)*len(x_200c): (i*len(sbinc)+j+1)*len(x_200c)] = p_e_1p2h
                    if self.fixed_profile:
                        P_e_snap_1h[snap][(i*len(sbinc)+j)*len(x_200c): (i*len(sbinc)+j+1)*len(x_200c)] = p_e_1h_gas
                        P_e_snap_2h[snap][(i*len(sbinc)+j)*len(x_200c): (i*len(sbinc)+j+1)*len(x_200c)] = p_e_2h_gas
        if self.fixed_profile:
            self.fixed_profs['P_e_1h'] = P_e_snap_1h
            self.fixed_profs['P_e_2h'] = P_e_snap_2h
        return P_e_snap


    def get_P_e_fixed(self, redshift_snap, param_dict, fixed_profs):
        # object attributes
        M = self.M
        m200c = self.m200c*M.h() # Msun/h
        x_200c = self.x_200c
        sbinc = self.sbinc
        p_e_gas_b12 = self.p_e_gas_b12
        p_e2h_gas_b12 = self.p_e2h_gas_b12

        # let's get rid of parameters that are not part of the 1h call
        A_2h = param_dict.pop('A2h0', 1.)
        alpha_m_A2h = param_dict.pop('alpha_m_A2h', 0.)
        alphap_m_A2h = param_dict.pop('alphap_m_A2h', 0.)
        alpha_c_A2h = param_dict.pop('alpha_c_A2h', 0.)
        alpha_z_A2h = param_dict.pop('alpha_z_A2h', 0.)
        
        # change name of parameter
        param_dict['mcut'] = 10.**param_dict.pop('Mcut_break', 14.)
        
        # store as dictionary
        P_e_snap = {}
        for snap in redshift_snap.keys():
            # initialize array
            P_e_snap[snap] = np.zeros(len(m200c)*len(x_200c)*len(sbinc))
            
        # loop over concentration values
        for j in range(len(sbinc)):
            for snap in redshift_snap.keys():
                # redshift
                z = redshift_snap[snap]
                print("SHOULD BE ZERO, ONE, ONE", sbinc[j], (1.-sbinc[j])**alpha_c_A2h, z);
                                
                # loop over each mass bin
                for i in range(len(m200c)):
                    # call one halo term # og
                    p_e_1h_gas = fixed_profs['P_e_1h'][snap][(i*len(sbinc)+j)*len(x_200c): (i*len(sbinc)+j+1)*len(x_200c)]
                    # call two halo term
                    p_e_2h_gas = fixed_profs['P_e_2h'][snap][(i*len(sbinc)+j)*len(x_200c): (i*len(sbinc)+j+1)*len(x_200c)]
                    #np.save("1h_zmin0.005.npy", fixed_profs['P_e_1h'][snap][:])
                    #np.save("2h_zmin0.005.npy", fixed_profs['P_e_2h'][snap][:])
                    #np.save("1h_zmin0.000.npy", fixed_profs['P_e_1h'][snap][:])
                    #np.save("2h_zmin0.000.npy", fixed_profs['P_e_2h'][snap][:])
                    #quit()
                    # TESTING!!!!!!!!!!!!! BIG SKETCH # somehow the 10^-8 difference makes a difference??????????????
                    #p_e_2h_gas = np.load("2h_zmin0.005.npy")[(i*len(sbinc)+j)*len(x_200c): (i*len(sbinc)+j+1)*len(x_200c)]
                    
                    # multiply by fictitious factor
                    if np.isclose(alphap_m_A2h, 0.):
                        #f = A_2h * (1.+sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/1.e14)**alpha_m_A2h * (1.+z)**alpha_z_A2h # og
                        #f = A_2h * (1.+sbinc[j]*alpha_c_A2h) * (m200c[i]/M.h()/1.e14)**alpha_m_A2h * (1.+z)**alpha_z_A2h # TESTING
                        #f = A_2h * (1.-sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/1.e14)**alpha_m_A2h * (1.+z)**alpha_z_A2h
                        f = A_2h * (1.-sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/10.**14.5)**alpha_m_A2h * (1.+z)**alpha_z_A2h
                    else:
                        if m200c[i]/M.h() > param_dict['mcut']:
                            #f = A_2h * (1.+sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/param_dict['mcut'])**alpha_m_A2h * (1.+z)**alpha_z_A2h # og
                            #f = A_2h * (1.+sbinc[j]*alpha_c_A2h) * (m200c[i]/M.h()/param_dict['mcut'])**alpha_m_A2h * (1.+z)**alpha_z_A2h # TESTING
                            f = A_2h * (1.-sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/param_dict['mcut'])**alpha_m_A2h * (1.+z)**alpha_z_A2h
                        else:
                            #f = A_2h * (1.+sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/param_dict['mcut'])**alphap_m_A2h * (1.+z)**alpha_z_A2h # og
                            #f = A_2h * (1.+sbinc[j]*alpha_c_A2h) * (m200c[i]/M.h()/param_dict['mcut'])**alphap_m_A2h * (1.+z)**alpha_z_A2h # TESTING
                            f = A_2h * (1.-sbinc[j])**alpha_c_A2h * (m200c[i]/M.h()/param_dict['mcut'])**alphap_m_A2h * (1.+z)**alpha_z_A2h 
                    
                    # combine one and two halo 
                    p_e_1p2h = p_e_1h_gas + p_e_2h_gas*f
                    p_e_1p2h *= self.eV_to_erg # erg/cm^3
                    p_e_1p2h *= ((2. + 2.*self.X_H) / (3. + 5.*self.X_H)) # can remove
                    
                    # add to array
                    P_e_snap[snap][(i*len(sbinc)+j)*len(x_200c): (i*len(sbinc)+j+1)*len(x_200c)] = p_e_1p2h

        return P_e_snap
        
    def get_rho(self, redshift_snap, param_dict):

        # object attributes
        M = self.M
        m200c = self.m200c*M.h() # Msun/h
        x_200c = self.x_200c
        rho_gas_nfw = self.rho_gas_nfw
        
        # store as dictionary
        rho_dm_snap = {}
        for snap in redshift_snap.keys():
            # initialize array
            rho_dm_snap[snap] = np.zeros(len(m200c)*len(x_200c))

        # since not computing two-halo term
        #M.compute_class_sz(param_dict)
            
        # initialize radial array
        self.r200c = np.zeros_like(m200c)
        
        # loop over snapshots
        for snap in redshift_snap.keys():
            # redshift
            z = redshift_snap[snap]

            # loop over each mass bin
            for i in range(len(m200c)):
                # radius (checked)
                self.r200c[i] = M.get_r_delta_of_m_delta_at_z(200., m200c[i], z) # Mpc/h
                r = x_200c*self.r200c[i]
                c200c = M.get_c200c_at_m_and_z_D08(m200c[i] , z)
                rs_200c = self.r200c[i]/c200c 
                xs_200c =  r/rs_200c
                
                # call one halo term
                #rho_1h_nfw = rho_gas_nfw(xs_200c, m200c[i], z)
                # already multiplied by (M.get_rho_crit_at_z(z)*M.get_f_b())
                # alternative
                #rho_1h_nfw = self.get_n_e_B12(x_200c, m200c[i]/M.h(), z, **param_dict)
                rho_1h_nfw = self.get_nfw(x_200c, **param_dict)
                #rho_1h_nfw *= (M.get_rho_crit_at_z(z) * M.get_f_b())  # needs if buba's
                rho_1h_nfw *= (M.get_rho_crit_at_z(z) * 200.)  # needs 200 rhocrit
                rho_1h_nfw *= self.rho_crit_unit # g/cm^3
                
                # add to array
                rho_dm_snap[snap][i*len(x_200c): (i+1)*len(x_200c)] = rho_1h_nfw
        return rho_dm_snap
