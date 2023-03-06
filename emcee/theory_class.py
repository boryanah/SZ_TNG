import numpy as np

from classy_sz import Class


class Theory(object):
    """
    Dummy object for calculating the theory
    """
    def __init__(self, cosmo_params, common_params, x_200c, m_bins, s_bins):
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
            
        # set up constants
        self.X_H = 0.76
        self.m_amu = 1.66e-24 # g
        self.k_B = 1.3807e-16 # erg/T
        self.Mpc_to_cm = 3.086e24 # cm
        self.Msun = 1.989e33 # g
        self.eV_to_erg = 1.60218e-12 # erg
        self.rho_crit_unit = (cosmo_params['H0']/100.)**2*self.Msun/self.Mpc_to_cm**3 # g/cm^3

        # initialize class_sz
        M = Class()
        M.set(common_params)
        M.set(cosmo_params)
        M.set({
            'output': 'mPk,pressure_profile_2h,pk_b_at_z_2h',
            'pressure profile':'B12',
            'gas profile':'B16',
            'gas profile mode' : 'custom', # important to read values of parameters
            'use_xout_in_density_profile_from_enclosed_mass' : 1,
            'M_min' : 1.0e10, 
            'M_max' : 5e15,
            'n_k_pressure_profile' :80, # this is l/ls
            'n_m_pressure_profile' :50, # default: 100, decrease for faster
            'n_z_pressure_profile' :50, # default: 100, decrease for faster
            'k_min_gas_pressure_profile' : 1e-3, # l/ls hence no need for very extreme values...
            'k_max_gas_pressure_profile' : 1e2, 
            'n_k_pressure_profile' :50, # this is kl
            'k_min_gas_pressure_profile_2h' : 1e-3, # l/ls hence no need for very extreme values...
            'k_max_gas_pressure_profile_2h' : 1e2, 
            'n_k_density_profile' :50,
            'n_m_density_profile' :20, # default: 100, decrease for faster
            'n_z_density_profile' :50, # default: 100, decrease for faster
            'k_min_gas_density_profile' : 1e-3,
            'k_max_gas_density_profile' : 1e1, 
            'k_min_samp_fftw' : 1e-3,
            'k_max_samp_fftw' : 1e3,
            'N_samp_fftw' : 200,
            'hm_consistency' : 0,
            'use_fft_for_profiles_transform' : 1,
            'x_min_gas_density_fftw' : 1e-5,
            'x_max_gas_density_fftw' : 1e2,
            'x_min_gas_pressure_fftw' : 1e-5,
            'x_max_gas_pressure_fftw' : 1e2,
        })
        M.compute_class_szfast()
        self.M = M

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
    
    def get_n_e(self, redshift_snap, param_dict):
        # object attributes
        M = self.M
        m200c = self.m200c*M.h() # Msun/h
        x_200c = self.x_200c
        sbinc = self.sbinc
        rho_gas_b16 = self.rho_gas_b16
        rho2h_gas_b16 = self.rho2h_gas_b16

        # TESTING!!!!!
        """
        # let's get rid of parameters that are not part of the 1h call
        A_2h = param_dict.pop('A_A2h') 
        alpha_m_A2h = param_dict.pop('alpha_m_A2h')
        alpha_z_A2h = param_dict.pop('alpha_z_A2h')
        alpha_c_A2h = 0. #param_dict.pop('alpha_c_A2h') # TESTING!!
        
        # change name of parameter
        param_dict['mcut'] = 10.**param_dict.pop('Mcut_break')
        """

        # store as dictionary
        n_e_snap = {}
        for snap in redshift_snap.keys():
            # initialize array
            n_e_snap[snap] = np.zeros(len(m200c)*len(x_200c)*len(sbinc))

        # initialize radial array
        self.r200c = np.zeros_like(m200c)
            
        # loop over concentration values
        for j in range(len(sbinc)):
            # I think we compute here?
            # TESTING!!!!!
            """
            param_dict['cp_B16'] = sbinc[j]
            print(sbinc[j], param_dict)
            M.compute_class_sz(param_dict)
            param_dict['c_asked'] = param_dict.pop('cp_B16')
            """
            for snap in redshift_snap.keys():
                # redshift
                z = redshift_snap[snap]

                # loop over each mass bin
                for i in range(len(m200c)):
                    # radius
                    self.r200c[i] = M.get_r_delta_of_m_delta_at_z(200., m200c[i], z) # Mpc/h
                    r = x_200c*self.r200c[i]

                    # call one halo term
                    #rho_1h_gas = rho_gas_b16(x_200c, m200c[i], z, **param_dict)
                    # (M.get_rho_crit_at_z(z) * M.get_f_b()) 

                    # TESTING!!!!!!!!!!!
                    rho_1h_gas = self.get_n_e_B12(x_200c, m200c[i]/M.h(), z, **param_dict)
                    rho_1h_gas *= (M.get_rho_crit_at_z(z) * M.get_f_b()) 
                    
                    # call two halo term 
                    rho_2h_gas = rho2h_gas_b16(r, m200c[i], z)

                    # multiply by fictitious factor
                    # TESTING!!!!!!!!!!!
                    """
                    rho_2h_gas *= A_2h * (1.+sbinc[j])**alpha_c_A2h * (m200c[i]/1.e14)**alpha_m_A2h * (1.+z)**alpha_z_A2h
                    """
                    
                    # combine one and two halo
                    rho_1p2h = rho_1h_gas + rho_2h_gas
                    rho_1p2h *= self.rho_crit_unit # g/cm^3
                    rho_1p2h *= 0.5*(self.X_H + 1.)/self.m_amu # could do 200./X_H/self.m_amu # cm^-3
                    
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
        # TESTING!!!!!!!!!!!
        """
        A_2h = param_dict.pop('A2h0')
        alpha_m_A2h = param_dict.pop('alpha_m_A2h')
        alpha_c_A2h = param_dict.pop('alpha_c_A2h')
        alpha_z_A2h = param_dict.pop('alpha_z_A2h')
        
        # change name of parameter
        param_dict['mcut_B12'] = 10.**param_dict.pop('Mcut_break')
        """
        
        # store as dictionary
        P_e_snap = {}
        for snap in redshift_snap.keys():
            # initialize array
            P_e_snap[snap] = np.zeros(len(m200c)*len(x_200c)*len(sbinc))

        # initialize radial array
        self.r200c = np.zeros_like(m200c)
        
        # loop over concentration values
        for j in range(len(sbinc)):
            # I think we compute here?
            # TESTING!!!!!!!!!!!
            """
            param_dict['cp_B12'] = sbinc[j]
            M.compute_class_sz(param_dict)
            param_dict['c_asked'] = param_dict.pop('cp_B12')
            """
            
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
                    
                    # call one halo term
                    #p_e_1h_gas = p_e_gas_b12(x_200c, m200c[i], z, **param_dict) # eV/cm^3

                    # TESTING!!!!!!!!!!!
                    p_e_1h_gas = self.get_P_e_B12(x_200c, m200c[i]/M.h(), z, **param_dict)
                    p_e_1h_gas *= pe200
                    
                    # call two halo term
                    p_e_2h_gas = p_e2h_gas_b12(r, m200c[i], z) # eV/cm^3

                    # multiply by fictitious factor
                    # TESTING!!!!!!!!!!!
                    """
                    p_e_2h_gas *= A_2h * (1.+sbinc[j])**alpha_c_A2h * (m200c[i]/1.e14)**alpha_m_A2h * (1.+z)**alpha_z_A2h
                    """
                    
                    # combine one and two halo 
                    p_e_1p2h = p_e_1h_gas + p_e_2h_gas
                    p_e_1p2h *= self.eV_to_erg # erg/cm^3
                    p_e_1p2h *= ((2. + 2.*self.X_H) / (3. + 5.*self.X_H)) # could do without
                    
                    # add to array
                    P_e_snap[snap][(i*len(sbinc)+j)*len(x_200c): (i*len(sbinc)+j+1)*len(x_200c)] = p_e_1p2h
        return P_e_snap
        
    def get_rho(self, z, param_dict):
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

        # initialize radial array
        self.r200c = np.zeros_like(m200c)

        # since not computing two-halo term
        M.compute_class_sz(param_dict)
        
        # loop over snapshots
        for snap in redshift_snap.keys():
            # redshift
            z = redshift_snap[snap]

            # loop over each mass bin
            for i in range(len(m200c)):
                # radius
                self.r200c[i] = M.get_r_delta_of_m_delta_at_z(200., m200c[i], z) # Mpc/h
                r = x_200c*self.r200c[i]
                c200c = M.get_c200c_at_m_and_z_D08(m200c, z)

                # call one halo term
                rs_200c = r200c/c200c
                xs_200c =  r/rs_200c
                rho_norm_nfw = rho_gas_nfw(xs_200c, m200c, z)
        
                # call one halo term
                rho_1h_nfw = rho_gas_nfw(xs_200c, m200c[i], z, *param_dict)

                # convert into dark matter density
                rho_1h_nfw /= M.get_f_b() #(M.get_rho_crit_at_z(z)*M.get_f_b())
                    
                # add to array
                rho_dm_snap[snap][i*len(x_200c): (i+1)*len(x_200c)] = rho_1h_nfw
 
