import gc
from pathlib import Path

import numpy as np


class Data(object):
    """
    Dummy object for calculating a likelihood
    """
    def __init__(self, data_params, cosmo_params, return_integral_quantities=False):
        """
        Constructor of the power spectrum data
        """
        # read relevant parameters
        snapshots = data_params['snapshots']
        data_dir = Path(data_params['data_dir'])
        field_dir = Path(data_params['field_dir'])
        fn_base = data_params['fn_base']
        snaps_fn = data_params['snaps_fn']
        profs_type = data_params['profs_type'] # for kSZ just take V com of halo
        sim_name = data_params['sim_name'] # units kpc vs Mpc
        self.profs_type = profs_type
        self.cosmo_params = cosmo_params
        assert len(profs_type) == 1, "More than one type of profile not implemented and doesn't make sense to fit simultaneously anyways"

        # define mass bins
        logmmin = data_params['logmmin']
        logmmax = data_params['logmmax']
        nbin = data_params['nbin_m']
        nbin_sec = data_params['nbin_sec']
        offset = 0.5 # TODO
        s_bins = np.linspace(-offset, offset, nbin_sec+1)
        m_bins = np.logspace(logmmin, logmmax, nbin+1)
        self.return_integral_quantities = return_integral_quantities
        
        # create dictionary of snapshots
        snaps, _, zs, _ = np.loadtxt(snaps_fn, skiprows=1, unpack=True)
        snaps_dic = dict(zip(snaps.astype(int), zs))

        # initialize dictionary with data and covariance
        profs_data = {}
        profs_icov = {}
        profs_mask = {}
        mbins_snap = {}
        xbinc_snap = {}
        redshift_snap = {}
        for prof_type in self.profs_type:
            prof_data = {}
            prof_icov = {}
            prof_mask = {}

            if self.return_integral_quantities:
                sz_dict = {}
                for snapshot in snapshots:
                    sz_dict[snapshot] = {}
            
            for snapshot in snapshots:
                # select redshift
                redshift = snaps_dic[snapshot]
                a = 1./(1+redshift)
                
                # load profiles
                data = np.load(data_dir / f"{fn_base}_snap_{snapshot:d}.npz")
                halo_inds = data['inds_halo']
                n_e = data['n_e']/data['V_d']
                T_e = data['T_e']/data['N_v']
                P_e = data['P_e']/data['V_d']
                if "rho" == prof_type:
                    data = np.load(data_dir / f"{dm_base}_snap_{snapshot:d}.npz")
                    rho = data['rho_dm']/data['V_d']
                    rho[data['V_d'] == 0.] = 0.
                    rho /= a**3.
                    prof = rho
                    
                # where there are no voxels, the profiles are zero
                n_e[data['V_d'] == 0.] = 0.
                T_e[data['N_v'] == 0.] = 0.
                P_e[data['V_d'] == 0.] = 0.
                
                # old cgs units
                n_e /= a**3.
                P_e /= a**3.

                # choose profile type
                if "n_e" == prof_type:
                    prof = n_e
                elif "T_e" == prof_type:
                    prof = T_e
                elif "P_e" == prof_type:
                    prof = P_e
                    
                # load secondary properties
                GroupVel = np.load(field_dir / f"GroupVel_fp_{snapshot:d}.npy")/a # km/s
                GroupConc = np.load(field_dir / f"GroupConc_fp_{snapshot:d}.npy")
                GroupShearAdapt = np.load(field_dir / f"GroupShearAdapt_fp_{snapshot:d}.npy")
                Group_M_Crit200 = np.load(field_dir / f"Group_M_Crit200_fp_{snapshot:d}.npy")*1.e10 # Msun/h
                Group_R_Crit200 = np.load(field_dir / f"Group_R_Crit200_fp_{snapshot:d}.npy") # Mpc/h
                halo_conc = GroupConc[halo_inds]
                halo_m200 = Group_M_Crit200[halo_inds]/self.cosmo_params['h'] # Msun
                halo_r200 = Group_R_Crit200[halo_inds]
                halo_shear = GroupShearAdapt[halo_inds]
                halo_vel = GroupVel[halo_inds]
                del Group_R_Crit200, Group_M_Crit200, GroupShearAdapt, GroupConc, GroupVel; gc.collect()


                if self.return_integral_quantities:
                    data_sz = np.load(data_dir / f"{sz_base}_snap_{snapshot:d}.npz")
                    sz_dict[snapshot]['Y_200c_sph'] = data['Y_200c_sph']
                    sz_dict[snapshot]['Y_200c_cyl_xy'] = data['Y_200c_cyl_xy']
                    sz_dict[snapshot]['b_200c_sph'] = data['b_200c_sph']
                    sz_dict[snapshot]['b_200c_cyl_xy'] = data['b_200c_cyl_xy']
                    sz_dict[snapshot]['b_200c_sph'] = np.sqrt(sz_dict[snapshot]['b_200c_sph'][:, 0]**2+sz_dict[snapshot]['b_200c_sph'][:, 1]**2+sz_dict[snapshot]['b_200c_sph'][:, 2]**2)
                    sz_dict[snapshot]['b_200c_cyl_xy'] /= halo_vel[:, 2]
                    sz_dict[snapshot]['b_200c_sph'] /= np.sqrt(halo_vel[:, 0]**2+halo_vel[:, 1]**2+halo_vel[:, 2]**2)
                    
                    sz_dict[snapshot]['tau_200c_sph'] = data['tau_200c_sph']
                    sz_dict[snapshot]['tau_200c_cyl_xy'] = data['tau_200c_cyl_xy']
                    assert np.sum(halo_inds - data['inds_halo']) == 0

                
                # select secondary halo property
                halo_secondary = halo_conc # TODO

                # mass bins
                mbins_snap[snapshot] = m_bins
                redshift_snap[snapshot] = redshift

                # radial bins
                r_bins = data['rbins'] # ratio to r200c
                r_binc = (r_bins[1:]+r_bins[:-1])*.5
                x_choice = r_binc > 0.01 # TODO
                r_binc = r_binc[x_choice]
                xbinc_snap[snapshot] = r_binc

                # initialize for this snapshot
                prof_data[snapshot] = np.zeros((len(m_bins)-1)*len(r_binc)*nbin_sec)
                prof_icov[snapshot] = np.zeros(((len(m_bins)-1)*len(r_binc)*nbin_sec, (len(m_bins)-1)*len(r_binc)*nbin_sec))
                prof_mask[snapshot] = np.ones((len(m_bins)-1)*len(r_binc)*nbin_sec, dtype=bool)
                if self.return_integral_quantities:
                    for key in sz_dict[snapshot].keys():
                        sz_dict[snapshot][f"{key}_mean"] = np.zeros((len(m_bins)-1)*nbin_sec)
                        sz_dict[snapshot][f"{key}_std"] = np.zeros((len(m_bins)-1)*nbin_sec)
                
                # loop over each mass bin
                for i in range(len(m_bins)-1):
                    # select mass bin
                    m_choice = (m_bins[i] < halo_m200) & (m_bins[i+1] >= halo_m200)
                    print("halos in mass bin", i, np.sum(m_choice))
                    
                    # secondary property in mass bin
                    m_sec = halo_secondary[m_choice]
                    rank_sec = (np.argsort(np.argsort(m_sec))+1)/len(m_sec)
                    rank_sec -= offset # -0.5 to 0.5

                    # loop over each secondary dependence bin
                    for j in range(nbin_sec):
                        c_choice = (rank_sec+offset <= (j+1)/nbin_sec) & (rank_sec+offset > j/nbin_sec)
                        print("halos in secondary bin", j, np.sum(c_choice))
                        
                        # if not enough, skip (but should have a way of skipping)
                        if np.sum(c_choice) < len(r_binc):
                            prof_mask[snapshot][(i*nbin_sec+j)*len(r_binc): (i*nbin_sec+j+1)*len(r_binc)] = False
                            continue
                        
                        # compute the means 
                        prof_mean = np.mean(prof[m_choice][c_choice][: , x_choice], axis=0)

                        # compute the covariance
                        prof_invc = np.linalg.inv(np.cov(prof[m_choice][c_choice][: , x_choice].T))
                        
                        # record
                        prof_data[snapshot][(i*nbin_sec+j)*len(r_binc): (i*nbin_sec+j+1)*len(r_binc)] = prof_mean
                        prof_icov[snapshot][(i*nbin_sec+j)*len(r_binc): (i*nbin_sec+j+1)*len(r_binc), (i*nbin_sec+j)*len(r_binc): (i*nbin_sec+j+1)*len(r_binc)] = prof_invc
                        # save
                        if self.return_integral_quantities:
                            for key in sz_dict[snapshot].keys():
                                sz_dict[snapshot][f"{key}_mean"][(i*nbin_sec+j)] = np.mean(sz_dict[snapshot][key][m_choice][c_choice])
                                sz_dict[snapshot][f"{key}_std"][(i*nbin_sec+j)] = np.std(sz_dict[snapshot][key][m_choice][c_choice])
                                del sz_dict[snapshot][key]

            # within the larger dictionary
            profs_data[prof_type] = prof_data
            profs_icov[prof_type] = prof_icov
            profs_mask[prof_type] = prof_mask
            
        # attr of object
        self.profs_data = profs_data
        self.profs_icov = profs_icov
        self.profs_mask = profs_mask
        self.xbinc_snap = xbinc_snap
        self.mbins_snap = mbins_snap
        self.redshift_snap = redshift_snap

        # since it's shared between all (just being lazy)
        self.xbinc = r_binc
        self.mbins = m_bins
        self.sbins = s_bins
        
    def compute_likelihood(self, profs_theory):
        """
        Computes the likelihood using information from the context
        """
        # Calculate a likelihood up to normalization
        lnprob = 0.
        dof = 0
        for prof_type in self.profs_data.keys():
            for snap in self.profs_data[prof_type].keys():
                
                delta = (self.profs_data[prof_type][snap] - profs_theory[prof_type][snap]).flatten()
                icov = self.profs_icov[prof_type][snap]
                mask = self.profs_mask[prof_type][snap]

                lnprob_snap = np.einsum('i,ij,j', delta[mask], icov[mask, mask], delta[mask])
                #lnprob_snap = np.sum(delta[mask]**2*inv_err2[mask])
                
                lnprob += lnprob_snap
                dof += np.sum(mask)
        lnprob *= -0.5

        # Return the likelihood
        print(" <><> Likelihood evaluated, lnprob = ", lnprob, dof)
        return lnprob
    
