#! /usr/bin/env python

import argparse
import os
import sys
import time

import emcee
import numpy as np
import yaml
from scipy.interpolate import interp1d
from scipy.integrate import quad
from likelihood import Data
from theory_class import Theory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/fit_profile.yaml'

# constants cgs
gamma = 5/3.
k_B = 1.3807e-16 # erg/T
m_p = 1.6726e-24 # g
unit_c = 1.023**2*1.e10
sigma_T = 6.6524587158e-29*1.e2**2 # cm^2
m_e = 9.10938356e-28 # g
c = 29979245800. # cm/s
const = k_B*sigma_T/(m_e*c**2) # cm^2/K
kpc_to_cm = ((1.*u.kpc).to(u.cm)).value # cm
solar_mass = 1.989e33 # g


def integrate_sph(rbinc, prof, rmax):
    """
    rbinc kpc
    """
    prof_f = interp1d(rbinc, prof, bounds_error=False, fill_value=0.)
    integrand = lambda r: prof_f(r)*4.*np.pi*r**2
    integral = quad(integrand, 0, rmax, epsabs=0.0, epsrel=1.e-4, limit=10000)[0]
    return integral

def integrate_cyl(rbinc, prof, rmax, l_bound):
    """
    rbinc lbound kpc
    """
    prof_f = interp1d(rbinc, prof, bounds_error=False, fill_value=0.)
    integral_R = np.zeros(len(rbinc))
    for j in range(len(rbinc)):
        integrand_R = lambda l: 2.*prof_f(np.sqrt(l**2. + rbinc[j]**2))*2.*np.pi*rbinc[j]
        integral_R[j] = quad(integrand_R, 0., l_bound, epsabs=0.0, epsrel=1.e-4, limit=10000)[0]
    integral_R_f = interp1d(rbinc, integral_R, bounds_error=False, fill_value=0.)
    integral = quad(integral_R_f, 0., rmax, epsabs=0.0, epsrel=1.e-4, limit=10000)[0]
    return integral

def main(path2config, time_likelihood):

    # load the yaml parameters 
    config = yaml.load(open(path2config))
    cosmo_params = config['cosmo_params']
    common_params = config['common_params']
    data_params = config['data_params']
    ch_config_params = config['ch_config_params']
    fit_params = config['fit_params']

    # snapshots
    snapshots = data_params['snapshots']

    # read data parameters
    Da = Data(data_params, cosmo_params, return_integral_quantities=True)
    
    # create a new abacushod object and load the subsamples
    Th = Theory(cosmo_params, common_params, Da.xbinc, Da.mbins, Da.sbins)

    # walkers ratio, number of params and burn in iterations
    dir_chains = ch_config_params['path2output']
    marg_outfile = os.path.join(dir_chains, (ch_config_params['chainsPrefix']+".txt"))
    lnprob_outfile = os.path.join(dir_chains, (ch_config_params['chainsPrefix']+"prob.txt"))
    
    # read in bestfit (names of parameters from fit_params)
    lnprob = np.loadtxt(lnprob_outfile)
    marg = np.loadtxt(marg_outfile)
    index_max = np.argmax(lnprob)
    p = marg[index_max]
    print("max", lnprob[index_max])
    
    # parameters to fit
    nparams = len(fit_params.keys())
    param_mapping = {}
    param_profile = {}
    for key in fit_params.keys():
        mapping_idx = fit_params[key][0]
        profile_type = fit_params[key][-1]
        param_mapping[key] = mapping_idx
        param_profile[key] = profile_type

    param_dict = {}
    for key in param_mapping.keys():
        mapping_idx = param_mapping[key]
        profile_type = param_profile[key]
        try:
            param_dict[profile_type][key] = p[mapping_idx]
        except:
            param_dict[profile_type] = {}
            param_dict[profile_type][key] = p[mapping_idx]
            
    # pass them to the mock dictionary
    profs_th = {}
    for prof_type in Da.profs_type:
        lcls = locals()
        exec(f"fn = Th.get_{prof_type}", globals(), lcls)
        fn = lcls["fn"]
        prof_th = fn(Da.redshift_snap, param_dict[prof_type])
        profs_th[prof_type] = prof_th
        
        # loop over each snapshot
        for i in range(len(snapshots)):
            # which snapshot
            snap = snapshots[i]
            
            # initiate figure
            skip = 5
            inds_mbins = np.arange(len(Da.mbins)-1)[::skip]
            inds_sbins = np.arange(len(Da.sbins)-1)
            plt.subplots(1, len(inds_mbins), figsize=(18, 5))

            # loop over each mass bin (or maybe just 5?)
            count = 0
            for j in inds_mbins:
                # loop over secondary property
                for k in inds_sbins:
                    
                    # theory profile
                    prof_theo = profs_th[prof_type][snap][(j*len(inds_sbins)+k)*len(Da.xbinc): (j*len(inds_sbins)+k+1)*len(Da.xbinc)]

                    # data profile
                    prof_data = Da.profs_data[prof_type][snap][(j*len(inds_sbins)+k)*len(Da.xbinc): (j*len(inds_sbins)+k+1)*len(Da.xbinc)]
                    #prof_data_err = 1./(np.sqrt(np.diag(Da.profs_icov[prof_type][snap][(j*len(inds_sbins)+k)*len(Da.xbinc): (j*len(inds_sbins)+k+1)*len(Da.xbinc)]))) # update
                
                    plt.subplot(1, len(inds_mbins), count+1)

                    plt.plot(Da.xbinc, prof_data*Da.xbinc**2, c='r')
                    plt.plot(Da.xbinc, prof_theo*Da.xbinc**2, c='r', ls='--')
                    
                    plt.xscale('log')
                count += 1
            plt.savefig(f"prof{prof_type}_snap{snap:d}.png")

    if Da.return_integral_quantities:
        # loop over each snapshot
        for i in range(len(snapshots)):
            # which snapshot
            snap = snapshots[i]
            
            # initiate figure
            inds_mbins = np.arange(len(Da.mbins)-1)
            inds_sbins = np.arange(len(Da.sbins)-1)
            
            # loop over integrated quantities
            for key in sz_dict[snap].keys():

                # only keep the sphericals
                if "cyl" in key: continue
                sz_type.split("_sph")[0]

                # initialize arrays
                sz_int_sph = np.zeros(len(inds_mbins))
                sz_int_cyl = np.zeros(len(inds_mbins))
                sz_dat_sph = np.zeros(len(inds_mbins))
                sz_dat_cyl = np.zeros(len(inds_mbins))            

                # loop over secondary property
                for k in inds_sbins:
                    # loop over each mass bin
                    for j in inds_mbins:

                        if sz_type[0] == "Y":
                            # P_e in units of erg/cm^3; when multiplied by sigmaT/mec2, we get 1/cm
                            # when multiplied by rbinc^3/1000**2 gives Mpc^2 kpc/cm so we multiply by kpc_to_cm
                            norm = sigma_T/(m_e*c**2)*kpc_to_cm/(1000.**3) # Mpc^2 (final)
                            prof_type = 'P_e'
                        elif sz_type[:3] == "tau":
                            # n_e in units of erg/cm^3; when multiplied by sigmaT/mec2, we get 1/cm
                            # when multiplied by rbinc^3/1000**2 gives Mpc^2 kpc/cm so we multiply by kpc_to_cm
                            norm = sigma_T*kpc_to_cm/(1000.**3) # Mpc^2 (final)
                            prof_type = 'n_e'
                        elif sz_type[0] == "b":
                            # n_e in units of erg/cm^3; when multiplied by sigmaT/mec2, we get 1/cm
                            # when multiplied by rbinc^3/1000**2 gives Mpc^2 kpc/cm so we multiply by kpc_to_cm
                            norm = sigma_T*kpc_to_cm/(1000.**3)*c # Mpc^2 (final)
                            # we are dividing by v/c the data tuks # double check  units cm/s and km/s should be of groupvel and isn't so go fix
                            prof_type = 'n_e'
                            
                        # compute theoretical profile
                        prof_theo = profs_th[prof_type][snap][(j*len(inds_sbins)+k)*len(Da.xbinc): (j*len(inds_sbins)+k+1)*len(Da.xbinc)]

                        # compute data profile
                        prof_data = Da.profs_data[prof_type][snap][(j*len(inds_sbins)+k)*len(Da.xbinc): (j*len(inds_sbins)+k+1)*len(Da.xbinc)]

                        # convert to kpc
                        rbinc = Da.xbinc*Th.r200c[j]/Da.cosmo_params['h']*1000. # kpc? tuks

                        # integral bounds
                        rmax = Th.r200c[j] # kpc
                        lbound = 500000./2./Da.cosmo_params['h']*1000. # kpc

                        # integrate profiles
                        sz_int_cyl[j] = integrate_cyl(rbinc, prof_theo, rmax, l_bound)*norm
                        sz_int_sph[j] = integrate_sph(rbinc, prof_theo, rmax)*norm

                        # just gather all the data measurements in one place
                        sz_dat_cyl[j] = sz_dict[snap][f"{sz_type}_cyl"][(j*len(inds_sbins)+k)]
                        sz_dat_sph[j] = sz_dict[snap][f"{sz_type}_sph"][(j*len(inds_sbins)+k)]

                # plot each secondary bin
                plt.subplots(1, len(inds_sbins), figsize=(18, 5))
                        
                count = 0
                # loop over secondary property
                for k in inds_sbins:
                
                    plt.subplot(1, len(inds_sbins), count+1)

                    plt.plot(Th.m200c, sz_int_cyl, c='r')
                    plt.plot(Th.m200c, sz_dat_cyl, c='r', ls='--')
                    plt.plot(Th.m200c, sz_int_sph, c='b')
                    plt.plot(Th.m200c, sz_dat_sph, c='b', ls='--')
                    
                    plt.xscale('log')
                    plt.yscale('log')
                    count += 1
            plt.savefig(f"sz_int_{sz_type}_snap{snap:d}.png")

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    parser.add_argument('--time_likelihood', dest='time_likelihood',  help='Times the likelihood calculations', action='store_true')

    args = vars(parser.parse_args())
    main(**args)
