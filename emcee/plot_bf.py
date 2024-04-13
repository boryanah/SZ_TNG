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
import astropy.units as u
from likelihood import Data
from theory_class import Theory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/fit_profile.yaml'
DEFAULTS['path2config_fixed'] = None

"""
python plot_bf.py --path2config config/mtng_fit_vanilla_density.yaml
python plot_bf.py --path2config config/mtng_fit_conc_density.yaml
python plot_bf.py --path2config config/mtng_fit_conc_power_density.yaml
python plot_bf.py --path2config config/mtng_fit_power_density.yaml

python plot_bf.py --path2config config/mtng_fit_vanilla_pressure.yaml
python plot_bf.py --path2config config/mtng_fit_conc_pressure.yaml
python plot_bf.py --path2config config/mtng_fit_conc_power_pressure.yaml
python plot_bf.py --path2config config/mtng_fit_power_pressure.yaml
"""

# constants cgs
gamma = 5/3.
k_B = 1.3807e-16 # erg/T
m_p = 1.6726e-24 # g
unit_c = 1.023**2*1.e10
sigma_T = 6.6524587158e-29*1.e2**2 # cm^2
m_e = 9.10938356e-28 # g
c = 29979245800 # cm/s
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

def get_norm_prof_type(sz_type):
    if sz_type[0] == "Y":
        # P_e in units of erg/cm^3; when multiplied by sigmaT/mec2, we get 1/cm
        # when multiplied by rbinc^3/1000**2 gives Mpc^2 kpc/cm so we multiply by kpc_to_cm
        norm = sigma_T/(m_e*c**2)*kpc_to_cm/(1000.**2) # Mpc^2 (final)
        prof_type = 'P_e'
    elif sz_type[:3] == "tau":
        # n_e in units of erg/cm^3; when multiplied by sigmaT/mec2, we get 1/cm
        # when multiplied by rbinc^3/1000**2 gives Mpc^2 kpc/cm so we multiply by kpc_to_cm
        norm = sigma_T*kpc_to_cm/(1000.**2) # Mpc^2 (final)
        prof_type = 'n_e'
    elif sz_type[0] == "b":
        # n_e in units of erg/cm^3; when multiplied by sigmaT/mec2, we get 1/cm
        # when multiplied by rbinc^3/1000**2 gives Mpc^2 kpc/cm so we multiply by kpc_to_cm
        norm = sigma_T*kpc_to_cm/(1000.**2) # Mpc^2 (final)
        prof_type = 'n_e'
    return norm, prof_type

def load_1p2h(path2config, r_min, r_max, nbin_sec, secondary_type, load_fixed_mini=False):
    # load the yaml parameters
    config = yaml.load(open(path2config))
    cosmo_params = config['cosmo_params']
    common_params = config['common_params']
    data_params = config['data_params']
    ch_config_params = config['ch_config_params']
    fit_params = config['fit_params']

    # change so as to match (tuks think about secondary parameter)
    data_params['r_min'] = r_min
    data_params['r_max'] = r_max
    data_params['nbin_sec'] = nbin_sec
    data_params['secondary_type'] = secondary_type

    # where are the chains saved
    dir_chains = ch_config_params['path2output']

    # read data parameters
    Da = Data(data_params, cosmo_params)
    
    # create a new abacushod object and load the subsamples
    Th = Theory(cosmo_params, common_params, Da.xbinc, Da.mbins, Da.sbins, fixed_profile=True)

    # parameters to fit
    nparams = len(fit_params.keys())
    param_mapping = {}
    param_profile = {}
    params = np.zeros((nparams, 4))
    for key in fit_params.keys():
        mapping_idx = fit_params[key][0]
        profile_type = fit_params[key][-1]
        param_mapping[key] = mapping_idx
        param_profile[key] = profile_type
        params[mapping_idx, :] = fit_params[key][1:-1]

    if load_fixed_mini:
        # where to record
        mini_outfile = os.path.join(dir_chains, f"{ch_config_params['chainsPrefix']}_bestfit.npz")
        data = np.load(mini_outfile)
        p = data['p']
        chi2 = data['chi2']
        print("fixed: p_bf, chi2 from txt", p, chi2)
    else:
        # walkers ratio, number of params and burn in iterations
        marg_outfile = os.path.join(dir_chains, (ch_config_params['chainsPrefix']+".txt"))
        lnprob_outfile = os.path.join(dir_chains, (ch_config_params['chainsPrefix']+"prob.txt"))
        
        # read in bestfit (names of parameters from fit_params)
        lnprob = np.loadtxt(lnprob_outfile)
        marg = np.loadtxt(marg_outfile)
        index_max = np.argmax(lnprob)
        p = marg[index_max]
        print("fixed: min chi2 from txt", -2.*lnprob[index_max])

    param_dict = {}
    for key in param_mapping.keys():
        mapping_idx = param_mapping[key]
        profile_type = param_profile[key]
        try:
            param_dict[profile_type][key] = p[mapping_idx]
        except:
            param_dict[profile_type] = {}
            param_dict[profile_type][key] = p[mapping_idx]

    # pass them to the dictionary
    for prof_type in Da.profs_type:
        lcls = locals()
        exec(f"fn = Th.get_{prof_type}", globals(), lcls)
        fn = lcls["fn"]
        prof_th = fn(Da.redshift_snap, param_dict[prof_type])

    return Th.fixed_profs


def main(path2config, time_likelihood, minimize, fixed_profile, path2config_fixed, load_fixed_mini):

    # load the yaml parameters
    config = yaml.load(open(path2config))
    cosmo_params = config['cosmo_params']
    common_params = config['common_params']
    data_params = config['data_params']
    ch_config_params = config['ch_config_params']
    fit_params = config['fit_params']
    other_params = config['other_params']

    # TESTING!!!!!!!!!!!!!!!!!!!
    r_min = 0.03
    r_max = 100.
    data_params['r_min'] = r_min
    data_params['r_max'] = r_max
    
    # chain name
    dir_chains = ch_config_params['path2output']
    chainsPrefix = ch_config_params['chainsPrefix']
    
    # snapshots
    snapshots = data_params['snapshots']

    # here we are just going to decide on which profiles to load
    #data_params['profs_type'] = ['n_e', 'P_e'] # TESTING!!!!!!!!!!!! # if you comment out, just default
    profs_type_plot = data_params['profs_type']
    
    # read data parameters
    Da = Data(data_params, cosmo_params, return_integral_quantities=True)
    
    # create a new abacushod object and load the subsamples
    Th = Theory(cosmo_params, common_params, Da.xbinc, Da.mbins, Da.sbins)

    if minimize:
        # where to record
        mini_outfile = os.path.join(dir_chains, f"{ch_config_params['chainsPrefix']}_bestfit.npz")
        data = np.load(mini_outfile)
        p = data['p']
        chi2 = data['chi2']
        print("p, chi2 from txt", p, chi2)
    else:
        # walkers ratio, number of params and burn in iterations
        marg_outfile = os.path.join(dir_chains, (ch_config_params['chainsPrefix']+".txt"))
        lnprob_outfile = os.path.join(dir_chains, (ch_config_params['chainsPrefix']+"prob.txt"))
        
        # read in bestfit (names of parameters from fit_params)
        lnprob = np.loadtxt(lnprob_outfile)
        marg = np.loadtxt(marg_outfile)
        index_max = np.argmax(lnprob)
        p = marg[index_max]
        print("min chi2 from txt", -2.*lnprob[index_max])
    
    # parameters to fit
    nparams = len(fit_params.keys())
    param_mapping = {}
    param_profile = {}
    for key in fit_params.keys():
        mapping_idx = fit_params[key][0]
        profile_type = fit_params[key][-1]
        param_mapping[key] = mapping_idx
        param_profile[key] = profile_type


    # tuks for loading
    if fixed_profile:
        assert path2config_fixed is not None
        # create the 1- and 2-halo
        r_min = data_params.get('r_min', 0.0)
        r_max = data_params.get('r_max', 100.)
        secondary_type = data_params.get('secondary_type', 'conc')
        nbin_sec = data_params['nbin_sec']
        fixed_profs = load_1p2h(path2config_fixed, r_min, r_max, nbin_sec, secondary_type, load_fixed_mini) 
    else:
        fixed_profs = None
    #print("SHOULDNT BE NONE", fixed_profs)
    
    param_dict = {}
    for key in param_mapping.keys():
        mapping_idx = param_mapping[key]
        profile_type = param_profile[key]
        try:
            param_dict[profile_type][key] = p[mapping_idx]
        except:
            param_dict[profile_type] = {}
            param_dict[profile_type][key] = p[mapping_idx]

    # print bestfit parameters
    #print("bf params", param_dict)
    
    # load in other parameters needed for other profiles
    for prof_type_plot in profs_type_plot:
        if prof_type_plot in param_dict.keys(): continue
        param_dict[prof_type_plot] = other_params # TODO could do it a bit better (to make sure it matches profile)
        
    # pass them to the dictionary
    profs_th = {}
    for prof_type in profs_type_plot:
        lcls = locals()
        if fixed_profs is not None:
            exec(f"fn = Th.get_{prof_type}_fixed", globals(), lcls)
            fn = lcls["fn"]
            prof_th = fn(Da.redshift_snap, param_dict[prof_type], fixed_profs)
        else:
            exec(f"fn = Th.get_{prof_type}", globals(), lcls)
            fn = lcls["fn"]
            prof_th = fn(Da.redshift_snap, param_dict[prof_type])
        profs_th[prof_type] = prof_th
    sz_dict = Da.sz_dict

    # colors
    hexcolors_bright = ['#CC3311','#0077BB','#EE7733','limegreen','#BBBBBB','#33BBEE','#EE3377','#0099BB']
    cs = hexcolors_bright
    
    # this only plots the yaml file stuff
    plot_bf = True
    if plot_bf:
        # pass them to the mock dictionary
        for prof_type in Da.profs_type:

            chi2_sum = 0.
            dof_sum = 0
            # loop over each snapshot
            for i in range(len(snapshots)):
                # which snapshot
                snap = snapshots[i]
                
                # initiate figure
                if len(Da.mbins)-1 > 5:
                    skip = 1 #3
                    n_row = 3 #1
                    figsize = (18, 13)
                else:
                    skip = 1
                    n_row = 1
                    figsize = (18, 5)
                inds_mbins = np.arange(len(Da.mbins)-1)[::skip]
                inds_sbins = np.arange(len(Da.sbins)-1)
                assert len(inds_mbins) % n_row == 0
                plt.subplots(n_row, len(inds_mbins)//n_row, figsize=figsize)

                # loop over each mass bin (or maybe just 5?)
                count = 0
                for j in inds_mbins:
                    
                    # loop over secondary property
                    for k in inds_sbins:

                        # theory profile
                        prof_theo = profs_th[prof_type][snap][(j*len(inds_sbins)+k)*len(Da.xbinc): (j*len(inds_sbins)+k+1)*len(Da.xbinc)]

                        # data profile
                        prof_data = Da.profs_data[prof_type][snap][(j*len(inds_sbins)+k)*len(Da.xbinc): (j*len(inds_sbins)+k+1)*len(Da.xbinc)]
                        prof_data_err = Da.profs_icov[prof_type][snap][(j*len(inds_sbins)+k)*len(Da.xbinc): (j*len(inds_sbins)+k+1)*len(Da.xbinc), (j*len(inds_sbins)+k)*len(Da.xbinc): (j*len(inds_sbins)+k+1)*len(Da.xbinc)]
                        delta = (prof_theo-prof_data)
                        chi2 = np.einsum('i,ij,j', delta, prof_data_err, delta)
                        chi2_sum += chi2
                        dof_sum += len(delta)
                        print("chi2, dof", chi2, len(delta), f"{np.log10(Da.mbins[j]):.2f} < log(M_h) < {np.log10(Da.mbins[j+1]):.2f}, snap, prof type", snap, prof_type)
                        try:
                            prof_data_err = np.sqrt(np.diag(np.linalg.inv(prof_data_err)))
                            yerr = prof_data_err*Da.xbinc**2
                        except:
                            print("singular matrix")
                            yerr = None

                        plt.subplot(n_row, len(inds_mbins)//n_row, count+1)

                        plt.title(rf"${np.log10(Da.mbins[j]):.2f} < \log (M_h) < {np.log10(Da.mbins[j+1]):.2f}$")
                        plt.errorbar(Da.xbinc, prof_data*Da.xbinc**2, yerr=yerr, c=cs[k])
                        plt.plot(Da.xbinc, prof_theo*Da.xbinc**2, c=cs[k], ls='--')

                        plt.xscale('log')
                        plt.yscale('log')
                    count += 1
                
                plt.savefig(f"figs/prof_{prof_type}_{chainsPrefix}_snap{snap:d}.png")
            print("chi2 sum, dof sum", chi2_sum, dof_sum)
    quit() # TESTING
    
    if Da.return_integral_quantities and len(profs_type_plot) > 1:
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
                if "std" in key: continue
                if "mean" not in key: continue
                print(key)
                sz_type = key

                # initialize arrays
                sz_int_sph = np.zeros(len(inds_mbins))
                sz_int_cyl = np.zeros(len(inds_mbins))
                sz_dat_sph = np.zeros(len(inds_mbins))
                sz_dat_cyl = np.zeros(len(inds_mbins))            

                # loop over secondary property
                for k in inds_sbins:
                    # loop over each mass bin
                    for j in inds_mbins:
                        # normalization for this quantity
                        norm, prof_type = get_norm_prof_type(sz_type)

                        # compute theoretical profile
                        prof_theo = profs_th[prof_type][snap][(j*len(inds_sbins)+k)*len(Da.xbinc): (j*len(inds_sbins)+k+1)*len(Da.xbinc)]

                        # compute data profile
                        prof_data = Da.profs_data[prof_type][snap][(j*len(inds_sbins)+k)*len(Da.xbinc): (j*len(inds_sbins)+k+1)*len(Da.xbinc)]

                        # convert to kpc
                        rbinc = Da.xbinc*Th.r200c[j]/(Da.cosmo_params['H0']/100.)*1000. # kpc

                        # integral bounds
                        rmax = Th.r200c[j] # kpc
                        Lbox = 500. # Mpc/h
                        l_bound = 0.5*Lbox*1000./(Da.cosmo_params['H0']/100.) # kpc

                        # integrate profiles
                        sz_int_cyl[j] = integrate_cyl(rbinc, prof_theo, rmax, l_bound)*norm
                        sz_int_sph[j] = integrate_sph(rbinc, prof_theo, rmax)*norm

                        # just gather all the data measurements in one place
                        sz_dat_cyl[j] = sz_dict[snap][f"{sz_type.split('_sph_mean')[0]}_cyl_xy_mean"][(j*len(inds_sbins)+k)]
                        sz_dat_sph[j] = sz_dict[snap][sz_type][(j*len(inds_sbins)+k)]

                # plot each secondary bin
                plt.subplots(1, len(inds_sbins), figsize=(18, 5))
                
                count = 0
                # loop over secondary property
                for k in inds_sbins:
                
                    plt.subplot(1, len(inds_sbins), count+1)
                    plt.title(rf"${Da.sbins[k]:.2f} < s < {Da.sbins[k+1]:.2f}$")
                    
                    plt.plot(Th.m200c, sz_int_cyl, c='r')
                    plt.plot(Th.m200c, sz_dat_cyl, c='r', ls='--')
                    plt.plot(Th.m200c, sz_int_sph, c='b')
                    plt.plot(Th.m200c, sz_dat_sph, c='b', ls='--')
                    
                    plt.xscale('log')
                    plt.yscale('log')
                    count += 1
                plt.savefig(f"figs/sz_int_{sz_type}_{chainsPrefix}_snap{snap:d}.png")

    plot_extra = True
    if plot_extra:
        # loop over each snapshot
        for i in range(len(snapshots)):
            # which snapshot
            snap = snapshots[i]
            
            # initiate figure
            inds_mbins = np.arange(len(Da.mbins)-1)
            inds_sbins = np.arange(len(Da.sbins)-1)

            # which pairs to include 
            plot_pairs = [['m200c', 'Y_200c_sph_mean', 0], ['m200c', 'Y_200c_cyl_xy_mean', 0], ['tau_200c_sph_mean', 'Y_200c_sph_mean', 1], ['tau_200c_sph_mean', 'Y_200c_cyl_xy_mean', 1], ['tau_200c_cyl_xy_mean', 'b_200c_sph_mean', 2], ['tau_200c_cyl_xy_mean', 'b_200c_cyl_xy_mean', 2], ['tau_200c_sph_mean', 'b_200c_sph_mean', 3], ['tau_200c_sph_mean', 'b_200c_cyl_xy_mean', 3]]
            label_pairs = [[r'$M_{\rm 200c}$', r'$Y_{\rm 200c}$'], [r'$\tau_{\rm 200c, sph}$', r'$Y_{\rm 200c}$'], [r'$\tau_{\rm 200c, cyl}$', r'$b_{\rm 200c}$'], [r'$\tau_{\rm 200c, sph}$', r'$b_{\rm 200c}$']]
            legends = ['sph', 'cyl', 'sph', 'cyl', 'sph', 'cyl', 'sph', 'cyl']
            
            # initialize plot
            plt.subplots(1, plot_pairs[-1][2]+1, figsize=(18, 5))
            
            # loop over integrated quantities
            count_prev = -1
            for i_pair, plot_pair in enumerate(plot_pairs):

                # initialize arrays
                quant_x = np.zeros(len(inds_mbins))
                quant_y = np.zeros(len(inds_mbins))
                pair_x = plot_pair[0]
                pair_y = plot_pair[1]
                count = plot_pair[2]
                if count == count_prev:
                    count_color += 1
                else:
                    count_color = 0
                
                # loop over each mass bin
                for j in inds_mbins:
                    # loop over secondary property
                    for k in inds_sbins:
                        if pair_x == "m200c":
                            quant_x[j] += Th.m200c[j]
                        else:
                            quant_x[j] += sz_dict[snap][pair_x][(j*len(inds_sbins)+k)]
                        quant_y[j] += sz_dict[snap][pair_y][(j*len(inds_sbins)+k)]
                    quant_x[j] /= len(inds_sbins)
                    quant_y[j] /= len(inds_sbins)
                                    
                plt.subplot(1, plot_pairs[-1][2]+1, count+1)
                plt.plot(quant_x, quant_y, c=cs[count_color], label=legends[i_pair])

                plt.ylabel(label_pairs[count][1])
                plt.xlabel(label_pairs[count][0])
                plt.legend()
                
                plt.xscale('log')
                plt.yscale('log')

                count_prev = count
                    
            plt.savefig(f"figs/sz_quant_{chainsPrefix}_snap{snap:d}.png")

            
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    parser.add_argument('--time_likelihood', dest='time_likelihood',  help='Times the likelihood calculations', action='store_true')
    parser.add_argument('--minimize', dest='minimize',  help='Run a minimizer instead of full emcee chain', action='store_true')
    parser.add_argument('--fixed_profile', dest='fixed_profile',  help='Fix the 1-halo term and vary other parameters', action='store_true')
    parser.add_argument('--path2config_fixed', dest='path2config_fixed', type=str, help='Path to config file with fixed params.', default=DEFAULTS['path2config_fixed'])
    parser.add_argument('--load_fixed_mini', dest='load_fixed_mini',  help='Load fixed parameters from minimizer (otherwise, from chain)', action='store_true')

    args = vars(parser.parse_args())
    main(**args)
