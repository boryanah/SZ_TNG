#! /usr/bin/env python

import argparse
import os
import sys
import time

import emcee
import numpy as np
import yaml
from likelihood import Data
from theory_class import Theory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/fit_profile.yaml'

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
    Da = Data(data_params, cosmo_params)
    
    # create a new abacushod object and load the subsamples
    Th = Theory(cosmo_params, common_params, Da.xbinc, Da.mbins)

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
            #print(key, param_dict[profile_type][key])
            
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
            plt.subplots(1, len(inds_mbins), figsize=(18, 5))

            # loop over each mass bin (or maybe just 5?)
            count = 0
            for j in inds_mbins:
                
                # theory profile tuks
                prof_theo = profs_th[prof_type][snap][j*len(Da.xbinc): (j+1)*len(Da.xbinc)]

                # data profile
                prof_data = Da.profs_data[prof_type][snap][j*len(Da.xbinc): (j+1)*len(Da.xbinc)]
                prof_data_err = 1./(np.sqrt(np.diag(Da.profs_icov[prof_type][snap][j*len(Da.xbinc): (j+1)*len(Da.xbinc)])))
                
                plt.subplot(1, len(inds_mbins), count+1)
                plt.plot(Da.xbinc, prof_data*Da.xbinc**2)
                plt.plot(Da.xbinc, prof_theo*Da.xbinc**2, ls='--')
                plt.xscale('log')
                count += 1
            
            plt.savefig(f"prof{prof_type}_snap{snap:d}.png")

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    parser.add_argument('--time_likelihood', dest='time_likelihood',  help='Times the likelihood calculations', action='store_true')

    args = vars(parser.parse_args())
    main(**args)
