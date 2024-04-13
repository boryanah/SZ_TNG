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

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/fit_profile.yaml'
DEFAULTS['path2config_fixed'] = None

class SampleFileUtil(object):
    """
    Util for handling sample files.
    Copied from Andrina's code.

    :param filePrefix: the prefix to use
    :param reuseBurnin: True if the burn in data from a previous run should be used
    """

    def __init__(self, filePrefix, carry_on=False):
        self.filePrefix = filePrefix
        if carry_on:
            mode = 'a'
        else:
            mode = 'w'
        self.samplesFile = open(self.filePrefix + '.txt', mode)
        self.probFile = open(self.filePrefix + 'prob.txt', mode)

    def persistSamplingValues(self, pos, prob):
        self.persistValues(self.samplesFile, self.probFile, pos, prob)

    def persistValues(self, posFile, probFile, pos, prob):
        """
        Writes the walker positions and the likelihood to the disk
        """
        posFile.write("\n".join(["\t".join([str(q) for q in p]) for p in pos]))
        posFile.write("\n")
        posFile.flush()

        probFile.write("\n".join([str(p) for p in prob]))
        probFile.write("\n")
        probFile.flush();

    def close(self):
        self.samplesFile.close()
        self.probFile.close()

    def __str__(self, *args, **kwargs):
        return "SampleFileUtil"

class DumPool(object):
    def __init__(self):
        pass

    def is_master(self):
        return True

    def close(self):
        pass

def time_lnprob(params, param_mapping, param_profile, Da, Th, fixed_profs=None):
    print('   ==========================================')
    print("   | Calculating likelihood evaluation time |")
    print('   ==========================================')

    # run once without timing since numba needs to compile
    lnprob(params[:, 0], params, param_mapping, param_profile, Da, Th)

    timing = np.zeros(10)
    for i in range(10):
        print('Test ',i,' of 9')
        start = time.time()
        if i<5:
            lnprob(params[:, 0]+i*0.1*params[:, 3], params, param_mapping, param_profile, Da, Th)
        else:
            lnprob(params[:, 0]-(i-4)*0.1*params[:, 3], params, param_mapping, param_profile, Da, Th)
        finish = time.time()
        timing[i] = finish-start

    mean = np.mean(timing)
    print('============================================================================')
    print('mean computation time: ', mean)
    stdev = np.std(timing)
    print('standard deviation : ', stdev)
    print('============================================================================')
    return

def inrange(p, params):
    return np.all((p<=params[:, 2]) & (p>=params[:, 1]))

def lnprob(p, params, param_mapping, param_profile, Da, Th, fixed_profs=None):
    if inrange(p, params):
        param_dict = {}
        # read the parameters 
        for key in param_mapping.keys():
            mapping_idx = param_mapping[key]
            profile_type = param_profile[key]
            #tracer_type = param_profile[params[mapping_idx, -1]]
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
            if fixed_profs is not None:
                exec(f"fn = Th.get_{prof_type}_fixed", globals(), lcls)
                fn = lcls["fn"]
                prof_th = fn(Da.redshift_snap, param_dict[prof_type], fixed_profs)
                profs_th[prof_type] = prof_th
            else:
                exec(f"fn = Th.get_{prof_type}", globals(), lcls)
                fn = lcls["fn"]
                prof_th = fn(Da.redshift_snap, param_dict[prof_type])
                profs_th[prof_type] = prof_th
        lnP = Da.compute_likelihood(profs_th)
        if np.isnan(lnP):
            lnP = -np.inf
    else:
        lnP = -np.inf
    return lnP

def get_chi2(p, params, param_mapping, param_profile, Da, Th, fixed_profs=None):
    lnP = lnprob(p, params, param_mapping, param_profile, Da, Th, fixed_profs)
    return -2.*lnP

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
        print("p, chi2", p, chi2)
    else:
        # walkers ratio, number of params and burn in iterations
        marg_outfile = os.path.join(dir_chains, (ch_config_params['chainsPrefix']+".txt"))
        lnprob_outfile = os.path.join(dir_chains, (ch_config_params['chainsPrefix']+"prob.txt"))
        
        # read in bestfit (names of parameters from fit_params)
        lnprob = np.loadtxt(lnprob_outfile)
        marg = np.loadtxt(marg_outfile)
        index_max = np.argmax(lnprob)
        p = marg[index_max]
        print("max", lnprob[index_max])

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

    # read data parameters
    newData = Data(data_params, cosmo_params)
    
    # create a new abacushod object and load the subsamples
    newTheory = Theory(cosmo_params, common_params, newData.xbinc, newData.mbins, newData.sbins)

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
    
    # Make path to output
    os.makedirs(os.path.expanduser(ch_config_params['path2output']), exist_ok=True)

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
    
    # MPI option
    if ch_config_params['use_mpi']:
        from schwimmbad import MPIPool
        pool = MPIPool()
        print("Using MPI")
        pool_use = pool
    else:
        pool = DumPool()
        print("Not using MPI")
        pool_use = None

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    # Time the likelihood calculation
    if time_likelihood:
        time_lnprob(params, param_mapping, param_profile, newData, newTheory, fixed_profs)
        return

    # emcee parameters
    nwalkers = nparams * ch_config_params['walkersRatio']
    nsteps = ch_config_params['burninIterations'] + ch_config_params['sampleIterations']

    # where to record
    prefix_chain = os.path.join(os.path.expanduser(ch_config_params['path2output']),
                                ch_config_params['chainsPrefix'])

    # fix initial conditions
    found_file = os.path.isfile(prefix_chain+'.txt')
    if (not found_file) or (not ch_config_params['rerun']):
        p_initial = params[:, 0] + np.random.normal(size=(nwalkers, nparams)) * params[:, 3][None, :]
        nsteps_use = nsteps
    else:
        print("Restarting from a previous run")
        old_chain = np.loadtxt(prefix_chain+'.txt')
        p_initial = old_chain[-nwalkers:,:]
        nsteps_use = max(nsteps-len(old_chain) // nwalkers, 0)
        
    if minimize:
        from scipy.optimize import minimize

        # run minimizer
        p0 = params[:, 0]
        res = minimize(get_chi2, p0, args=(params, param_mapping, param_profile, newData, newTheory, fixed_profs), method='Nelder-Mead', options={'maxiter': 40000})
        print("result", res)
        chi2 = res['fun']
        p = res['x']

        # save output
        np.savez(f'{prefix_chain}_bestfit.npz', p=p, chi2=chi2)
        return
    
    # initializing sampler
    chain_file = SampleFileUtil(prefix_chain, carry_on=ch_config_params['rerun'])
    sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(params, param_mapping, param_profile, newData, newTheory, fixed_profs), pool=pool_use)
    start = time.time()
    print("Running %d samples" % nsteps_use)

    # record every iteration
    counter = 1
    for pos, prob, _ in sampler.sample(p_initial, iterations=nsteps_use):
        if pool.is_master():
            print('Iteration done. Persisting.')
            chain_file.persistSamplingValues(pos, prob)

            if counter % 10:
                print(f"Finished sample {counter}")
        counter += 1

    pool.close()
    end = time.time()
    print("Took ",(end - start)," seconds")


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
