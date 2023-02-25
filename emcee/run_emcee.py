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

def time_lnprob(params, param_mapping, param_profile, Da, Th):
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

def lnprob(p, params, param_mapping, param_profile, Da, Th):
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
            exec(f"fn = Th.get_{prof_type}", globals(), lcls)
            fn = lcls["fn"]
            prof_th = fn(Da.redshift_snap, param_dict[prof_type])
            #profs_th = np.hstack((profs_th, prof_th))
            profs_th[prof_type] = prof_th
        lnP = Da.compute_likelihood(profs_th)
    else:
        lnP = -np.inf
    return lnP


def main(path2config, time_likelihood):

    # load the yaml parameters # tuks
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

    # parameters to fit # tuks read from blah
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
        time_lnprob(params, param_mapping, param_profile, newData, newTheory)
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

    # initializing sampler
    chain_file = SampleFileUtil(prefix_chain, carry_on=ch_config_params['rerun'])
    sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(params, param_mapping, param_profile, newData, newTheory), pool=pool_use)
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

    args = vars(parser.parse_args())
    main(**args)
