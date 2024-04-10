# Eve's code
# two things that don't make sense: if i == 0; and then for i in len(ras)
# tau and y

# improvements:
# are the std's used? (could be used for fitting)
# periodic bc (could improve things a bit)
# how to quantify error on prediction?

# realism:
# beam ор no beam? (could simulate this for realism)
# div matter? (could simulate this for realism)
# disk and not ring: leads to big difference (a couple of times)

# future:
# so goal of this would be to redo Nick's analysis for different apertures and different samples and go to lower masses
# SBPL for clusters
import sys
import gc

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pixell import enmap, utils, enplot
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.coordinates import SkyCoord

from tools import extractStamp, calc_T_AP#, get_tzav_fast


import argparse
import multiprocessing
from multiprocessing import Pool
from itertools import repeat
np.random.seed(42)

from mpi4py import MPI
"""
#mpirun -np 40 python save_stacks.py 40 EX_0 r200c;
mpirun -np 40 python save_stacks.py 40 EX_1 r200c;
mpirun -np 40 python save_stacks.py 40 EX_2 r200c;
mpirun -np 40 python save_stacks.py 40 EX_3 r200c;

mpirun -np 8 python save_stacks.py 8 67 r200c; mpirun -np 8 python save_stacks.py 8 78 r200c
mpirun -np 8 python save_stacks.py 8 67 r200m; mpirun -np 8 python save_stacks.py 8 78 r200m
mpirun -np 8 python save_stacks.py 8 67 r200t; mpirun -np 8 python save_stacks.py 8 78 r200t
"""
myrank = MPI.COMM_WORLD.Get_rank()
n_ranks = int(sys.argv[1])

k_B = 1.380649e-23 # J/K, Boltzmann constant
h_P =  6.62607015e-34 # Js, Planck's constant
T_CMB = 2.7255 # K


def f_nu(nu):
    """ f(ν) = x coth (x/2) − 4, with x = hν/kBTCMB """
    x = h_P*nu/(k_B*T_CMB)
    f = x / np.tanh(x/2.) - 4.
    return f

def eshow(x, fn, **kwargs):
    ''' Define a function to help us plot the maps neatly '''
    plots = enplot.get_plots(x, **kwargs)
    #enplot.show(plots, method = "python")
    enplot.write("figs/"+fn, plots)

def compute_aperture(mp, msk, ra, dec, Th_arcmin, r_max_arcmin, resCutoutArcmin, projCutout, measurement):
    
    # extract stamp
    _, stampMap, stampMask = extractStamp(mp, ra, dec, rApMaxArcmin=r_max_arcmin, resCutoutArcmin=resCutoutArcmin, projCutout=projCutout, pathTestFig='figs/', test=False, cmbMask=msk)
    
    # skip if mask is zero everywhere
    if (np.sum(stampMask) == 0.) or (stampMask is None): return 0., 0., 0., 0.

    # record T_AP
    dT_i, dT_o, dT_i_std, dT_o_std = calc_T_AP(stampMap, Th_arcmin, mask=stampMask, divmap=None, measurement=measurement)

    return dT_i, dT_o, dT_i_std, dT_o_std

def main():

    # parameters
    nu_ACT = 150.e9 # Hz, f150 (150 GHz) and f090 (98 GHz)
    sim_name = sys.argv[2] # "TNG300"  # EX_0  EX_1  EX_2  EX_3 TNG300
    sim_str = f"CAMELS/{sim_name}" if sim_name != "TNG300" else ""
    save_dir = f"/mnt/marvin1/boryanah/SZ_TNG/{sim_str}/"
    want_random = False
    data_dir = save_dir
    N_gal = 45000 # 30 mean 31, 45 inte 5 (random)
    if want_random:
        down = 5
        rand_str = "_rand"
    else:
        rand_str = ""
    if sim_name == "TNG300":
        field_dir = "/mnt/gosling1/boryanah/TNG300/"
        Lbox = 205000. # ckpc/h
        snapshot = int(sys.argv[2])
        if snapshot == 67:
            z = 0.5
        elif snapshot == 78:
            z = 0.3
        elif snapshot == 91:
            z = 0.1
    elif "CAMELS" in sim_str:
        field_dir = save_dir
        Lbox = 25000. # ckpc/h
        snapshot = 25; z = 0.47
        if sim_name == "EX_3":
            N_gal = 18000
        else:
            N_gal = 20000
    beam_fwhm = 2.1 # arcmin
    aperture_mode = sys.argv[3] #"r500c" "r200t" "200c" "200m" # ignores theta_arcmin
    #aperture_mode = "fixed"
    if aperture_mode == "fixed":
        theta_arcmin = 1.3 #2.1 # arcmin
    else:
        theta_arcmin = None
    orientation = "xy"
    measurement = "integrated" # "mean", "integrated"
    if measurement == "integrated":
        meas_str = "_intgr"
    else:
        meas_str = ""

    # galaxy/halo choices
    n_gal = 300/165000**3. # Nick Battaglia density in (ckpc/h)^-3
    #galaxy_choice = "star_mass"
    galaxy_choice = "halo_mass"
    #N_gal = int(np.round(n_gal*Lbox**3))
    #N_gal = 31000 # integrated, 30000 mean
    #N_gal = 300

    # pixell params
    projCutout = 'cea'
    resCutoutArcmin = 0.05 # resolution
    rApMaxArcmin = theta_arcmin

    # define cosmology
    h = 0.6774
    cosmo = FlatLambdaCDM(H0=h*100., Om0=0.3089, Tcmb0=2.725)

    # compute angular distance
    d_L = cosmo.luminosity_distance(z).to(u.Mpc).value
    d_C = d_L/(1.+z) # dC = dL/(1+z) Mpc
    d_A = d_L/(1.+z)**2 # dA = dL/(1+z)^2 # Mpc
    print("d_A [Mpc], z", z, d_A)
    d_A *= h # Mpc/h
    d_C *= h # Mpc/h
    print("comoving distance = ", d_C)

    # get size on the sky of each pixel at given redshift
    a = 1./(1+z)
    Lbox_deg = (a*Lbox/1000.)/d_A*(180./np.pi) # degrees
    print("Lbox deg = ", Lbox_deg)

    # tuks
    resCutoutMpc = resCutoutArcmin/60.*np.pi/180*d_A/h # Mpc
    print("resCutoutMpc", resCutoutMpc)
    
    # load y compton, tau and b maps
    Y = np.load(save_dir+f"Y_compton_{orientation}_snap_{snapshot:d}.npy")
    b = np.load(save_dir+f"b_{orientation}_snap_{snapshot:d}.npy")
    tau = np.load(save_dir+f"tau_{orientation}_snap_{snapshot:d}.npy")
    f_ACT = f_nu(nu_ACT)

    # cell size
    N_cell = Y.shape[0]
    cell_size = Lbox/N_cell # ckpc/h

    # fixing units
    Y *= N_cell**2
    b *= N_cell**2
    tau *= N_cell**2
    
    # box size in degrees   
    cell_size_deg = Lbox_deg/N_cell

    # create pixell map
    box = np.array([[0., 0.],[Lbox_deg, Lbox_deg]]) * utils.degree
    shape, wcs = enmap.geometry(pos=box, res=cell_size_deg * utils.degree, proj='car')
    tau_map = enmap.zeros(shape, wcs=wcs)
    y_map = enmap.zeros(shape, wcs=wcs)
    b_map = enmap.zeros(shape, wcs=wcs)
    tau_map[:] = tau
    y_map[:] = Y
    b_map[:] = b
    print("intrinsic size of pixel in arcmin", cell_size_deg*60.)
    
    # load subhalo fields
    if galaxy_choice == "star_mass":
        # load important galaxy quantities
        SubhaloPos = np.load(field_dir+f"SubhaloPos_{snapshot:d}_fp.npy") # ckpc/h
        SubhaloVel = np.load(field_dir+f"SubhaloVel_{snapshot:d}_fp.npy") # km/s
        SubhaloMst = np.load(field_dir+f"SubhaloMassType_{snapshot:d}_fp.npy")[:, 4]*1.e10 # Msun/h

        # select most stellar massive subhalos
        i_sort = (np.argsort(SubhaloMst)[::-1])[:N_gal]
        pos = SubhaloPos[i_sort]
        vel = SubhaloVel[i_sort]
        mstar = SubhaloMst[i_sort]

        del SubhaloPos, SubhaloVel, SubhaloMst; gc.collect()
    elif galaxy_choice == "halo_mass":
        # load important halo quantities
        GroupPos = np.load(field_dir+f"GroupPos_{snapshot:d}_fp.npy") # ckpc/h
        GroupVel = np.load(field_dir+f"GroupVel_{snapshot:d}_fp.npy")/a # km/s
        Group_M_Crit500 = np.load(field_dir+f"Group_M_Crit500_{snapshot:d}_fp.npy")*1.e10 # Msun/h
        Group_R_Crit500 = np.load(field_dir+f"Group_R_Crit500_{snapshot:d}_fp.npy") # ckpc/h
        #Group_M_TopHat200 = np.load(field_dir+f"Group_M_TopHat200_{snapshot:d}_fp.npy")*1.e10 # Msun/h
        #Group_R_TopHat200 = np.load(field_dir+f"Group_R_TopHat200_{snapshot:d}_fp.npy") # ckpc/h
        Group_M_TopHat200 = Group_M_Crit500 # TESTING
        Group_R_TopHat200 = Group_R_Crit500 # TESTING
        Group_M_Crit200 = np.load(field_dir+f"Group_M_Crit200_{snapshot:d}_fp.npy")*1.e10 # Msun/h
        Group_R_Crit200 = np.load(field_dir+f"Group_R_Crit200_{snapshot:d}_fp.npy") # ckpc/h
        #Group_M_Mean200 = np.load(field_dir+f"Group_M_Mean200_{snapshot:d}_fp.npy")*1.e10 # Msun/h
        #Group_R_Mean200 = np.load(field_dir+f"Group_R_Mean200_{snapshot:d}_fp.npy") # ckpc/h
        Group_M_Mean200 = Group_M_Crit200 # TESTING
        Group_R_Mean200 = Group_R_Crit200 # TESTING
        # select most stellar massive subhalos
        """
        if "r500" == aperture_mode:
            i_sort = (np.argsort(Group_M_Crit500)[::-1])[:N_gal]
        else:
            i_sort = (np.argsort(Group_M_TopHat200)[::-1])[:N_gal]
        """
        i_sort = np.arange(N_gal)
        pos = GroupPos[i_sort]
        vel = GroupVel[i_sort]
        m500c = Group_M_Crit500[i_sort]
        m200t = Group_M_TopHat200[i_sort]
        m200c = Group_M_Crit200[i_sort]
        m200m = Group_M_Mean200[i_sort]
        if "r500c" == aperture_mode:
            rhalo = Group_R_Crit500[i_sort]
        elif "r200t" == aperture_mode:
            rhalo = Group_R_TopHat200[i_sort]
        elif "r200c" == aperture_mode:
            rhalo = Group_R_Crit200[i_sort]
        elif "r200m" == aperture_mode:
            rhalo = Group_R_Mean200[i_sort]
        if "r" == aperture_mode[0]:
            theta_arcmin = (rhalo*a/1000.)/d_A*(180.*60)/np.pi
            print("min, max, mean, median theta arcmin = ", np.min(theta_arcmin), np.max(theta_arcmin), np.mean(theta_arcmin), np.median(theta_arcmin))
            rApMaxArcmin = theta_arcmin.max()

            Npix = theta_arcmin/(cell_size_deg*60.)
            print("number of pixels = ", Npix.min(), Npix.max())


        print(f"lowest halo mass = {np.min(m500c):.2e}")

        del GroupPos, GroupVel, Group_M_Crit500, Group_R_Crit500, Group_M_Crit200, Group_R_Crit200, Group_M_TopHat200, Group_R_TopHat200, Group_M_Mean200, Group_R_Mean200, rhalo; gc.collect()

    # convert comoving distance to redshift
    pos_deg = (pos*cell_size_deg/cell_size) # degrees
    DEC = pos_deg[:, 0]
    RA = pos_deg[:, 1]
    print("RA", RA.min(), RA.max())
    print("DEC", DEC.min(), DEC.max())
    sys.stdout.flush()

    if want_random:
        inds = np.arange(len(DEC))
        np.random.shuffle(inds)
        DEC = DEC[inds]
        RA = RA[::down]
        DEC = DEC[::down]
        theta_arcmin = theta_arcmin[::down]
        m500c = m500c[::down]
        m200t = m200t[::down]
        m200c = m200c[::down]
        m200m = m200m[::down]
        pos = pos[::down]
        vel = vel[::down]
        i_sort = i_sort[::down]
        N_gal //= down
    """
    hist_dec, bins_dec = np.histogram(DEC, bins=101)
    hist_ra, bins_ra = np.histogram(RA, bins=101)
    binc_dec = (bins_dec[1:] + bins_dec[:-1])*.5
    binc_ra = (bins_ra[1:] + bins_ra[:-1])*.5
    plt.plot(binc_dec, hist_dec, ls='--', label='DEC')
    plt.plot(binc_ra, hist_ra, ls='--', label='RA')

    def cart_to_eq(pos, frame='galactic'):
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        sc = SkyCoord(x, y, z, representation_type='cartesian', frame=frame)
        scg = sc.transform_to(frame='icrs')
        scg.representation_type = 'unitspherical'

        ra, dec = scg.ra.value, scg.dec.value
        return ra, dec

    # before this pos is comoving (kpc/h)
    pos[:, 2] += d_C*1000. - Lbox/2.
    chi = np.linalg.norm(pos, axis=1)
    unit_pos = pos/chi[:, None]
    theta, phi = hp.vec2ang(unit_pos)
    RA = phi*180./np.pi
    DEC = (np.pi/2. - theta)*180./np.pi
    print("RA", RA.min(), RA.max())
    print("DEC", DEC.min(), DEC.max())

    hist_dec, bins_dec = np.histogram(DEC, bins=101)
    hist_ra, bins_ra = np.histogram(RA, bins=101)
    binc_dec = (bins_dec[1:] + bins_dec[:-1])*.5
    binc_ra = (bins_ra[1:] + bins_ra[:-1])*.5
    plt.plot(binc_dec, hist_dec, ls='-', label='DEC')
    plt.plot(binc_ra, hist_ra, ls='-', label='RA')

    from nbodykit.transform import CartesianToEquatorial
    #RA, DEC = CartesianToEquatorial(pos/1000., frame='icrs')
    RA, DEC = cart_to_eq(pos/1000., frame='icrs')
    RA, DEC = np.asarray(RA), np.asarray(DEC)
    print("RA", RA.min(), RA.max())
    print("DEC", DEC.min(), DEC.max())
    hist_dec, bins_dec = np.histogram(DEC, bins=101)
    hist_ra, bins_ra = np.histogram(RA, bins=101)
    binc_dec = (bins_dec[1:] + bins_dec[:-1])*.5
    binc_ra = (bins_ra[1:] + bins_ra[:-1])*.5
    plt.plot(binc_dec, hist_dec, ls=':', label='DEC')
    plt.plot(binc_ra, hist_ra, ls=':', label='RA')
    plt.legend()
    plt.show()
    """

    # compute the aperture photometry for each galaxy
    want_plot = False
    if want_plot:
        r = 0.7 * utils.arcmin
        srcs = ([DEC*utils.degree, RA*utils.degree])
        mask = enmap.distance_from(shape, wcs, srcs, rmax=r) >= r
        """
        mask = enmap.distance_from(shape, wcs, ([DEC[0:1]*utils.degree, RA[0:1]*utils.degree]), rmax=theta_arcmin[0:1] * utils.arcmin) >= (theta_arcmin[0:1] * utils.arcmin)
        for i in range(1, len(theta_arcmin)):
            print(i)
            mask *= enmap.distance_from(shape, wcs, ([DEC[i:(i+1)]*utils.degree, RA[i:(i+1)]*utils.degree]), rmax=theta_arcmin[i:(i+1)] * utils.arcmin) >= (theta_arcmin[i:(i+1)] * utils.arcmin)
        """
        eshow(tau_map * mask, 'galaxies', **{"colorbar":True, "ticks": 5, "downgrade": 4})
        plt.close()

    # mask which in this case is just ones
    msk = tau_map.copy()
    msk *= 0.
    msk += 1.

    # split for all the ranks
    n_jump = len(RA)//n_ranks
    assert len(RA) % n_ranks == 0
    
    # for each galaxy, create a submap and select mean and so on
    tau_inns = np.zeros(n_jump)
    tau_outs = np.zeros(n_jump)
    y_inns = np.zeros(n_jump)
    y_outs = np.zeros(n_jump)
    b_inns = np.zeros(n_jump)
    b_outs = np.zeros(n_jump)
    
    start = myrank*n_jump
    for i in range(myrank*n_jump, (myrank+1)*n_jump):
        
        tau_inn, tau_out, tau_inn_std, tau_out_std = compute_aperture(tau_map, msk, RA[i], DEC[i], theta_arcmin[i], rApMaxArcmin, resCutoutArcmin, projCutout, measurement)
        tau_inns[i-start] = tau_inn
        tau_outs[i-start] = tau_out
        y_inn, y_out, y_inn_std, y_out_std = compute_aperture(y_map, msk, RA[i], DEC[i], theta_arcmin[i], rApMaxArcmin, resCutoutArcmin, projCutout, measurement)
        y_inns[i-start] = y_inn
        y_outs[i-start] = y_out
        b_inn, b_out, b_inn_std, b_out_std = compute_aperture(b_map, msk, RA[i], DEC[i], theta_arcmin[i], rApMaxArcmin, resCutoutArcmin, projCutout, measurement)
        b_inns[i-start] = b_inn
        b_outs[i-start] = b_out
        if i % 100 == 0: print("i, ra, dec, inner, outer, std = ", i, RA[i], DEC[i], tau_inn, tau_out, y_inn, y_out, b_inn, b_out)
        sys.stdout.flush()

    if measurement == "integrated":
        # tuks
        tau_inns *= resCutoutMpc**2
        tau_outs *= resCutoutMpc**2
        y_inns *= resCutoutMpc**2
        y_outs *= resCutoutMpc**2
        b_inns *= resCutoutMpc**2
        b_outs *= resCutoutMpc**2

    if "r" == aperture_mode[0]:
        fn_start = f"{data_dir}/galaxies{rand_str}_AP{aperture_mode}{meas_str}_top{N_gal:d}_{orientation}_{snapshot:d}_fp"
    elif aperture_mode == "fixed":
        fn_start = f"{data_dir}/galaxies{rand_str}_AP{theta_arcmin:.1f}{meas_str}_top{N_gal:d}_{orientation}_{snapshot:d}_fp"

    if n_ranks == 1:
        np.savez(f"{fn_start}.npz", RA=RA, DEC=DEC, m500c=m500c, m200t=m200t, m200m=m200m, m200c=m200c, pos=pos, vel=vel, tau_disks=tau_inns, tau_rings=tau_outs, y_disks=y_inns, y_rings=y_outs, b_disks=b_inns, b_rings=b_outs, halo_inds=i_sort)
    else:
        np.savez(f"{fn_start}_rank{myrank:d}_{n_ranks:d}.npz", RA=RA[myrank*n_jump: (myrank+1)*n_jump], DEC=DEC[myrank*n_jump: (myrank+1)*n_jump], m500c=m500c[myrank*n_jump: (myrank+1)*n_jump], m200t=m200t[myrank*n_jump: (myrank+1)*n_jump], m200m=m200m[myrank*n_jump: (myrank+1)*n_jump], m200c=m200c[myrank*n_jump: (myrank+1)*n_jump], pos=pos[myrank*n_jump: (myrank+1)*n_jump], vel=vel[myrank*n_jump: (myrank+1)*n_jump], tau_disks=tau_inns, tau_rings=tau_outs, y_disks=y_inns, y_rings=y_outs, b_disks=b_inns, b_rings=b_outs, halo_inds=i_sort[myrank*n_jump: (myrank+1)*n_jump])

main()
