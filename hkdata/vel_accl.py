import numpy as np
import pandas as pd
import datetime
import sys, os
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

from sotodlib import core, tod_ops
from sotodlib.io import hkdb
from sotodlib.utils.procs_pool import get_exec_env

sys.path.append('/home/ys5857/workspace/script/so_ana/')
from config import *
from util.misc import save_pkl


def lowpass_filter_tod(tod, sample_rate, cutoff_hz, order=4):
    """
    Apply a Butterworth low-pass filter to TOD.

    Parameters
    ----------
    tod : array-like
        Time-ordered data (1D array)
    sample_rate : float
        Sampling rate in Hz
    cutoff_hz : float
        Cutoff frequency of the low-pass filter in Hz
    order : int, optional
        Order of the Butterworth filter (default=4)

    Returns
    -------
    filtered_tod : ndarray
        Filtered TOD
    """

    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_hz / nyquist

    # Butterworth filter design
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Zero-phase filtering (no phase shift)
    filtered_tod = filtfilt(b, a, tod)

    return filtered_tod

def get_vel_accl(ctx, obsid, platform = 'satp3'):
    # Should be a way to get commanded vel and accl
    hkdb_file = f'/scratch/gpfs/SIMONSOBS/so/tracked/hkdb/250404/hkdb-{platform}.cfg'
    cfg = hkdb.HkConfig.from_yaml(hkdb_file)
    
    iobslist = ctx.obsdb.get(obsid)
    t0 = iobslist['timestamp'] + 120
    t1 = iobslist['timestamp'] + 240
    lspec = hkdb.LoadSpec(
        cfg=cfg, start=t0, end=t1,
        #fields=['acu.*.*'],
        fields=['acu.acu_udp_stream.Corrected_Azimuth'],
    )
    res = hkdb.load_hk(lspec)
    its, iaz = res.data['acu.acu_udp_stream.Corrected_Azimuth']
    
    sampling_interval = np.median(np.diff(its))
    azvel = np.round(np.median(np.abs(np.diff(iaz))/sampling_interval), 2)
    
    deriv2 = np.diff(np.diff(iaz))
    filtered = lowpass_filter_tod(deriv2, sample_rate=1/sampling_interval, cutoff_hz=1)
    azaccl = np.round(np.max(np.abs(filtered/sampling_interval**2))/3.7, 2) # 3.7 is approximation, not exact one.
    return azvel, azaccl

def main(obsid):
    ctx1 = core.Context(CTX_PATH1)
    platform = 'satp1'
    azvel, azacl = get_vel_accl(ctx1, obsid, platform)
    #print(azvel, azacl)
    save_pkl((azvel,azacl), f'/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/vel_accl/{platform}/{obsid}.pkl')
    return


def main_multiprocess(executor, as_completed_callable, ctx, ys, ms, ye, me, platform):
    scan_start = datetime.datetime(ys, ms, 1, 0, 0, 0, 0, tzinfo=datetime.timezone.utc)
    scan_stop = datetime.datetime(ye, me, 1, 0, 0, 0, 0, tzinfo=datetime.timezone.utc)
    obslist = ctx.obsdb.query(f'timestamp > {scan_start.timestamp()} and timestamp < {scan_stop.timestamp()}')
    obss = []
    for iobs in obslist:
        #print(iobs['obs_id'], ctx.obsdb.get(iobs['obs_id'], tags=True)['tags'])
        if ctx.obsdb.get(iobs['obs_id'], tags=True)['subtype'] == 'cmb':
            obss.append(iobs['obs_id'])
    print(f'{scan_start} - {scan_stop}: Length = {len(obss)}')

    runlist = []
    for iobsid in obss:
        irunlist = {'obsid': iobsid}
        runlist.append(irunlist)
    n_runs = len(runlist)
    print(f'number of runs: {n_runs}')
    future_to_rl = {executor.submit(main, obsid=rl['obsid']): rl for rl in runlist}
    futures = list(future_to_rl)

    n = 0
    for future in as_completed_callable(futures):
        rl = future_to_rl[future]
        try:
            n += 1
            future.result()
            futures.remove(future)
            print(f'Processing Finished correctly {n}/{n_runs}')
        except Exception as e:
            print(f'Processing {n}/{n_runs} generated an exception: {e}')
            futures.remove(future)


if __name__ == '__main__':
    # SATP3 yaml
    #CTX_PATH3 = '/so/metadata/satp3/contexts/use_this.yaml'
    #CTX3 = core.Context(CTX_PATH3)
    #CTX_PATH1 = '/so/metadata/satp1/contexts/use_this.yaml'
    #CTX1 = core.Context(CTX_PATH1)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ys', type=int)
    parser.add_argument('ms', type=int)
    parser.add_argument('ye', type=int)
    parser.add_argument('me', type=int)
    parser.add_argument('--sat', type=str, default='satp1', help='Start year for analysis')
    parser.add_argument('--nproc', type=int, default=2, help='Number of processes to use')
    args = parser.parse_args()
    rank, executor, as_completed_callable = get_exec_env(args.nproc)
    if args.sat == 'satp3':
        CTX_PATH3 = '/scratch/gpfs/SIMONSOBS/so/tracked/metadata/satp3/contexts/use_this_local.yaml'
        ctx = core.Context(CTX_PATH3)
        platform = 'satp3'
    elif args.sat == 'satp1':
        CTX_PATH1 = '/scratch/gpfs/SIMONSOBS/so/tracked/metadata/satp1/contexts/use_this_local.yaml'
        ctx = core.Context(CTX_PATH1)
        platform = 'satp1'
    else:
        raise ValueError('Platform not recognized, please use satp1 or satp3')

    if rank == 0:
        main_multiprocess(executor, as_completed_callable, ctx, args.ys, args.ms, args.ye, args.me, platform=platform)
        
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ys', type=int)
    parser.add_argument('ms', type=int)
    parser.add_argument('ye', type=int)
    parser.add_argument('me', type=int)
    parser.add_argument('--sat', type=str, default='satp1', help='Start year for analysis')
    args = parser.parse_args()
    if args.sat == 'satp3':
        CTX_PATH3 = '/scratch/gpfs/SIMONSOBS/so/tracked/metadata/satp3/contexts/use_this_local.yaml'
        ctx = core.Context(CTX_PATH3)
        platform = 'satp3'
    elif args.sat == 'satp1':
        CTX_PATH1 = '/scratch/gpfs/SIMONSOBS/so/tracked/metadata/satp1/contexts/use_this_local.yaml'
        ctx = core.Context(CTX_PATH1)
        platform = 'satp1'
    else:
        raise ValueError('Platform not recognized, please use satp1 or satp3')

    main(ctx, args.ys, args.ms, args.ye, args.me, platform=platform)
    """
    """
    div_set = [
        #[2023,10,2023,11],
        #[2023,11,2023,12],
        #[2023,12,2024,1],
        #[2024,1,2024,2],
        #[2024,2,2024,3],
        #[2024,3,2024,4],
        #[2024,4,2024,5],
        #[2024,5,2024,6],
        #[2024,6,2024,7],
        #[2024,7,2024,8],
        [2024,8,2024,9],
        [2024,9,2024,10],
        [2024,10,2024,11],
        [2024,11,2024,12],
        [2024,12,2025,1],
        [2025,1,2025,2],
        ]
    """
    #for ys, ms, ye, me in div_set:
    #    print(ys, ms, ye, me)
    #    main(CTX3, ys, ms, ye, me, platform='satp3')
    #   #main(CTX1, ys, ms, ye, me, platform='satp1')