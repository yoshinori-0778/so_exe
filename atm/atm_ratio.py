import numpy as np
import pandas as pd
import os, sys, datetime, yaml
from scipy.signal import csd
import gc
import subprocess
from memory_profiler import profile
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import spearmanr

from sodetlib.operations.bias_steps import BiasStepAnalysis
from sodetlib.operations.iv import IVAnalysis
import sodetlib.tes_param_correction as tpc
from sotodlib.io.hk_utils import get_hkaman
from sotodlib.io import hkdb
from sotodlib.utils.procs_pool import get_exec_env
from sotodlib import core
from sotodlib.preprocess.preprocess_util import init_logger, Pipeline

sys.path.append('/home/ys5857/workspace/script/so_ana/')
from config import *
from util.misc import read_pkl, save_pkl

import psutil, os
def print_mem():
    print(f"Memory: {psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB")


def corr(x, y):
    xz = (x - np.mean(x)) / np.std(x)
    yz = (y - np.mean(y)) / np.std(y)
    return np.corrcoef(xz, yz)[0, 1]

def get_correlation(x,y):
    rho, pval = spearmanr(x, y)
    cor = corr(x,y)
    return rho, pval, cor

def exe_atm_ratio_satp3(iobsid, iws, savep=None, config_init_path=None):
    print(iobsid, iws, savep)
    tracemalloc.start()
    if config_init_path is None:
        config_init_path = '/home/ys5857/workspace/script/so_exe/atm/config/satp3/preprocessing_config_20260520_init.yaml'
    configs_init = yaml.safe_load(open(config_init_path, "r"))
    context_init = core.Context(configs_init["context_file"])
    logger = init_logger("preprocess", verbosity=0)

    iband = 'f090'
    dets090 = {'wafer_slot':iws, 'wafer.bandpass':iband}
    meta090 = context_init.get_meta(iobsid, dets=dets090)
    pipe = Pipeline(configs_init["process_pipe"], logger=logger)
    aman090 = context_init.get_obs(meta090)
    pipe.run(aman090)

    iband = 'f150'
    dets150 = {'wafer_slot':iws, 'wafer.bandpass':iband}
    meta150 = context_init.get_meta(iobsid, dets=dets150)
    pipe = Pipeline(configs_init["process_pipe"], logger=logger)
    aman150 = context_init.get_obs(meta150)
    pipe.run(aman150)

    i = 0
    idetid = aman090.det_info.det_id[i]
    UFM = idetid.split('_')[0]
    detid_posf090 = np.array([idetid.split('_')[-1][:-1] for idetid in aman090.det_info.det_id])
    detid_posf150 = np.array([idetid.split('_')[-1][:-1] for idetid in aman150.det_info.det_id])
    uni_detid_pos = np.unique(np.concatenate([detid_posf090, detid_posf150]))

    uni_detid_full = []
    for idetid_pos in uni_detid_pos:
        idxf090 = np.where(detid_posf090 == idetid_pos)[0]
        idxf150 = np.where(detid_posf150 == idetid_pos)[0]
        if len(idxf090) + len(idxf150) == 4:
            uni_detid_full.append(idetid_pos)
    uni_detid_full = np.array(uni_detid_full)

    # from Lyman's calculation
    CMB_per_RJ = {'f090': 1.245, 'f150': 1.669}
    pW_per_RJ = {'f090': 0.363, 'f150': 0.447}
    fs = 200
    div = 4
    scan_leng = 200*160


    rets = {}
    j = 0
    for j in range(len(uni_detid_full)):
        ret = {}
        idetid_pos = uni_detid_full[j]
        
        band = 'f090'
        A090 = aman090.signal[aman090.det_info.det_id == f'{UFM}_{band}_{idetid_pos}A'][0]
        B090 = aman090.signal[aman090.det_info.det_id == f'{UFM}_{band}_{idetid_pos}B'][0]
        A090 = A090 / CMB_per_RJ[band] * pW_per_RJ[band]
        B090 = B090 / CMB_per_RJ[band] * pW_per_RJ[band]
        
        band = 'f150'
        A150 = aman150.signal[aman150.det_info.det_id == f'{UFM}_{band}_{idetid_pos}A'][0]
        B150 = aman150.signal[aman150.det_info.det_id == f'{UFM}_{band}_{idetid_pos}B'][0]
        A150 = A150 / CMB_per_RJ[band] * pW_per_RJ[band]
        B150 = B150 / CMB_per_RJ[band] * pW_per_RJ[band]
        xys = [[A090, A150], [B090, B150]]
        pols = ['A', 'B']
        h = 0
        for h in range(2):
            x,y = xys[h]
            ipol = pols[h]
            leng = len(x)
            coeff_full, _ = np.polyfit(x, y, 1)
            corr_full = np.corrcoef(x, y)[0][1]
            
            num_scan = leng//scan_leng
            coeffs_scan = []
            corrs_scan = []
            for i in range(num_scan):
                s = slice(i*scan_leng, (i+1)*scan_leng)
                icoeff, _ = np.polyfit(x[s], y[s], 1)
                icorr = np.corrcoef(x[s], y[s])[0][1]
                coeffs_scan.append(icoeff)
                corrs_scan.append(icorr)
            coeffs_scan = np.array(coeffs_scan)
            corrs_scan = np.array(corrs_scan)
            #plt.plot(coeffs_scan)
            coeff_full, np.median(coeffs_scan), np.mean(coeffs_scan)
            
            ret[f'poly_full_{ipol}'] = coeff_full
            ret[f'corr_poly_full_{ipol}'] = corr_full
            ret[f'poly_scan_{ipol}'] = coeffs_scan
            ret[f'corr_poly_scan_{ipol}'] = corrs_scan
            
            nperseg = int(leng/div)
            f, Pxy = csd(x, y, fs=fs, nperseg=nperseg)
            f, Pxx = csd(x, x, fs=fs, nperseg=nperseg)
            f, Pyy = csd(y, y, fs=fs, nperseg=nperseg)
            
            ratio = Pxy/Pxx            
            ifl = f < 0.01
            ratio_low = np.median(np.abs(ratio)[ifl])
            ifl = (f > 0.01) & (f < 0.1)
            ratio_med = np.median(np.abs(ratio)[ifl])
            ifl = (f > 0.1) & (f < 1)
            ratio_high = np.median(np.abs(ratio)[ifl])
            
            ratio_low, ratio_med, ratio_high
            
            ret[f'ratio_low_{ipol}'] = ratio_low
            ret[f'ratio_med_{ipol}'] = ratio_med
            ret[f'ratio_high_{ipol}'] = ratio_high
            
        rets[f'{idetid_pos}'] = ret
        if savep is None:
            savep = f'/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/2026/06/atm_ratio/satp3/{iobsid}_{iws}.pkl'
        save_pkl(rets, savep)

        del x, y, f, Pxy, Pxx, Pyy, ratio, ret, A150, B150, A090, B090
        gc.collect()

    #snapshot = tracemalloc.take_snapshot()
    #top_stats = snapshot.statistics("lineno")
    #for stat in top_stats[:10]:
    #    print(stat)

    #print('before del aman090, aman150, meta090, meta150, rets')
    #print_mem()
    context_init.clear()
    del aman090, aman150, meta090, meta150, context_init, rets
    gc.collect()
    #print('after del aman090, aman150, meta090, meta150, rets')
    #print_mem()

    #snapshot = tracemalloc.take_snapshot()
    #top_stats = snapshot.statistics("lineno")
    #for stat in top_stats[:10]:
    #    print(stat)

    return None

def exe_atm_ratio_satp1(iobsid, iws, savep=None, config_init_path=None):
    if config_init_path is None:
        config_init_path = '/home/ys5857/workspace/script/so_exe/atm/config/satp1/preprocessing_config_20260520_init.yaml'
    configs_init = yaml.safe_load(open(config_init_path, "r"))
    context_init = core.Context(configs_init["context_file"])
    logger = init_logger("preprocess", verbosity=0)

    iband = 'f090'
    dets090 = {'wafer_slot':iws, 'wafer.bandpass':iband}
    meta090 = context_init.get_meta(iobsid, dets=dets090)
    pipe = Pipeline(configs_init["process_pipe"], logger=logger)
    aman090 = context_init.get_obs(meta090)
    pipe.run(aman090)
    del pipe

    iband = 'f150'
    dets150 = {'wafer_slot':iws, 'wafer.bandpass':iband}
    meta150 = context_init.get_meta(iobsid, dets=dets150)
    pipe = Pipeline(configs_init["process_pipe"], logger=logger)
    aman150 = context_init.get_obs(meta150)
    pipe.run(aman150)
    del pipe

    i = 0
    idetid = aman090.det_info.det_id[i]
    UFM = idetid.split('_')[0]
    detid_posf090 = np.array([idetid.split('_')[-1][:-1] for idetid in aman090.det_info.det_id])
    detid_posf150 = np.array([idetid.split('_')[-1][:-1] for idetid in aman150.det_info.det_id])
    uni_detid_pos = np.unique(np.concatenate([detid_posf090, detid_posf150]))

    uni_detid_full = []
    for idetid_pos in uni_detid_pos:
        idxf090 = np.where(detid_posf090 == idetid_pos)[0]
        idxf150 = np.where(detid_posf150 == idetid_pos)[0]
        if len(idxf090) + len(idxf150) == 4:
            uni_detid_full.append(idetid_pos)
    uni_detid_full = np.array(uni_detid_full)

    # from Lyman's calculation
    CMB_per_RJ = {'f090': 1.245, 'f150': 1.669}
    pW_per_RJ = {'f090': 0.363, 'f150': 0.447}
    fs = 200
    div = 4
    scan_leng = 200*160


    rets = {}
    j = 0
    for j in range(len(uni_detid_full)):
        ret = {}
        idetid_pos = uni_detid_full[j]
        
        band = 'f090'
        A090 = aman090.signal[aman090.det_info.det_id == f'{UFM}_{band}_{idetid_pos}A'][0]
        B090 = aman090.signal[aman090.det_info.det_id == f'{UFM}_{band}_{idetid_pos}B'][0]
        A090 = A090 / CMB_per_RJ[band] * pW_per_RJ[band]
        B090 = B090 / CMB_per_RJ[band] * pW_per_RJ[band]
        
        band = 'f150'
        A150 = aman150.signal[aman150.det_info.det_id == f'{UFM}_{band}_{idetid_pos}A'][0]
        B150 = aman150.signal[aman150.det_info.det_id == f'{UFM}_{band}_{idetid_pos}B'][0]
        A150 = A150 / CMB_per_RJ[band] * pW_per_RJ[band]
        B150 = B150 / CMB_per_RJ[band] * pW_per_RJ[band]
        xys = [[A090, A150], [B090, B150]]
        pols = ['A', 'B']
        h = 0
        for h in range(2):
            x,y = xys[h]
            ipol = pols[h]
            leng = len(x)
            coeff_full, _ = np.polyfit(x, y, 1)
            corr_full = np.corrcoef(x, y)[0][1]
            
            num_scan = leng//scan_leng
            coeffs_scan = []
            corrs_scan = []
            for i in range(num_scan):
                s = slice(i*scan_leng, (i+1)*scan_leng)
                icoeff, _ = np.polyfit(x[s], y[s], 1)
                icorr = np.corrcoef(x[s], y[s])[0][1]
                coeffs_scan.append(icoeff)
                corrs_scan.append(icorr)
            coeffs_scan = np.array(coeffs_scan)
            corrs_scan = np.array(corrs_scan)
            #plt.plot(coeffs_scan)
            coeff_full, np.median(coeffs_scan), np.mean(coeffs_scan)
            
            ret[f'poly_full_{ipol}'] = coeff_full
            ret[f'corr_poly_full_{ipol}'] = corr_full
            ret[f'poly_scan_{ipol}'] = coeffs_scan
            ret[f'corr_poly_scan_{ipol}'] = corrs_scan
            
            nperseg = int(leng/div)
            f, Pxy = csd(x, y, fs=fs, nperseg=nperseg)
            f, Pxx = csd(x, x, fs=fs, nperseg=nperseg)
            f, Pyy = csd(y, y, fs=fs, nperseg=nperseg)
            
            ratio = Pxy/Pxx            
            ifl = f < 0.01
            ratio_low = np.median(np.abs(ratio)[ifl])
            ifl = (f > 0.01) & (f < 0.1)
            ratio_med = np.median(np.abs(ratio)[ifl])
            ifl = (f > 0.1) & (f < 1)
            ratio_high = np.median(np.abs(ratio)[ifl])
            
            ratio_low, ratio_med, ratio_high
            
            ret[f'ratio_low_{ipol}'] = ratio_low
            ret[f'ratio_med_{ipol}'] = ratio_med
            ret[f'ratio_high_{ipol}'] = ratio_high
            
        rets[f'{idetid_pos}'] = ret
        if savep is None:
            savep = f'/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/2026/06/atm_ratio/satp1/{iobsid}_{iws}.pkl'
        save_pkl(rets, savep)

        del x, y, f, Pxy, Pxx, Pyy, ratio, ret, A150, B150, A090, B090
        gc.collect()

    #print('before del aman090, aman150, meta090, meta150, rets')
    #print_mem()
    del aman090, aman150, meta090, meta150, rets
    gc.collect()
    #print('before del aman090, aman150, meta090, meta150, rets')
    #print_mem()

    return None



def exe_atm_ratio_satp3_multiprocess(executor, as_completed_callable, ys=None, ye=None, ms=None, me=None, saved=None):
    print(ys, ye, ms, me, saved)
    if ys is None:
        start = datetime.datetime(2024,7,1).timestamp()
        end = datetime.datetime(2025,11,1).timestamp()
    else:
        start = datetime.datetime(ys,ms,1).timestamp()
        end = datetime.datetime(ye,me,1).timestamp()
    ctx3 = core.Context('/home/ys5857/workspace/script/so_exe/atm/context/satp3/use_this_local_260520.yaml')
    obslist_all = ctx3.obsdb.query(f'timestamp > {start} and timestamp < {end} and type="obs" and subtype="cmb"')
    obslist = []
    for iobs in obslist_all:
        obslist.append(iobs['obs_id'])
    obslist = np.array(obslist)

    runlist = []
    for i in range(len(obslist)):
        iobsid = obslist[i].item()
        for iws in wss:
            if saved is None:
                savep = f'/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/2026/06/atm_ratio/satp3/{iobsid}_{iws}.pkl'
            else:
                savep = os.path.join(saved, f'{iobsid}_{iws}.pkl')
            if not os.path.exists(savep):
            #if True:
                irunlist = {'obsid': iobsid, 'ws': iws, 'savep': savep}
                runlist.append(irunlist)

    n_runs = len(runlist)
    print(f'number of runs: {n_runs}')
    future_to_rl = {executor.submit(exe_atm_ratio_satp3, iobsid=rl['obsid'], iws=rl['ws'], savep=rl['savep']): rl for rl in runlist}
    futures = list(future_to_rl)

    n = 0
    for future in as_completed_callable(futures):
        rl = future_to_rl[future]
        try:
            n += 1
            _ = future.result()
            futures.remove(future)
            print(f'Processing Finished correctly {n}/{n_runs}')
        except Exception as e:
            print(f'Processing {n}/{n_runs} generated an exception: {e}')
            futures.remove(future)
        finally:
            del future
            gc.collect()

def exe_atm_ratio_satp1_multiprocess(executor, as_completed_callable, ys=None, ye=None, ms=None, me=None, saved=None):
    if ys is None:
        start = datetime.datetime(2024,7,1).timestamp()
        end = datetime.datetime(2025,11,1).timestamp()
    else:
        start = datetime.datetime(ys,ms,1).timestamp()
        end = datetime.datetime(ye,me,1).timestamp()
    ctx1 = core.Context('/home/ys5857/workspace/script/so_exe/atm/context/satp1/use_this_local_260520.yaml')
    obslist_all = ctx1.obsdb.query(f'timestamp > {start} and timestamp < {end} and type="obs" and subtype="cmb"')
    obslist = []
    for iobs in obslist_all:
        obslist.append(iobs['obs_id'])
    obslist = np.array(obslist)

    runlist = []
    for i in range(len(obslist)):
        iobsid = obslist[i].item()
        for iws in wss:
            if saved is None:
                savep = f'/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/2026/06/atm_ratio/satp1/{iobsid}_{iws}.pkl'
            else:
                savep = os.path.join(saved, f'{iobsid}_{iws}.pkl')
            if not os.path.exists(savep):
            #if True:
                irunlist = {'obsid': iobsid, 'ws': iws, 'savep': savep}
                runlist.append(irunlist)

    n_runs = len(runlist)
    print(f'number of runs: {n_runs}')
    future_to_rl = {executor.submit(exe_atm_ratio_satp1, iobsid=rl['obsid'], iws=rl['ws'], savep=rl['savep']): rl for rl in runlist}
    futures = list(future_to_rl)

    n = 0
    for future in as_completed_callable(futures):
        rl = future_to_rl[future]
        try:
            n += 1
            _ = future.result()
            futures.remove(future)
            print(f'Processing Finished correctly {n}/{n_runs}')
        except Exception as e:
            print(f'Processing {n}/{n_runs} generated an exception: {e}')
            futures.remove(future)
        finally:
            del future
            gc.collect()

def test1():
    start = datetime.datetime(2025,7,2).timestamp()
    end = datetime.datetime(2025,11,3).timestamp()
    ctx3 = core.Context('/home/ys5857/workspace/script/so_exe/atm/context/satp3/use_this_local_260520.yaml')
    obslist_all = ctx3.obsdb.query(f'timestamp > {start} and timestamp < {end} and type="obs" and subtype="cmb"')
    obslist = []
    for iobs in obslist_all:
        obslist.append(iobs['obs_id'])
    obslist = np.array(obslist)
    print(len(obslist))

    runlist = []
    for i in range(len(obslist))[:1]:
        iobsid = obslist[i].item()
        for iws in wss:
            savep = f'/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/2026/06/atm_ratio/satp3/{iobsid}_{iws}.pkl'
            if not os.path.exists(savep):
            #if True:
                irunlist = {'obsid': iobsid, 'ws': iws}
                runlist.append(irunlist)

    n_runs = len(runlist)
    print(f'number of runs: {n_runs}')
    for rl in runlist:
        exe_atm_ratio_satp3(iobsid=rl['obsid'], iws=rl['ws'])

def test2():
    #exe_atm_ratio_satp3(iobsid='obs_1754702754_satp3_1111111', iws='ws0')
    exe_atm_ratio_satp3(iobsid='obs_1726651313_satp3_1111111', iws='ws4', savep = '/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/2026/06/atm_ratio/satp3/obs_1726651313_satp3_1111111_ws4.pkl')

if __name__ == "__main__":
    """
    #test()

    iobsid = 'obs_1730261106_satp3_1011111'
    iws = 'ws0'
    iband = 'f150'
    main_each_satp3(iobsid, iws, iband)

    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', type=int, default=2, help='Number of processes to use')
    parser.add_argument('--sat', type=str, default='satp3', help='SAT to use')
    parser.add_argument('--ys', type=int, default=None, help='Year start')
    parser.add_argument('--ye', type=int, default=None, help='Year end')
    parser.add_argument('--ms', type=int, default=None, help='Month start')
    parser.add_argument('--me', type=int, default=None, help='Month end')
    parser.add_argument('--saved', type=str, default=None, help='Save directory')
    parser.add_argument('--test', action='store_true', help='perform test function or not')
    args = parser.parse_args()
    if args.test:
        print('test2')
        test2()
    else:
        rank, executor, as_completed_callable = get_exec_env(args.nproc)
        if rank == 0:
            if args.sat == 'satp3':
                print('satp3')
                exe_atm_ratio_satp3_multiprocess(executor, as_completed_callable, ys=args.ys, ye=args.ye, ms=args.ms, me=args.me, saved=args.saved)
            elif args.sat == 'satp1':
                exe_atm_ratio_satp1_multiprocess(executor, as_completed_callable, ys=args.ys, ye=args.ye, ms=args.ms, me=args.me, saved=args.saved)
            else:
                print('Error: --sat should be satp1 or satp3')