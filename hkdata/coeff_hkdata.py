import numpy as np
import sys, datetime, os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from pathlib import Path
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from hkdata.hkall import read_hk_level3, get_apex_data 
from so_utils.misc import save_pkl
from sotodlib.utils.procs_pool import get_exec_env

def get_all_hkdata(ys, ms, ds, ye, me, de):
    print(f'Loading hk data from {ys}-{ms}-{ds} to {ye}-{me}-{de}')
    hkcfg_file3 = '/home/ys5857/workspace/script/so_exe/hkdata/hk_yaml/satp3_hk_all.yaml'
    hkcfg_file1 = '/home/ys5857/workspace/script/so_exe/hkdata/hk_yaml/satp1_hk_all.yaml'
    hkcfg_files = '/home/ys5857/workspace/script/so_exe/hkdata/hk_yaml/site_hk.yaml'

    dtst = datetime.datetime(ys, ms, ds, 0, 0, 0, 0, tzinfo=datetime.timezone.utc)
    dten = datetime.datetime(ye, me, de, 0, 0, 0, 0, tzinfo=datetime.timezone.utc) 
    ist = dtst.timestamp()
    ien = dten.timestamp()

    ret3, key3 = read_hk_level3(ist, ien, hkcfg_file=hkcfg_file3)
    ret1, key1 = read_hk_level3(ist, ien, hkcfg_file=hkcfg_file1)
    rets, keys = read_hk_level3(ist, ien, hkcfg_file=hkcfg_files)
    try:
        pwv_apex = get_apex_data(dtst, dten)
        rets['pwv_apex'] = {}
        rets['pwv_apex']['ts'] = np.array(pwv_apex['timestamps'])
        rets['pwv_apex']['val'] = 0.03+0.84*np.array(pwv_apex['pwv']) # converted to number corresponds to Class
        # Numbers are from : https://github.com/simonsobs/mf-cmg-paper-noise-db/blob/050a3ec49831a9030b1574dc05976eed63417fd3/scripts/z_data_add_pwv.py#L118
    except Exception as e:
        print(e)
        pwv_apex = None
        rets['pwv_apex'] = {}
        rets['pwv_apex']['ts'] = np.nan
        rets['pwv_apex']['val'] = np.nan
    print('Finished loading hk data')
    return ret3, key3, ret1, key1, rets, keys, pwv_apex

def get_val(its, ref_ts, ref_data, max_dt=10.0):
    """
    Parameters
    ----------
    its : array-like
        input timestamps
    ref_ts : array-like
        reference timestamps (sorted)
    ref_data : array-like
        reference data
    max_dt : float
        maximum allowed time difference [same unit as timestamps]

    Returns
    -------
    ref_datas : ndarray
        matched data (NaN if no valid match)
    final_idxs : ndarray
        matched indices (-1 if no valid match)
    """
    idxs = np.searchsorted(ref_ts, its)

    idxs_closest = np.clip(idxs, 1, len(ref_ts)-1)
    lefts  = ref_ts[idxs_closest - 1]
    rights = ref_ts[idxs_closest]

    dt_left  = np.abs(its - lefts)
    dt_right = np.abs(rights - its)

    closer_to_left = dt_left < dt_right
    final_idxs = idxs_closest - closer_to_left.astype(int)
    
    min_dt = np.minimum(dt_left, dt_right)
    valid = min_dt <= max_dt
    
    ref_datas = np.full(len(its), np.nan)
    ref_datas[valid] = ref_data[final_idxs[valid]]

    final_idxs = np.where(valid, final_idxs, -1)
    return ref_datas, final_idxs

def group_by_time_gap(ts, data, gap=10.0):
    ts = np.asarray(ts)
    data = np.asarray(data)
    dts = np.diff(ts)
    split_idx = np.where(dts > gap)[0] + 1
    ts_groups = np.split(ts, split_idx)
    data_groups = np.split(data, split_idx)
    return ts_groups, data_groups

def calc_mean(ts, data):
    ts_groups, data_groups = group_by_time_gap(ts, data, gap=10.0)
    ts_mean   = np.array([g.mean() for g in ts_groups])
    data_mean = np.array([g.mean() for g in data_groups])
    return ts_mean, data_mean

def _linfunc(x, a, b):
    return a * x + b

def sigma_clip_linfit(x, y, yerr=None, sigma=5, max_iter=5):
    """
    Sigma-clipped linear fit using scipy.curve_fit.

    Returns
    -------
    coeffs : (a, b)
        best-fit parameters
    errs : (σ_a, σ_b)
        parameter uncertainties
    chi2 : float
        chi-square
    dof : int
        degrees of freedom
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    if yerr is not None:
        yerr = np.asarray(yerr)
        mask &= np.isfinite(yerr)

    coeffs = (np.nan, np.nan)
    pcov = np.full((2, 2), np.nan)

    for i in range(max_iter):
        if mask.sum() < 2:
            break

        if yerr is None:
            popt, pcov = curve_fit(_linfunc, x[mask], y[mask])
            resid = y[mask] - _linfunc(x[mask], *popt)
            std = np.std(resid, ddof=2)
        else:
            popt, pcov = curve_fit(_linfunc, x[mask], y[mask], sigma=yerr[mask], absolute_sigma=True)
            resid = y[mask] - _linfunc(x[mask], *popt)
            std = np.std(resid, ddof=2)
        coeffs = popt
        if std == 0:
            break
        if i != max_iter - 1:
            mask[mask] = np.abs(resid) < sigma * std

    dof = mask.sum() - 2
    if dof > 0:
        if yerr is None:
            chi2 = np.sum((resid / std) ** 2)
        else:
            chi2 = np.sum((resid / yerr[mask]) ** 2)
    else:
        chi2 = np.nan
    errs = np.sqrt(np.diag(pcov))
    return coeffs, errs, chi2, dof, mask

def fit_each(ret3, rets, ikey, iref_key, normal=True, max_dt=10.0, elcen=None, eldif=0.5, bscen=None, bsdif=0.5, plot= False):
    if normal:
        target_data, final_idxs = get_val(rets[iref_key]['ts'], ret3[ikey]['ts'], ret3[ikey]['val'], max_dt = max_dt)
        el_data, final_idxs_el = get_val(rets[iref_key]['ts'], ret3['el']['ts'], ret3['el']['val'], max_dt = max_dt)
        bs_data, final_idxs_bs = get_val(rets[iref_key]['ts'], ret3['bs']['ts'], ret3['bs']['val'], max_dt = max_dt)
        ref_data = rets[iref_key]['val']
    else:
        # take mean of each chunck 
        ts_mean, data_mean = calc_mean(ret3[ikey]['ts'], ret3[ikey]['val'])
        target_data = data_mean
        ref_data, final_idxs = get_val(ts_mean, rets[iref_key]['ts'], rets[iref_key]['val'], max_dt=10.0)
        el_data, final_idxs_el = get_val(ts_mean, ret3['el']['ts'], ret3['el']['val'], max_dt = max_dt)
        bs_data, final_idxs_bs = get_val(ts_mean, ret3['bs']['ts'], ret3['bs']['val'], max_dt = max_dt)
    
    # select data
    if elcen is None:
        iflel = np.full(len(el_data), True)
    else:        
        iflel = (el_data > (elcen - eldif)) & (el_data < (elcen + eldif))
    if bscen is None:
        iflbs = np.full(len(bs_data), True)
    else:
        iflbs = (bs_data > (bscen - bsdif)) & (bs_data < (bscen + bsdif))
    if iref_key == 'pwv_class':
        iflpwv = (ref_data > 0) & (ref_data < 3)
    else:
        iflpwv = np.full(len(target_data), True)
    ifl = iflel & iflbs & iflpwv
    ref_data = ref_data[ifl]
    target_data = target_data[ifl]

    # fit
    yerr = None
    coeffs, errs, chi2, dof, mask = sigma_clip_linfit(ref_data, target_data, yerr, sigma=5, max_iter=5)

    if plot:
        if len(ref_data) != 0:
            ix = np.linspace(np.min(ref_data), np.max(ref_data), 10)
            ifit = coeffs[0]*ix + coeffs[1]
            plt.figure()
            plt.plot(ref_data, target_data,'.')
            plt.plot(ref_data[mask], target_data[mask],'.')
            plt.plot(ix, ifit)
            plt.xlabel(iref_key)
            plt.ylabel(ikey)
            plt.title(f'{ikey}_{iref_key}_el_{elcen}_bs_{bscen}')

    ret = {}
    ret['coeff'] = coeffs
    ret['err'] = errs
    ret['chi2'] = chi2
    ret['dof'] = dof
    return ret

def exe_fit(ret3, rets, allkeys, iref_key, keys_normal, keys_sparse,
            max_dt=10.0, elcen=60, eldif=0.5, bscen=0, bsdif=0.5, plot= False, save=False,
            saved='/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/hkdata_coeff/satp3/',
            savelabel = 'date.pkl'):
    print(f'======={saved}, {savelabel}, {iref_key}, {elcen}, {bscen}, {plot}, {save}=======')
    savedata = {}
    savedata['key_ref'] = iref_key
    for ikey in allkeys:
        try:
            if ikey in keys_normal:
                iret = fit_each(ret3, rets, ikey, iref_key, normal = True, max_dt=max_dt, elcen=elcen, eldif=eldif, bscen=bscen, bsdif=bsdif, plot=plot)
            elif ikey in keys_sparse:
                iret = fit_each(ret3, rets, ikey, iref_key, normal = False, max_dt=max_dt, elcen=elcen, eldif=eldif, bscen=bscen, bsdif=bsdif, plot=plot)
            else:
                print(f'{ikey} is no in referenced keys')
                break
            savedata[ikey] = iret
        except Exception as e:
            print(ikey)
            print(e)
            
    if save:
        os.makedirs(saved, exist_ok=True)
        csaved = os.path.join(saved, f'coeff_{iref_key}')
        os.makedirs(csaved, exist_ok=True)
        if elcen is None:
            elcen = 'all'
        if bscen is None:
            bscen = 'all'
        csaved2 = os.path.join(csaved, f'el_{elcen}_bs_{bscen}')
        os.makedirs(csaved2, exist_ok=True)
        savep = os.path.join(csaved2, savelabel)
        print(f'Saving to {savep}: {iref_key}')
        save_pkl(savedata, savep)
    else:
        print('====NOT SAVING======')
    return savedata

def labels():
    use_thermo_satp3_normal = ['4K_filter', 'YBCO', '40K_filter', 'URH_40K_plate','fp',
                  'dr_still_stage_temp', '1K-1', '1K-2', '1K-3', '1K-5', '1K-11',
                  'D2.1', 'D2.2', 'D2.3', 'D2.4', 'D2.5', 'D2.6',
                  'D1.1', 'D1.2', 'D1.5', 'D1.6']
    use_thermo_satp3_sparse = ['dr_mxc', '100mK-2', '100mK-3', '100mK-4', 'DR_PTC2', 'DR_PTC1_CH']
    keys_satp3 = use_thermo_satp3_normal + use_thermo_satp3_sparse

    use_thermo_satp1_normal = ['fp', 'dr_mxc', 'heater',
                        'cra', 'stop', 'lpe', 'lens2', 'hs_double_rod',
                        '4K_urh', '4K_coldhead', '4K_filter', 'heatstrap',
                        '40K_urh', 'be_dr_port', 'busbar', '40K_coldhead', 'filter_plate',
                        'chwp_ybco', 'chwp_rotor1', 'chwp_rotor2', 'chwp_rotor3']
    use_thermo_satp1_sparse = []
    keys_satp1 = use_thermo_satp1_normal + use_thermo_satp1_sparse
    return keys_satp3, keys_satp1, use_thermo_satp3_normal, use_thermo_satp3_sparse, use_thermo_satp1_normal, use_thermo_satp1_sparse

def main_each_refkey(ys, ms, ds, ye, me, de, iref_key = 'outside_temp'):
    max_dt = 10
    elcen = 60
    eldif = 0.5
    bscen = 45
    bsdif = 0.5
    plot = False
    save = True
    elcens = [60, 48, 40, None]
    bscens_satp1 = [0, -45, 45, None]
    bscens_satp3 = [0,]

    ret3, key3, ret1, key1, rets, keys, pwv_apex = get_all_hkdata(ys, ms, ds, ye, me, de)
    keys_satp3, keys_satp1, use_thermo_satp3_normal, use_thermo_satp3_sparse, use_thermo_satp1_normal, use_thermo_satp1_sparse = labels()
    savelabel=f'{ys}_{ms:02d}_{ds:02d}_{ye}_{me:02d}_{de:02d}.pkl'
    print('starting fitting')
    for elcen in elcens:
        for bscen in bscens_satp3:
            print(f'SATp3: fitting with {elcen}, {bscen}')
            _ = exe_fit(ret3, rets, keys_satp3, iref_key, use_thermo_satp3_normal, use_thermo_satp3_sparse,
                            max_dt, elcen, eldif, bscen, bsdif, plot=plot, save=save,
                            saved='/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/hkdata_coeff/satp3/',
                            savelabel=savelabel)
        for bscen in bscens_satp1:
            print(f'SATp1: fitting with {elcen}, {bscen}')
            _ = exe_fit(ret1, rets, keys_satp1, iref_key, use_thermo_satp1_normal, use_thermo_satp1_sparse,
                            max_dt, elcen, eldif, bscen, bsdif, plot=plot, save=save,
                            saved='/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/hkdata_coeff/satp1/',
                            savelabel=savelabel)
    del ret3, key3, ret1, key1, rets, keys, pwv_apex
    return

def main(ys, ms, ds, ye, me, de):
    print(ys, ms, ds, ye, me, de)
    ref_keys = ['outsite_temp', 'pwv_class', 'pwv_apex']
    for iref_key in ref_keys:
        main_each_refkey(ys, ms, ds, ye, me, de, iref_key)
    return

def main_debug(ys, ms, ds, ye, me, de):
    #ref_keys = ['outsite_temp', 'pwv_class', 'pwv_apex']
    #for iref_key in ref_keys:
    #    main_each_refkey(ys, ms, ds, ye, me, de, iref_key)
    print(ys, ms, ds, ye, me, de)

def test():
    ys = 2026
    ms = 1
    ds = 3
    ye = 2026
    me = 1
    de = 4

    ret3, key3, ret1, key1, rets, keys, pwv_apex = get_all_hkdata(ys, ms, ds, ye, me, de)
    iref_key = 'outsite_temp'
    keys_satp3, keys_satp1, use_thermo_satp3_normal, use_thermo_satp3_sparse, use_thermo_satp1_normal, use_thermo_satp1_sparse = labels()
    savelabel=f'{ys}_{ms:02d}_{ds:02d}_{ye}_{me:02d}_{de:02d}.pkl'

    max_dt = 10
    elcen = 60
    eldif = 0.5
    bscen = 45
    bsdif = 0.5
    plot = False
    save = True

    elcens = [60, 48, 40, None]
    bscens_satp1 = [0, -45, 45, None]
    bscens_satp3 = [0,]
    for elcen in elcens:
        for bscen in bscens_satp3:
            savedata = exe_fit(ret3, rets, keys_satp3, iref_key, use_thermo_satp3_normal, use_thermo_satp3_sparse,
                            max_dt, elcen, eldif, bscen, bsdif, plot, save,
                            saved='/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/hkdata_coeff/satp3/',
                            savelabel=savelabel)
        for bscen in bscens_satp1:
            savedata = exe_fit(ret1, rets, keys_satp1, iref_key, use_thermo_satp1_normal, use_thermo_satp1_sparse,
                            max_dt, elcen, eldif, bscen, bsdif, plot, save,
                            saved='/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/hkdata_coeff/satp1/',
                            savelabel=savelabel)
            
def exe_main_multiprocess(executor, as_completed_callable, 
                          start_year=2023, end_year=2026,
                          start_month=10, end_month=2, overwrite=False):
    start = datetime.datetime(start_year, start_month, 1)
    end   = datetime.datetime(end_year, end_month, 1)
    print('exe_main_multiprocess')
    print(start, end)

    runlist = []
    current = start
    while current <= end:
        ys, ms, ds = current.year, current.month, current.day
        next_day = current + datetime.timedelta(days=1)
        ye, me, de = next_day.year, next_day.month, next_day.day
        irunlist = {'ys': ys, 'ms': ms, 'ds': ds, 'ye': ye, 'me': me, 'de': de}
        runlist.append(irunlist)
        current = next_day

    n_runs = len(runlist)
    print(f'number of runs: {n_runs}')
    print(runlist[0])
    print(runlist[-1])
    future_to_rl = {executor.submit(main, ys=rl['ys'], ms=rl['ms'], ds=rl['ds'], ye=rl['ye'], me=rl['me'], de=rl['de']): rl for rl in runlist[:10]}
    #future_to_rl = {executor.submit(main, ys=rl['ys'], ms=rl['ms'], ds=rl['ds'], ye=rl['ye'], me=rl['me'], de=rl['de']): rl for rl in runlist}
    #future_to_rl = {executor.submit(main_debug, ys=rl['ys'], ms=rl['ms'], ds=rl['ds'], ye=rl['ye'], me=rl['me'], de=rl['de']): rl for rl in runlist}
    futures = list(future_to_rl)

    n = 0
    for future in as_completed_callable(futures):
        rl = future_to_rl[future]
        try:
            n += 1
            _ = future.result()
            futures.remove(future)
            #print(f'Processing Finished correctly {n}/{n_runs}')
        except Exception as e:
            #print(f'Processing {n}/{n_runs} generated an exception: {e}')
            futures.remove(future)
        finally:
            del future

def exe_each_month(ys, ms, ye, me):
    start = datetime.datetime(ys, ms, 1)
    end   = datetime.datetime(ye, me, 1)
    print(start, end)

    runlist = []
    current = start
    while current <= end:
        ys, ms, ds = current.year, current.month, current.day
        next_day = current + datetime.timedelta(days=1)
        ye, me, de = next_day.year, next_day.month, next_day.day
        irunlist = {'ys': ys, 'ms': ms, 'ds': ds, 'ye': ye, 'me': me, 'de': de}
        runlist.append(irunlist)
        current = next_day
    
    for rl in runlist:
        print(rl)
        try:
            main(ys=rl['ys'], ms=rl['ms'], ds=rl['ds'], ye=rl['ye'], me=rl['me'], de=rl['de'])
        except Exception as e:
            print(f'Processing generated an exception: {e}')
    
def main_each_month():
    div_set = [
            [2023,10,2023,11],
            [2023,11,2023,12],
            [2023,12,2024,1],
            [2024,1,2024,2],
            [2024,2,2024,3],
            [2024,3,2024,4],
            [2024,4,2024,5],
            [2024,5,2024,6],
            [2024,6,2024,7],
            [2024,7,2024,8],
            [2024,8,2024,9],
            [2024,9,2024,10],
            [2024,10,2024,11],
            [2024,11,2024,12],
            [2024,12,2025,1],
            [2025,1,2025,2],
            [2025,2,2025,3],
            [2025,3,2025,4],
            [2025,4,2025,5],
            [2025,5,2025,6],
            [2025,6,2025,7],
            [2025,7,2025,8],
            [2025,8,2025,9],
            ]
    for ys, ms, ye, me in div_set:
        print(ys, ms, ye, me)
        exe_each_month(ys, ms, ye, me)

def test_each_month():
    print('Test each month')
    div_set = [
            [2025,11,2025,12],
            ]
    for ys, ms, ye, me in div_set:
        print(ys, ms, ye, me)
        exe_each_month(ys, ms, ye, me)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--nproc', type=int, default=2, help='Number of processes to use')
    parser.add_argument('--ys', type=int, default=2024, help='Year start')
    parser.add_argument('--ye', type=int, default=2024, help='Year end')
    parser.add_argument('--ms', type=int, default=2, help='Month start')
    parser.add_argument('--me', type=int, default=3, help='Month end')
    parser.add_argument('--test', action='store_true', help='Run test function')
    args = parser.parse_args()
    if args.test:
        #test()
        #ys, ms, ds = 2025, 11, 1
        #ye, me, de = 2025, 11, 2
        #main(ys, ms, ds, ye, me, de)
        test_each_month()
    elif args.nproc > 1:
        rank, executor, as_completed_callable = get_exec_env(args.nproc)
        if rank == 0:
            exe_main_multiprocess(executor, as_completed_callable, 
                          start_year=args.ys, end_year=args.ye,
                          start_month=args.ms, end_month=args.me)
    else:
        #main_each_month()
        exe_each_month(ys=args.ys, ms=args.ms, ye=args.ye, me=args.me)
