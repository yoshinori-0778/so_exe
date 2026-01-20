import datetime, yaml, sys, os
import numpy as np
from tqdm import tqdm

from sotodlib import core, tod_ops
from sotodlib.io import hkdb

from pathlib import Path
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from so_utils.config import DEG, WSS
from so_utils.misc import save_pkl

def load_aman(ctx, iobs, band = 'f090'):
    """Loading axis manager for given observation ID.
    
    Parameters:
    - ctx: sotodlib core context
    - iobs: observation ID string
    - band: frequency band string, default 'f090'.
    """
    band = 'f090'
    for i, iws in zip(iobs.split('_')[-1], WSS):
        if i == 0:
            pass
        else:
            ws = iws
            break
    meta = ctx.get_meta(iobs, dets={'wafer.type': 'OPTC', 'wafer_slot': ws, 'wafer.bandpass': band, })
    meta.restrict('dets', meta.det_info.det_id == meta.det_info.det_id[0])
    aman = ctx.get_obs(meta)

    # get turnaround flags    
    _ = tod_ops.flags.get_turnaround_flags(aman, t_buffer=0.1, truncate=True)
    if len(aman.timestamps) > 4000*200:
        print('Skip this data due to the data length is > 4000*200.')
        return np.nan
    return aman


def make_binning(sigs, rs, r_bins):
    """binning signals based on input bins.
    Parameters:
    - sigs: input signals
    - rs: input positions
    - r_bins: binned center.
    Returns:
    - dictionary with binned results
    """
    inds = np.array([np.argmin(np.abs(ir - r_bins)) for ir in rs])
    nbins = len(r_bins)
    
    binned_count = np.zeros(nbins)
    binned_signal = np.zeros(nbins)
    binned_squared = np.zeros(nbins)
    
    np.add.at(binned_count, inds, 1)
    np.add.at(binned_signal, inds, sigs)
    np.add.at(binned_squared, inds, sigs**2)
    
    binned_signal = binned_signal/binned_count
    binned_squared = binned_squared/binned_count
    return {'bin': r_bins, 'sig': binned_signal, 'sig_sqre': binned_squared, 'count': binned_count}

def load_az(st, en, fields, hkcfg_path):
    """Load housekeeping data for given time range and fields.
    Parameters:
    - st: start timestamp
    - en: end timestamp
    - fields: list of field names to load
    - hkcfg_path: path to hk configuration yaml file
    Returns:
    - res: loaded housekeeping data
    """
    cfg = hkdb.HkConfig.from_yaml(hkcfg_path)
    spec = hkdb.LoadSpec(
        cfg=cfg,
        fields=fields,
        start=st, end=en,
    )
    res = hkdb.load_hk(spec, show_pb=True)
    return res

def get_az(its, azts, azdata):
    """Get azimuths corresponding to input timestamps.
    Parameters:
    - its: input timestamps
    - azts: azimuth timestamps
    - azdata: azimuth data
    Returns:
    - azs: azimuths corresponding to input timestamps
    - final_idxs: indices of azimuth data corresponding to input timestamps
    """
    idxs = np.searchsorted(azts, its)

    idxs_closest = np.clip(idxs, 1, len(azts)-1)
    lefts  = azts[idxs_closest - 1]
    rights = azts[idxs_closest]
    
    closer_to_left = np.abs(its - lefts) < np.abs(rights - its)
    final_idxs = idxs_closest - closer_to_left.astype(int)
    azs = azdata[final_idxs]
    return azs, final_idxs

def exe_azbin(ctx, iobs, hkcfg_path = '/home/ys5857/workspace/script/so_exe/hkdata/hk_yaml/satp3_temp_az.yaml',
               saved = '/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/2025/07/azbin/'):
    """Execute azimuth binning for given observation ID.
    Parameters:
    - ctx: sotodlib core context
    - iobs: observation ID string
    - hkcfg_path: path to hk configuration yaml file
    - saved: directory to save output pickle file
    """
    aman = load_aman(ctx, iobs)

    # to get fields and keys
    table = yaml.safe_load(open(hkcfg_path, "r"))['aliases']
    keys = []
    fields = []
    for ikey in table.keys():
        keys.append(ikey)
        fields.append(table[ikey])

    ist = aman.timestamps[0]
    ien = aman.timestamps[-1]
    res = load_az(ist, ien, fields, hkcfg_path)

    rmask = aman.flags.right_scan.mask()
    lmask = aman.flags.left_scan.mask()

    azts = aman.timestamps
    azdata = aman.boresight.az/DEG
    azmax = int(np.max(azdata)) + 1
    azmin = int(np.min(azdata)) - 1
    az_bin = np.arange(azmin, azmax, 0.2)
    
    ret = {}
    ret['az'] = az_bin
    for i in range(len(fields)):
        ifield = fields[i]
        ikey = keys[i]
        its, itemp = res.data[ifield]
        # select only within axis manager
        ifl = (its >= ist) & (its <= ien)
        its = its[ifl]
        itemp = itemp[ifl]

        # get corrected azimuths and indices
        iazs, idxs = get_az(its, azts, azdata)
        irmask = rmask[idxs]
        ilmask = lmask[idxs]
        binr = make_binning(itemp[irmask], iazs[irmask], az_bin)
        binl = make_binning(itemp[ilmask], iazs[ilmask], az_bin)
        ret[f'rcount_{ikey}'] = binr['count']
        ret[f'lcount_{ikey}'] = binl['count']
        ret[f'rsig_{ikey}'] = binr['sig']
        ret[f'rsig_sqre_{ikey}'] = binr['sig_sqre']
        ret[f'lsig_{ikey}'] = binl['sig']
        ret[f'lsig_sqre_{ikey}'] = binl['sig_sqre']

    savep = os.path.join(saved, f'{iobs}.pkl')
    save_pkl(ret, savep)
    
    del aman, res

def get_obsids(ctx, ys = 2024, ms = 1, ye = 2024, me = 2, overwrite = False, saved ='/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/az_temp/satp3/'):
    """Get observation IDs within given time range.
    Parameters:
    - ctx: sotodlib core context
    - ys: start year
    - ms: start month
    - ye: end year
    - me: end month
    Returns:
    - obss: numpy array of observation IDs
    """
    scan_start = datetime.datetime(ys, ms, 1, 0, 0, 0, 0, tzinfo=datetime.timezone.utc)
    scan_stop = datetime.datetime(ye, me, 1, 0, 0, 0, 0, tzinfo=datetime.timezone.utc)
    obslist = ctx.obsdb.query(f'timestamp > {scan_start.timestamp()} and timestamp < {scan_stop.timestamp()}')
    obss = []
    for iobs in obslist:
        if ctx.obsdb.get(iobs['obs_id'], tags=True)['subtype'] == 'cmb':
            obss.append(iobs['obs_id'])
    print(f'{scan_start} - {scan_stop}: # all obs = {len(obss)}')

    if not overwrite:
        obss_new = []
        for iobs in obss:
            savep = os.path.join(saved, f'{iobs}.pkl')
            if not os.path.exists(savep):
                obss_new.append(iobs)
        print(f'Overwrite is set to False, # all obs = {len(obss_new)}')
    else:
        obss_new = obss
        print('Overwrite is set to True, all obs will be processed again even if the file already exists.')
    return np.array(obss_new)

def main(ctx, ys = 2024, ms = 1, ye = 2024, me = 2,
         hkcfg_path = '/home/ys5857/workspace/script/so_exe/hkdata/hk_yaml/satp3_temp_az.yaml',
         saved ='/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/az_temp/'):
    obss = get_obsids(ctx, ys, ms, ye, me)
    for iobs in tqdm(obss):
        try:
            exe_azbin(ctx, iobs, hkcfg_path, saved)
        except Exception as e:
            print(e)

def test():
    hkcfg_path = '/home/ys5857/workspace/script/so_exe/hkdata/hk_yaml/satp3_temp_az.yaml'
    ctx_path3 = '/home/ys5857/workspace/script/so_exe/hkdata/contexts/satp3_contexts_20260109.yaml'
    ctx = core.Context(ctx_path3)
    iobs = 'obs_1735699616_satp3_1111111'
    exe_azbin(ctx, iobs, hkcfg_path, saved = '/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/az_temp/satp3/')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sat', type=str, default='satp3', help='Start year for analysis')
    parser.add_argument('--test', action='store_true', help='Run test function')
    args = parser.parse_args()

    if args.test:
        print('test')
        test()
    else:
        if args.sat == 'satp3':
            hkcfg_path = '/home/ys5857/workspace/script/so_exe/hkdata/hk_yaml/satp3_temp_az.yaml'
            ctx_path3 = '/home/ys5857/workspace/script/so_exe/hkdata/contexts/satp3_contexts_20260109.yaml'
            ctx = core.Context(ctx_path3)
            saved = f'/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/az_temp/{args.sat}/'
        elif args.sat == 'satp1':
            hkcfg_path = '/home/ys5857/workspace/script/so_exe/hkdata/hk_yaml/satp1_temp_az.yaml'
            ctx_path1 = '/home/ys5857/workspace/script/so_exe/hkdata/contexts/satp1_contexts_20260109.yaml'
            ctx = core.Context(ctx_path1)
            saved = f'/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/az_temp/{args.sat}/'
        else:
            raise ValueError('Platform not recognized, please use satp1 or satp3')

        if not os.path.exists(saved):
            os.makedirs(saved, exist_ok=True)

        print(ctx)
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
            [2025,9,2025,10],
            [2025,10,2025,11],
            [2025,11,2025,12],
            [2025,12,2026,1],
            [2025,1,2026,2],
            ]
        for ys, ms, ye, me in div_set:
            print(ys, ms, ye, me)
            main(ctx, ys, ms, ye, me, hkcfg_path, saved)