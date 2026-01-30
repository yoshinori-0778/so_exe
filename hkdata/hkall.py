import numpy as np
import pandas as pd
import datetime, sys, os
from sotodlib.io import hkdb
import requests
from io import StringIO

from pathlib import Path
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def read_hk_level3(ist, ien, hkcfg_file = None):
    '''
    hkcfg_file: path to yaml file for HK config
    '''
    if hkcfg_file is None:
        hkcfg_file = f'/home/ys5857/workspace/script/so_ana/meta/yaml/satp3_hk_all.yaml'
    cfg = hkdb.HkConfig.from_yaml(hkcfg_file)
    #cfg.aliases.keys()
    spec = hkdb.LoadSpec(
        cfg=cfg,
        fields=cfg.aliases.keys(),
        start=ist, end=ien,
    )
    res = hkdb.load_hk(spec, show_pb=True)

    fields = list(cfg.aliases.values())
    keys = list(cfg.aliases)
    ret= {}
    for ikey, ifield in zip(keys, fields):
        try:
            idata = res.data[ifield]
            ret[ikey] = {}
            ifl = (idata[0] > ist) & (idata[0] < ien)
            ret[ikey]['ts'] = idata[0][ifl]
            ret[ikey]['val'] = idata[1][ifl]
        except:
            print(f'There is not data: {ikey}')
            ret[ikey] = {}
            ret[ikey]['ts'] = np.nan
            ret[ikey]['val'] = np.nan

    return ret, keys

def read_hk_level3_tslist(tss, dt = 100, hkcfg_file = None):
    '''
    tss: list of timestamps for HK data you want
    dt: time duration. you will get mean value of data within [tss - dt, tss + dt]
    hkcfg_file: path to yaml file for HK config
    '''
    if hkcfg_file is None:
        hkcfg_file = '/home/ys5857/workspace/script/so_exe/hkdata/hk_yaml/satp3_hk_all.yaml'
    cfg = hkdb.HkConfig.from_yaml(hkcfg_file)
    cfg.aliases.keys()
    dfs = []
    for its in tss:
        try:
            spec = hkdb.LoadSpec(
                cfg=cfg,
                fields=cfg.aliases.keys(),
                start=its - dt, end=its + dt,
            )
            res = hkdb.load_hk(spec, show_pb=True)

            idf = pd.DataFrame()
            idf['ts'] = [its]
            for ikey in cfg.aliases.keys():
                iarg = cfg.aliases[ikey]
                if iarg in res.data.keys():
                    _, val = res.data[iarg]
                    if ikey == 'pwv_class':
                        ifl = (val > 0) & (val < 3)
                        idf[ikey] = [np.nanmean(val[ifl])]
                    else:
                        idf[ikey] = [np.nanmean(val)]
                else:
                    #print(f'There is not data: {ikey}')
                    idf[ikey] = [np.nan]
                    pass
            dfs.append(idf)
        except:
            pass
    return pd.concat(dfs).reset_index()

def read_hk_level3_each(its, dt = 100, hkcfg_file = None):
    if hkcfg_file is None:
        hkcfg_file = '/home/ys5857/workspace/script/so_exe/hkdata/hk_yaml/satp3_hk_all.yaml'
    cfg = hkdb.HkConfig.from_yaml(hkcfg_file)
    cfg.aliases.keys()
    spec = hkdb.LoadSpec(
        cfg=cfg,
        fields=cfg.aliases.keys(),
        start=its - dt, end=its + dt,
    )
    res = hkdb.load_hk(spec, show_pb=True)
    rets = []
    keys = []
    for ikey in cfg.aliases.keys():
        iarg = cfg.aliases[ikey]
        if iarg in res.data.keys():
            _, val = res.data[iarg]
            if ikey == 'pwv_class':
                ifl = (val > 0) & (val < 3)
                rets.append(np.nanmean(val[ifl]))
            else:
                rets.append(np.nanmean(val))
            rets.append(np.nanmean(val))
            keys.append(ikey)
        else:
            #print(f'There is not data: {ikey}')
            rets.append(np.nan)
            keys.append(np.nan)
            pass
    return rets, keys


def get_apex_data(start_date=datetime.datetime(2024,5,19),
                  end_date=datetime.datetime(2024,5,21)):
    """
    Get APEX weather data from the ESO archive.

    Parameters
    ----------
    start_date : datetime.datetime
        Start date for the data.
    end_date : datetime.datetime
        End date for the data.
    
    Returns
    -------
    outdata : dict
        Dictionary with keys 'timestamps' and 'pwv', which are lists of
        unix ctimestamps and precipitable water vapor values, respectively.
    """
    APEX_DATA_URL = 'http://archive.eso.org/wdb/wdb/eso/meteo_apex/query'

    request = requests.post(APEX_DATA_URL, data={
            'wdbo': 'csv/download',
            'max_rows_returned': 1000000,
            'start_date': start_date.strftime('%Y-%m-%dT%H:%M:%S') + '..' \
                + end_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'tab_pwv': 'on',
            'shutter': 'SHUTTER_OPEN',
            #'tab_shutter': 'on',
        })

    def date_converter(d):
        dt = datetime.datetime.fromisoformat(d)
        return dt.replace(tzinfo=datetime.timezone.utc)
    try:
        data = np.genfromtxt(
            StringIO(request.text),
            delimiter=',', skip_header=2,
            converters={0: date_converter},
            dtype=[('dates', datetime.datetime), ('pwv', float)],
        )
        
        outdata = {'timestamps':[d.timestamp() for d in data['dates']],
                'pwv':data['pwv']}
    except:
        outdata = {'timestamps':[], 'pwv':[]}
    return outdata

def main(ys, ms, ye, me, saved, dt=100, hkcfg_file = None, overwrite=False):
    savep = os.path.join(saved, f'hkdata_{ys}_{ms}_{ye}_{me}.pkl')
    if os.path.exists(savep) and (not overwrite):
        print(f'File exists: {savep}')
        return
    else:
        scan_start = datetime.datetime(ys, ms, 1, 0, 0, 0, 0, tzinfo=datetime.timezone.utc)
        scan_stop = datetime.datetime(ye, me, 1, 0, 0, 0, 0, tzinfo=datetime.timezone.utc)
        inverval = 3600*1 # sec

        if hkcfg_file is None:
            hkcfg_file = '/home/ys5857/workspace/script/so_exe/hkdata/hk_yaml/satp3_hk_all.yaml'

        tss = np.arange(scan_start.timestamp(), scan_stop.timestamp(), inverval)
        df = read_hk_level3_tslist(tss, dt, hkcfg_file=hkcfg_file)
        df['dt'] = [datetime.datetime.fromtimestamp(its, datetime.UTC) for its in df['ts'].values]
        df.to_pickle(savep)
        del df, tss
        return

def test1():
    print('test1')
    scan_start = datetime.datetime(2025, 7, 20, 0, 0, 0, 0, tzinfo=datetime.timezone.utc)
    scan_stop = datetime.datetime(2025, 7, 20, 5, 0, 0, 0, tzinfo=datetime.timezone.utc)
    inverval = 3600*1 # sec
    tss = np.arange(scan_start.timestamp(), scan_stop.timestamp(), inverval)

    hkcfg_file = '/home/ys5857/workspace/script/so_exe/hkdata/hk_yaml/satp3_hk_all.yaml'
    dt = 100 # sec
    df = read_hk_level3_tslist(tss, dt, hkcfg_file=hkcfg_file)
    df['dt'] = [datetime.datetime.fromtimestamp(its, datetime.UTC) for its in df['ts'].values]
    savep = f'/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/2025/07/hkdata/hkdata_test.pkl'
    df.to_pickle(savep)


if __name__ == '__main__':
    test = False
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sat', type=str, default='satp3', help='Start year for analysis')
    parser.add_argument('--dt', type=int, default=100, help='Time duration for averaging HK data (sec)')
    parser.add_argument('--hkcfg_file', type=str, default=None, help='Path to HK config yaml file')
    parser.add_argument('--test', action='store_true', help='Run test function')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing files')
    args = parser.parse_args()

    if args.test:
        test1()
    else:
        if args.hkcfg_file is not None:
            hkcfg_file = args.hkcfg_file
        else:
            if args.sat == 'satp3':
                hkcfg_file = '/home/ys5857/workspace/script/so_exe/hkdata/hk_yaml/satp3_hk_all.yaml'
                saved = '/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/hkdata/satp3/'
            elif args.sat == 'satp1':
                hkcfg_file = '/home/ys5857/workspace/script/so_exe/hkdata/hk_yaml/satp1_hk_all.yaml'
                saved = '/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/hkdata/satp1/'
            elif args.sat == 'site':
                hkcfg_file = '/home/ys5857/workspace/script/so_exe/hkdata/hk_yaml/site_hk.yaml'
                saved = '/scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/hkdata/site/'
            else:
                raise ValueError('sat must be satp3 or satp1')
        """
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
        """

        start = datetime.date(2023, 10, 1)
        end   = datetime.date(2026, 2, 1) 

        div_set = []
        cur = start
        while cur <= end:
            y1, m1 = cur.year, cur.month
            if m1 == 12:
                y2, m2 = y1 + 1, 1
            else:
                y2, m2 = y1, m1 + 1
            div_set.append([y1, m1, y2, m2])
            cur = datetime.date(y2, m2, 1)
        print(div_set)

        for ys, ms, ye, me in div_set:
            print(ys, ms, ye, me)
            main(ys, ms, ye, me, saved=saved,dt=args.dt, hkcfg_file = hkcfg_file, overwrite=args.overwrite)