import datetime, os
import numpy as np
import requests
from io import StringIO


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
        return datetime.datetime.fromisoformat(d)

    data = np.genfromtxt(
        StringIO(request.text),
        delimiter=',', skip_header=2,
        converters={0: date_converter},
        dtype=[('dates', datetime.datetime), ('pwv', float)],
    )
    
    outdata = {'timestamps':[d.timestamp() for d in data['dates']],
               'pwv':data['pwv']}
    return outdata

def save_apex_data(sy, sm, ey, em,
                   saved='/scratch/gpfs/SIMONSOBS/users/ys5857/share/pwv_apex',
                   overwrite=False):

    start_date=datetime.datetime(sy,sm,1)
    end_date=datetime.datetime(ey,em,1)
    savep = os.path.join(saved, f'apex_pwv_{sy}_{sm}_{ey}_{em}.npz')
    if os.path.exists(savep) and (not overwrite):
        print(f'File exists: {savep}')
        return
    else:
        data = get_apex_data(start_date, end_date)
        np.savez(
                savep,
                ts=np.array(data['timestamps']),
                pwv=np.array(data['pwv']),
            )
        del data
        
if __name__ == '__main__':
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

    saved='/scratch/gpfs/SIMONSOBS/users/ys5857/share/pwv_apex'
    for ys, ms, ye, me in div_set:
        print(ys, ms, ye, me)
        save_apex_data(ys, ms, ye, me,
                        saved=saved)