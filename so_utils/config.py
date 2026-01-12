import numpy as np

BANDS = np.array(['f090', 'f150'])

WSS = np.array(['ws0', 'ws1', 'ws2', 'ws3', 'ws4', 'ws5', 'ws6'])

WAFER_SLOTS = {}
WAFER_SLOTS['satp1'] = {}
WAFER_SLOTS['satp3'] = {}

WAFER_SLOTS['satp1']['SCR4'] = {}
WAFER_SLOTS['satp1']['SCR5'] = {}
WAFER_SLOTS['satp1']['SCR6'] = {}

WAFER_SLOTS['satp3']['run12'] = {}
WAFER_SLOTS['satp3']['run13'] = {}
WAFER_SLOTS['satp3']['run14'] = {}

WAFER_SLOTS['satp1']['SCR4']['ws'] = np.array(['ws0', 'ws1', 'ws2', 'ws3', 'ws4', 'ws5', 'ws6'])
WAFER_SLOTS['satp1']['SCR4']['Mv'] = np.array(['Mv19', 'Mv18', 'Mv22', 'Mv29', 'Mv7', 'Mv9', 'Mv15'])
WAFER_SLOTS['satp1']['SCR4']['mv'] = np.array(['mv19', 'mv18', 'mv22', 'mv29', 'mv7', 'mv9', 'mv15'])

WAFER_SLOTS['satp1']['SCR5']['ws'] = np.array(['ws0', 'ws1', 'ws2', 'ws3', 'ws4', 'ws5', 'ws6'])
WAFER_SLOTS['satp1']['SCR5']['Mv'] = np.array(['Mv19', 'Mv48', 'Mv50', 'Mv22', 'Mv18', 'Mv52', 'Mv51'])
WAFER_SLOTS['satp1']['SCR5']['mv'] = np.array(['mv19', 'mv48', 'mv50', 'mv22', 'mv18', 'mv52', 'mv51'])

WAFER_SLOTS['satp1']['SCR6']['ws'] = np.array(['ws0', 'ws1', 'ws2', 'ws3', 'ws4', 'ws5', 'ws6'])
WAFER_SLOTS['satp1']['SCR6']['Mv'] = np.array(['Mv19', 'Mv48', 'Mv50', 'Mv22', 'Mv18', 'Mv52', 'Mv51'])
WAFER_SLOTS['satp1']['SCR6']['mv'] = np.array(['mv19', 'mv48', 'mv50', 'mv22', 'mv18', 'mv52', 'mv51'])

WAFER_SLOTS['satp3']['run12']['ws'] = np.array(['ws0', 'ws1', 'ws2', 'ws3', 'ws4', 'ws5', 'ws6'])
WAFER_SLOTS['satp3']['run12']['mv'] = np.array(['mv5', 'mv27', 'mv35', 'mv12', 'mv23', 'mv33', 'mv17'])
WAFER_SLOTS['satp3']['run12']['Mv'] = np.array(['Mv5', 'Mv27', 'Mv35', 'Mv12', 'Mv23', 'Mv33', 'Mv17'])

WAFER_SLOTS['satp3']['run13']['ws'] = np.array(['ws0', 'ws1', 'ws2', 'ws3', 'ws4', 'ws5', 'ws6'])
WAFER_SLOTS['satp3']['run13']['mv'] = np.array(['mv5', 'mv27', 'mv35', 'mv12', 'mv23', 'mv33', 'mv17'])
WAFER_SLOTS['satp3']['run13']['Mv'] = np.array(['Mv5', 'Mv27', 'Mv35', 'Mv12', 'Mv23', 'Mv33', 'Mv17'])

WAFER_SLOTS['satp3']['run14']['ws'] = np.array(['ws0', 'ws1', 'ws2', 'ws3', 'ws4', 'ws5', 'ws6'])
WAFER_SLOTS['satp3']['run14']['mv'] = np.array(['mv5', 'mv27', 'mv35', 'mv12', 'mv23', 'mv33', 'mv17'])
WAFER_SLOTS['satp3']['run14']['Mv'] = np.array(['Mv5', 'Mv27', 'Mv35', 'Mv12', 'Mv23', 'Mv33', 'Mv17'])

# plots
colors = np.array([
    "#F0E442", # yellow, ws0, mv5
    "#D55E00", # vermillion, ws1, mv27
    "#E69F00", # orange, ws2, mv35   
    "#009E73", # bluishgreen, ws3, mv12
    "#56B4E9", # skyblue, ws4, mv23
    "#0072B2", # blue, ws5, mv33
    "#CC79A7", # reddishpurple, ws6, mv17
])

DEG = np.pi / 180.
SITE = 'so_lat'
SITE1 = 'so_sat1'
SITE3 = 'so_sat3'
WEATHER = 'typical'

# Tiger3
CTX_PATH = {}
CTX_PATH['satp3'] = '/scratch/gpfs/SIMONSOBS/so/tracked/metadata/satp3/contexts/use_this_local.yaml'
CTX_PATH['satp1'] = '/scratch/gpfs/SIMONSOBS/so/tracked/metadata/satp1  a/contexts/use_this_local.yaml'