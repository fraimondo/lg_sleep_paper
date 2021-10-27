import numpy as np
from collections import OrderedDict
# stages -> 'sleep stages'
# stages_groups -> 'groups of stages to be considered together for reductions'
# stage_regexp -> dict (regexp -> stage label)
# N1 Stages
stages = {'Stage0': 10, 'Stage0/mr0': 11, 'Stage0/mr1': 12,
          'Stage1': 13, 'Stage1/mr0': 14, 'Stage1/mr1': 15,
          'Stage2': 16, 'Stage2/mr0': 17, 'Stage2/mr1': 18,
          'Stage3': 19, 'Stage3/mr0': 20, 'Stage3/mr1': 21,
          'Stage4': 22, 'Stage4/mr0': 23, 'Stage4/mr1': 24,
          'Stage5': 25, 'Stage5/mr0': 26, 'Stage5/mr1': 27,
          'Stage6': 28, 'Stage6/mr0': 29, 'Stage6/mr1': 30,
          'Stage7': 31, 'Stage7/mr0': 32, 'Stage7/mr1': 33,
          'Stage8': 34, 'Stage8/mr0': 35, 'Stage8/mr1': 36,
          'Stage9': 37, 'Stage9/mr0': 38, 'Stage9/mr1': 39,
          'Stage10': 40, 'Stage10/mr0': 41, 'Stage10/mr1': 42,
          'Stage11': 43, 'Stage11/mr0': 44, 'Stage11/mr1': 45,
          'Stage12': 46, 'Stage12/mr0': 47, 'Stage12/mr1': 48,
          'Stage13': 49, 'Stage13/mr0': 50, 'Stage13/mr1': 51,
          'Stage14': 52, 'Stage14/mr0': 53, 'Stage14/mr1': 54,
          'Stage15': 55, 'Stage15/mr0': 56, 'Stage15/mr1': 57,
          'Stage16': 58, 'Stage16/mr0': 59, 'Stage16/mr1': 60
          }

stage_regexp = {}

for i in range(17):
    t_stage = 'Stage{}'.format(i)
    stage_regexp[fr"\S*_{t_stage}_(?!\S*(mr0|mr1))\S*"] = t_stage
    stage_regexp[fr"\S*_{t_stage}_\S*mr0\S*"] = f'{t_stage}/mr0'
    stage_regexp[fr"\S*_{t_stage}_\S*mr1\S*"] = f'{t_stage}/mr1'


stage_groups = {
    'Group0': ['Stage0', 'Stage0/mr0', 'Stage0/mr1'],
    'Group1': ['Stage1', 'Stage1/mr0', 'Stage1/mr1',
               'Stage2', 'Stage2/mr0', 'Stage2/mr1',
               'Stage3', 'Stage3/mr0', 'Stage3/mr1'],
    'Group2': ['Stage4', 'Stage4/mr0', 'Stage4/mr1'],
    'Group3': ['Stage5', 'Stage5/mr0', 'Stage5/mr1'],
    'Group4': ['Stage6', 'Stage6/mr0', 'Stage6/mr1',
               'Stage7', 'Stage7/mr0', 'Stage7/mr1',
               'Stage8', 'Stage8/mr0', 'Stage8/mr1'],
    'Group5': ['Stage9', 'Stage9/mr0', 'Stage9/mr1',
               'Stage10', 'Stage10/mr0', 'Stage10/mr1',
               'Stage12', 'Stage12/mr0', 'Stage12/mr1',
               'Stage13', 'Stage13/mr0', 'Stage13/mr1',
               'Stage16', 'Stage16/mr0', 'Stage16/mr1']
}

hori_groups = {
    'W': ['LG1/nmr', 'LG1/mr', 'LG3/nmr', 'LG3/mr'],
    'H0': ['Stage0', 'Stage0/mr0', 'Stage0/mr1'],
    'H1': ['Stage1', 'Stage1/mr0', 'Stage1/mr1'],
    'H2': ['Stage2', 'Stage2/mr0', 'Stage2/mr1'],
    'H3': ['Stage3', 'Stage3/mr0', 'Stage3/mr1'],
    'H4': ['Stage4', 'Stage4/mr0', 'Stage4/mr1'],
    'H5': ['Stage5', 'Stage5/mr0', 'Stage5/mr1'],
    'H6to8': ['Stage6', 'Stage6/mr0', 'Stage6/mr1',
              'Stage7', 'Stage7/mr0', 'Stage7/mr1',
              'Stage8', 'Stage8/mr0', 'Stage8/mr1'],
    'N2': ['N2/new', 'N2/old', 'Stage9', 'Stage10', 'Stage12',
           'Stage13', 'Stage16']
}

mr_groups = {
    'Awake_MR0': ['LG1/mr/mr0', 'LG3/mr/mr0'],
    'Awake_MR1': ['LG1/mr/mr1', 'LG3/mr/mr1'],
    'Group0_MR0': ['Stage0/mr0'],
    'Group0_MR1': ['Stage0/mr1'],
    'Group1_MR0': ['Stage1/mr0', 'Stage2/mr0', 'Stage3/mr0'],
    'Group1_MR1': ['Stage1/mr1', 'Stage2/mr1', 'Stage3/mr1'],
    'Group2_MR0': ['Stage4/mr0'],
    'Group2_MR1': ['Stage4/mr1'],
    'Group3_MR0': ['Stage5/mr0'],
    'Group3_MR1': ['Stage5/mr1'],
    'Group4_MR0': ['Stage6/mr0', 'Stage7/mr0', 'Stage8/mr0'],
    'Group4_MR1': ['Stage6/mr1', 'Stage7/mr1', 'Stage8/mr1'],
    'Group5_MR0': ['Stage9/mr0', 'Stage10/mr0', 'Stage12/mr0', 'Stage13/mr0',
                   'Stage16/mr0'],
    'Group5_MR1': ['Stage9/mr1', 'Stage10/mr1', 'Stage12/mr1', 'Stage13/mr1',
                   'Stage16/mr1']
}

hori_mr_groups = {
    'Awake_MR0': ['LG1/mr/mr0', 'LG3/mr/mr0'],
    'Awake_MR1': ['LG1/mr/mr1', 'LG3/mr/mr1'],
    'H0_MR0': ['Stage0/mr0'],
    'H0_MR1': ['Stage0/mr1'],
    'H1_MR0': ['Stage1/mr0'],
    'H1_MR1': ['Stage1/mr1'],
    'H2_MR0': ['Stage2/mr0'],
    'H2_MR1': ['Stage2/mr1'],
    'H3_MR0': ['Stage3/mr0'],
    'H3_MR1': ['Stage3/mr1'],
    'H4_MR0': ['Stage4/mr0'],
    'H4_MR1': ['Stage4/mr1'],
    'H5_MR0': ['Stage5/mr0'],
    'H5_MR1': ['Stage5/mr1'],
    'H6to8_MR0': ['Stage6/mr0', 'Stage7/mr0', 'Stage8/mr0'],
    'H6to8_MR1': ['Stage6/mr1', 'Stage7/mr1', 'Stage8/mr1'],
}

# Awake -> LG1 and LG3
stages['LG1/nmr'] = 71
stages['LG1/mr'] = 72
stages['LG1/mr/mr0'] = 73
stages['LG1/mr/mr1'] = 74
stages['LG3/nmr'] = 75
stages['LG3/mr'] = 76
stages['LG3/mr/mr0'] = 77
stages['LG3/mr/mr1'] = 78

stage_regexp[r"\S*_LG1_\S*_new0_\S*"] = 'LG1/nmr'
stage_regexp[r"\S*_LG1_\S*_new1_\S*"] = 'LG1/mr'
stage_regexp[r"\S*_LG1_\S*_old0_\S*"] = 'LG1/nmr'
stage_regexp[r"\S*_LG1_\S*_old1_\S*"] = 'LG1/mr'
stage_regexp[r"\S*_LG3_\S*_new0_\S*"] = 'LG3/nmr'
stage_regexp[r"\S*_LG3_\S*_new1_\S*"] = 'LG3/mr'
stage_regexp[r"\S*_LG3_\S*_old0_\S*"] = 'LG3/nmr'
stage_regexp[r"\S*_LG3_\S*_old1_\S*"] = 'LG3/mr'

stage_groups['W'] = ['LG1/nmr', 'LG1/mr', 'LG3/nmr', 'LG3/mr']
# stage_groups['LG1'] = ['LG1/nmr', 'LG1/mr']
# stage_groups['LG3'] = ['LG3/nmr', 'LG3/mr']
# stage_groups['LG1_MR0'] = ['LG1/nmr']
# stage_groups['LG1_MR1'] = ['LG1/mr']
# stage_groups['LG3_MR0'] = ['LG3/nmr']
# stage_groups['LG3_MR1'] = ['LG3/mr']

# N2
stages['N2/new'] = 81
stages['N2/old'] = 82

stage_regexp[r"\S*_N2_\S*_new0_\S*"] = 'N2/new'
stage_regexp[r"\S*_N2_\S*_new1_\S*"] = 'N2/new'
stage_regexp[r"\S*_N2_\S*_old0_\S*"] = 'N2/old'
stage_regexp[r"\S*_N2_\S*_old1_\S*"] = 'N2/old'

stage_groups['N2'] = ['N2/new', 'N2/old']
stage_groups['N2_G5'] = ['N2/new', 'N2/old', 'Stage9', 'Stage10', 'Stage12',
                         'Stage13', 'Stage16']
# stage_groups['N2_NEW'] = ['N2/new']
# stage_groups['N2_OLD'] = ['N2/old']

# REM -> Separate in phasic and tonic

stages['REM/phasic'] = 91
stages['REM/tonic'] = 92

stage_regexp[r"\S*_REM_\S*_rem1_\S*"] = 'REM/phasic'
stage_regexp[r"\S*_REM_\S*_rem0_\S*"] = 'REM/tonic'

stage_groups['REM'] = ['REM/phasic', 'REM/tonic']
stage_groups['REM_phasic'] = ['REM/phasic']
stage_groups['REM_tonic'] = ['REM/tonic']

# SWS -> Only SWS
stages['SWS'] = 99
stage_regexp['\\S*_SWS_\\S*'] = 'SWS'
stage_groups['SWS'] = ['SWS']


################
# Now for every stage, we can have local or global stuff.
lg_regexp = {
    r"\S*_GSLS_\S*": 'GSLS',
    r"\S*_GSLD_\S*": 'GSLD',
    r"\S*_GDLD_\S*": 'GDLD',
    r"\S*_GDLS_\S*": 'GDLS',
}
lg_trials = {
    'GSLS': 100,
    'GDLD': 200,
    'GSLD': 300,
    'GDLS': 400,
}

sleep_lg_event_id = {
    f'{st_k}/{lg_k}': st_v + lg_v for st_k, st_v in stages.items()
    for lg_k, lg_v in lg_trials.items()
}

reverse_sleep_lg_event_id = {
    v: k for k, v in sleep_lg_event_id.items()
}
#################

meg_ch_names = [
    'MEG 0113', 'MEG 0112', 'MEG 0111', 'MEG 0122', 'MEG 0123', 'MEG 0121',
    'MEG 0132', 'MEG 0133', 'MEG 0131', 'MEG 0143', 'MEG 0142', 'MEG 0141',
    'MEG 0213', 'MEG 0212', 'MEG 0211', 'MEG 0222', 'MEG 0223', 'MEG 0221',
    'MEG 0232', 'MEG 0233', 'MEG 0231', 'MEG 0243', 'MEG 0242', 'MEG 0241',
    'MEG 0313', 'MEG 0312', 'MEG 0311', 'MEG 0322', 'MEG 0323', 'MEG 0321',
    'MEG 0333', 'MEG 0332', 'MEG 0331', 'MEG 0343', 'MEG 0342', 'MEG 0341',
    'MEG 0413', 'MEG 0412', 'MEG 0411', 'MEG 0422', 'MEG 0423', 'MEG 0421',
    'MEG 0432', 'MEG 0433', 'MEG 0431', 'MEG 0443', 'MEG 0442', 'MEG 0441',
    'MEG 0513', 'MEG 0512', 'MEG 0511', 'MEG 0523', 'MEG 0522', 'MEG 0521',
    'MEG 0532', 'MEG 0533', 'MEG 0531', 'MEG 0542', 'MEG 0543', 'MEG 0541',
    'MEG 0613', 'MEG 0612', 'MEG 0611', 'MEG 0622', 'MEG 0623', 'MEG 0621',
    'MEG 0633', 'MEG 0632', 'MEG 0631', 'MEG 0642', 'MEG 0643', 'MEG 0641',
    'MEG 0713', 'MEG 0712', 'MEG 0711', 'MEG 0723', 'MEG 0722', 'MEG 0721',
    'MEG 0733', 'MEG 0732', 'MEG 0731', 'MEG 0743', 'MEG 0742', 'MEG 0741',
    'MEG 0813', 'MEG 0812', 'MEG 0811', 'MEG 0822', 'MEG 0823', 'MEG 0821',
    'MEG 0913', 'MEG 0912', 'MEG 0911', 'MEG 0923', 'MEG 0922', 'MEG 0921',
    'MEG 0932', 'MEG 0933', 'MEG 0931', 'MEG 0942', 'MEG 0943', 'MEG 0941',
    'MEG 1013', 'MEG 1012', 'MEG 1011', 'MEG 1023', 'MEG 1022', 'MEG 1021',
    'MEG 1032', 'MEG 1033', 'MEG 1031', 'MEG 1043', 'MEG 1042', 'MEG 1041',
    'MEG 1112', 'MEG 1113', 'MEG 1111', 'MEG 1123', 'MEG 1122', 'MEG 1121',
    'MEG 1133', 'MEG 1132', 'MEG 1131', 'MEG 1142', 'MEG 1143', 'MEG 1141',
    'MEG 1213', 'MEG 1212', 'MEG 1211', 'MEG 1223', 'MEG 1222', 'MEG 1221',
    'MEG 1232', 'MEG 1233', 'MEG 1231', 'MEG 1243', 'MEG 1242', 'MEG 1241',
    'MEG 1312', 'MEG 1313', 'MEG 1311', 'MEG 1323', 'MEG 1322', 'MEG 1321',
    'MEG 1333', 'MEG 1332', 'MEG 1331', 'MEG 1342', 'MEG 1343', 'MEG 1341',
    'MEG 1412', 'MEG 1413', 'MEG 1411', 'MEG 1423', 'MEG 1422', 'MEG 1421',
    'MEG 1433', 'MEG 1432', 'MEG 1431', 'MEG 1442', 'MEG 1443', 'MEG 1441',
    'MEG 1512', 'MEG 1513', 'MEG 1511', 'MEG 1522', 'MEG 1523', 'MEG 1521',
    'MEG 1533', 'MEG 1532', 'MEG 1531', 'MEG 1543', 'MEG 1542', 'MEG 1541',
    'MEG 1613', 'MEG 1612', 'MEG 1611', 'MEG 1622', 'MEG 1623', 'MEG 1621',
    'MEG 1632', 'MEG 1633', 'MEG 1631', 'MEG 1643', 'MEG 1642', 'MEG 1641',
    'MEG 1713', 'MEG 1712', 'MEG 1711', 'MEG 1722', 'MEG 1723', 'MEG 1721',
    'MEG 1732', 'MEG 1733', 'MEG 1731', 'MEG 1743', 'MEG 1742', 'MEG 1741',
    'MEG 1813', 'MEG 1812', 'MEG 1811', 'MEG 1822', 'MEG 1823', 'MEG 1821',
    'MEG 1832', 'MEG 1833', 'MEG 1831', 'MEG 1843', 'MEG 1842', 'MEG 1841',
    'MEG 1912', 'MEG 1913', 'MEG 1911', 'MEG 1923', 'MEG 1922', 'MEG 1921',
    'MEG 1932', 'MEG 1933', 'MEG 1931', 'MEG 1943', 'MEG 1942', 'MEG 1941',
    'MEG 2013', 'MEG 2012', 'MEG 2011', 'MEG 2023', 'MEG 2022', 'MEG 2021',
    'MEG 2032', 'MEG 2033', 'MEG 2031', 'MEG 2042', 'MEG 2043', 'MEG 2041',
    'MEG 2113', 'MEG 2112', 'MEG 2111', 'MEG 2122', 'MEG 2123', 'MEG 2121',
    'MEG 2133', 'MEG 2132', 'MEG 2131', 'MEG 2143', 'MEG 2142', 'MEG 2141',
    'MEG 2212', 'MEG 2213', 'MEG 2211', 'MEG 2223', 'MEG 2222', 'MEG 2221',
    'MEG 2233', 'MEG 2232', 'MEG 2231', 'MEG 2242', 'MEG 2243', 'MEG 2241',
    'MEG 2312', 'MEG 2313', 'MEG 2311', 'MEG 2323', 'MEG 2322', 'MEG 2321',
    'MEG 2332', 'MEG 2333', 'MEG 2331', 'MEG 2343', 'MEG 2342', 'MEG 2341',
    'MEG 2412', 'MEG 2413', 'MEG 2411', 'MEG 2423', 'MEG 2422', 'MEG 2421',
    'MEG 2433', 'MEG 2432', 'MEG 2431', 'MEG 2442', 'MEG 2443', 'MEG 2441',
    'MEG 2512', 'MEG 2513', 'MEG 2511', 'MEG 2522', 'MEG 2523', 'MEG 2521',
    'MEG 2533', 'MEG 2532', 'MEG 2531', 'MEG 2543', 'MEG 2542', 'MEG 2541',
    'MEG 2612', 'MEG 2613', 'MEG 2611', 'MEG 2623', 'MEG 2622', 'MEG 2621',
    'MEG 2633', 'MEG 2632', 'MEG 2631', 'MEG 2642', 'MEG 2643', 'MEG 2641'
]

meg_ch_type = ['mag' if x.endswith('1') else 'grad' for x in meg_ch_names]


plot3d_rois = OrderedDict()

plot3d_rois['Front Right'] = {
    'idx': [
        'MEG 0811', 'MEG 0812', 'MEG 0813',
        'MEG 0911', 'MEG 0912', 'MEG 0913',
        'MEG 0921', 'MEG 0922', 'MEG 0923',
        'MEG 0931', 'MEG 0932', 'MEG 0933',
        'MEG 0941', 'MEG 0942', 'MEG 0943',
        'MEG 1011', 'MEG 1012', 'MEG 1013',
        'MEG 1021', 'MEG 1022', 'MEG 1023',
        'MEG 1031', 'MEG 1032', 'MEG 1033',
        'MEG 1211', 'MEG 1212', 'MEG 1213',
        'MEG 1221', 'MEG 1222', 'MEG 1223',
        'MEG 1231', 'MEG 1232', 'MEG 1233',
        'MEG 1241', 'MEG 1242', 'MEG 1243',
        'MEG 1411', 'MEG 1412', 'MEG 1413'
    ],
    'outline': [
        # 'MEG 0811',
        'MEG 0911',
        'MEG 0921',
        # 'MEG 0931',
        'MEG 0941',
        # 'MEG 1011',
        'MEG 1021',
        'MEG 1031',
        'MEG 1211',
        'MEG 1221',
        'MEG 1231',
        'MEG 1241',
        'MEG 1411',
    ],
    'center': 'MEG 0931'
}

plot3d_rois['Front Left'] = {
    'idx': [
        'MEG 0121', 'MEG 0122', 'MEG 0123',
        'MEG 0311', 'MEG 0312', 'MEG 0313',
        'MEG 0321', 'MEG 0322', 'MEG 0323',
        'MEG 0331', 'MEG 0332', 'MEG 0333',
        'MEG 0341', 'MEG 0342', 'MEG 0343',
        'MEG 0511', 'MEG 0512', 'MEG 0513',
        'MEG 0521', 'MEG 0522', 'MEG 0523',
        'MEG 0531', 'MEG 0532', 'MEG 0533',
        'MEG 0541', 'MEG 0542', 'MEG 0543',
        'MEG 0611', 'MEG 0612', 'MEG 0613',
        'MEG 0621', 'MEG 0622', 'MEG 0623',
        'MEG 0641', 'MEG 0642', 'MEG 0643',
        'MEG 0821', 'MEG 0822', 'MEG 0823'
    ],
    'outline': [
        'MEG 0121',
        'MEG 0311',
        'MEG 0321',
        'MEG 0331',
        'MEG 0341',
        'MEG 0511',
        'MEG 0521',
        'MEG 0531',
        # 'MEG 0541',
        'MEG 0611',
        # 'MEG 0621',
        'MEG 0641',
        # 'MEG 0821',
    ],
    'center': 'MEG 0541'
}

plot3d_rois['Temporal Left'] = {
    'idx': [
        'MEG 0111', 'MEG 0112', 'MEG 0113',
        'MEG 0221', 'MEG 0222', 'MEG 0223',
        'MEG 0211', 'MEG 0212', 'MEG 0213',
        'MEG 0131', 'MEG 0132', 'MEG 0133',
        'MEG 0141', 'MEG 0142', 'MEG 0143',
        'MEG 0231', 'MEG 0232', 'MEG 0233',
        'MEG 0241', 'MEG 0242', 'MEG 0243',
        'MEG 1511', 'MEG 1512', 'MEG 1513',
        'MEG 1621', 'MEG 1622', 'MEG 1623',
        'MEG 1611', 'MEG 1612', 'MEG 1613',
        'MEG 1541', 'MEG 1542', 'MEG 1543',
        'MEG 1521', 'MEG 1522', 'MEG 1523',
        'MEG 1531', 'MEG 1532', 'MEG 1533',

    ],
    'outline': [
        'MEG 0111',
        'MEG 0221',
        'MEG 0211',
        'MEG 0131',
        'MEG 0141',
        'MEG 0231',
        # 'MEG 0241',
        # 'MEG 1511',
        'MEG 1621',
        'MEG 1611',
        'MEG 1541',
        'MEG 1521',
        'MEG 1531',
    ],
    'center': 'MEG 0241'
}

plot3d_rois['Temporal Right'] = {
    'idx': [
        'MEG 1421', 'MEG 1422', 'MEG 1423',
        'MEG 1311', 'MEG 1312', 'MEG 1313',
        'MEG 1321', 'MEG 1322', 'MEG 1323',
        'MEG 1441', 'MEG 1442', 'MEG 1443',
        'MEG 1431', 'MEG 1432', 'MEG 1433',
        'MEG 1341', 'MEG 1342', 'MEG 1343',
        'MEG 1331', 'MEG 1332', 'MEG 1333',
        'MEG 2611', 'MEG 2612', 'MEG 2613',
        'MEG 2411', 'MEG 2412', 'MEG 2413',
        'MEG 2421', 'MEG 2422', 'MEG 2423',
        'MEG 2621', 'MEG 2622', 'MEG 2623',
        'MEG 2641', 'MEG 2642', 'MEG 2643',
        'MEG 2631', 'MEG 2632', 'MEG 2633',
    ],
    'outline': [
        'MEG 1421',
        'MEG 1311',
        'MEG 1321',
        'MEG 1441',
        'MEG 1431',
        'MEG 1341',
        'MEG 1331',
        # 'MEG 2611',
        'MEG 2411',
        'MEG 2421',
        'MEG 2621',
        'MEG 2641',
        'MEG 2631',
    ],
    'center': 'MEG 1331'
}

plot3d_rois['Parietal Left'] = {
    'idx': [
        'MEG 0411', 'MEG 0412', 'MEG 0413',
        'MEG 0421', 'MEG 0422', 'MEG 0423',
        'MEG 0631', 'MEG 0632', 'MEG 0633',
        'MEG 0441', 'MEG 0442', 'MEG 0443',
        'MEG 0431', 'MEG 0432', 'MEG 0433',
        'MEG 0711', 'MEG 0712', 'MEG 0713',
        'MEG 1811', 'MEG 1812', 'MEG 1813',
        'MEG 1821', 'MEG 1822', 'MEG 1823',
        'MEG 0741', 'MEG 0742', 'MEG 0743',
        'MEG 1831', 'MEG 1832', 'MEG 1833',
        'MEG 1841', 'MEG 1842', 'MEG 1843',
        'MEG 1631', 'MEG 1632', 'MEG 1633',
        'MEG 2011', 'MEG 2012', 'MEG 2013',

    ],
    'outline': [
        'MEG 0411',
        'MEG 0421',
        'MEG 0631',
        'MEG 0441',
        # 'MEG 0431',
        'MEG 0711',
        'MEG 1811',
        # 'MEG 1821',
        'MEG 0741',
        'MEG 1831',
        'MEG 1841',
        'MEG 1631',
        'MEG 2011',
    ],
    'center': 'MEG 0431'
}

plot3d_rois['Parietal Right'] = {
    'idx': [
        'MEG 1041', 'MEG 1042', 'MEG 1043',
        'MEG 1111', 'MEG 1112', 'MEG 1113',
        'MEG 1121', 'MEG 1122', 'MEG 1123',
        'MEG 0721', 'MEG 0722', 'MEG 0723',
        'MEG 1141', 'MEG 1142', 'MEG 1143',
        'MEG 1131', 'MEG 1132', 'MEG 1133',
        'MEG 0731', 'MEG 0732', 'MEG 0733',
        'MEG 2211', 'MEG 2212', 'MEG 2213',
        'MEG 2221', 'MEG 2222', 'MEG 2223',
        'MEG 2241', 'MEG 2242', 'MEG 2243',
        'MEG 2231', 'MEG 2232', 'MEG 2233',
        'MEG 2441', 'MEG 2442', 'MEG 2443',
        'MEG 2021', 'MEG 2022', 'MEG 2023',

    ],
    'outline': [
        'MEG 1041',
        'MEG 1111',
        'MEG 1121',
        'MEG 0721',
        # 'MEG 1141',
        'MEG 1131',
        'MEG 0731',
        # 'MEG 2211',
        'MEG 2221',
        'MEG 2241',
        'MEG 2231',
        'MEG 2441',
        'MEG 2021',
    ],
    'center': 'MEG 1141'
}

plot3d_rois['Occipital Left'] = {
    'idx': [
        'MEG 1641', 'MEG 1642', 'MEG 1643',
        'MEG 1911', 'MEG 1912', 'MEG 1913',
        'MEG 2041', 'MEG 2042', 'MEG 2043',
        'MEG 1721', 'MEG 1722', 'MEG 1723',
        'MEG 1941', 'MEG 1942', 'MEG 1943',
        'MEG 1921', 'MEG 1922', 'MEG 1923',
        'MEG 2111', 'MEG 2112', 'MEG 2113',
        'MEG 1711', 'MEG 1712', 'MEG 1713',
        'MEG 1731', 'MEG 1732', 'MEG 1733',
        'MEG 1931', 'MEG 1932', 'MEG 1933',
        'MEG 1741', 'MEG 1742', 'MEG 1743',
        'MEG 2141', 'MEG 2142', 'MEG 2143'
    ],
    'outline': [
        'MEG 1641',
        'MEG 1911',
        'MEG 2041',
        'MEG 1721',
        'MEG 1941',
        'MEG 1921',
        # 'MEG 2111',
        'MEG 1711',
        # 'MEG 1731',
        'MEG 1931',
        'MEG 1741',
        'MEG 2141',
    ],
    'center': 'MEG 1731'
}

plot3d_rois['Occipital Right'] = {
    'idx': [
        'MEG 2031', 'MEG 2032', 'MEG 2033',
        'MEG 2311', 'MEG 2312', 'MEG 2313',
        'MEG 2431', 'MEG 2432', 'MEG 2433',
        'MEG 2341', 'MEG 2342', 'MEG 2343',
        'MEG 2321', 'MEG 2322', 'MEG 2323',
        'MEG 2521', 'MEG 2522', 'MEG 2523',
        'MEG 2121', 'MEG 2122', 'MEG 2123',
        'MEG 2331', 'MEG 2332', 'MEG 2333',
        'MEG 2511', 'MEG 2512', 'MEG 2513',
        'MEG 2131', 'MEG 2132', 'MEG 2133',
        'MEG 2541', 'MEG 2542', 'MEG 2543',
        'MEG 2531', 'MEG 2532', 'MEG 2533'
    ],
    'outline': [
        'MEG 2031',
        'MEG 2311',
        'MEG 2431',
        'MEG 2341',
        'MEG 2321',
        'MEG 2521',
        # 'MEG 2121',
        'MEG 2331',
        # 'MEG 2511',
        'MEG 2131',
        'MEG 2541',
        'MEG 2531',
    ],
    'center': 'MEG 2511'
}

_mmn = ['MEG 1621', 'MEG 1811', 'MEG 1631', 'MEG 2411', 'MEG 2221', 'MEG 2441',
        'MEG 0242', 'MEG 0243', 'MEG 0233', 'MEG 0232', 'MEG 1642', 'MEG 1643',
        'MEG 1332', 'MEG 1333', 'MEG 1343', 'MEG 1342', 'MEG 2422', 'MEG 2423']

_p3a = ['MEG 1611', 'MEG 1621', 'MEG 1811', 'MEG 1631', 'MEG 0231', 'MEG 2421',
        'MEG 2411', 'MEG 2221', 'MEG 1341', 'MEG 2441',
        'MEG 0242', 'MEG 0243', 'MEG 0233', 'MEG 0232', 'MEG 0442', 'MEG 0443',
        'MEG 1332', 'MEG 1333', 'MEG 1342', 'MEG 1343', 'MEG 1132', 'MEG 1133']

_p3b = ['MEG 0211', 'MEG 0341', 'MEG 0131', 'MEG 0121',
        'MEG 1322', 'MEG 1323', 'MEG 1312', 'MEG 1313', 'MEG 1412', 'MEG 1413',
        'MEG 1342', 'MEG 1343',
        'MEG 0612', 'MEG 0613', 'MEG 1012', 'MEG 1013', 'MEG 1022', 'MEG 1023',
        'MEG 0822', 'MEG 0823', 'MEG 0622', 'MEG 0623']

sleep_meg_rois = {
    'p3a': np.array([meg_ch_names.index(x) for x in _p3a]),
    'p3b': np.array([meg_ch_names.index(x) for x in _p3b]),
    'mmn': np.array([meg_ch_names.index(x) for x in _mmn]),
    'cnv': np.array([meg_ch_names.index(x) for x in _mmn]),

    'Fz': np.array([103, 104, 102, 70, 69, 71, 64, 63, 65, 109, 110, 108]),
    'Cz': np.array([67, 68, 66, 112, 111, 113, 73, 74, 72, 76, 77, 75]),
    'Pz': np.array([205, 206, 204, 256, 257, 255, 223, 224, 222, 226, 227,
                    225]),
    'scalp': np.arange(306),
    'nonscalp': None
}
