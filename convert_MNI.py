import glob
import pandas as pd
import ants
import json

datadir = '/home/arjitm/a/'
contfiles = glob.glob(f'{datadir}*/derivatives/*/registrations/Contacts_T1_space.csv')

for cf in contfiles:
    df = pd.read_csv(cf)
    pdir = cf.split('registrations')[0]
    df['x'] = df['x'] * -1
    df['y'] = df['y'] * -1
    tdf = ants.apply_transforms_to_points(dim=3,
                                          points=df,
                                          transformlist=[
                                              f'{pdir}transforms/t1_to_MNI_0GenericAffine.mat',
                                              f'{pdir}transforms/t1_to_MNI_1InverseWarp.nii.gz',
                                          ],
                                          whichtoinvert=[True, False]
                                          )
    tdf['x'] = tdf['x'] * -1
    tdf['y'] = tdf['y'] * -1
       
    try:
        cj = cf.replace("Contacts_T1_space.csv", "Contacts_CT_native.mrk.json")
        with open(cj, 'r') as ff:
            parsed = json.load(ff)
    except FileNotFoundError:
        cj = cf.replace("Contacts_T1_space.csv", "Contacts_CT_Native.mrk.json")
        with open(cj, 'r') as ff:
            parsed = json.load(ff)

    labels = []
    ids = []

    coords3d = list()
    labels = list()
    for cp in parsed.get('markups')[0].get('controlPoints'):
        coords3d.append(tuple(cp.get('position')))
        ids.append(cp.get('id'))
        labels.append(cp.get('label'))

    id_to_label = {int(k): v for k, v in zip(ids, labels)}

    tdf['name'] = [id_to_label.get(int(k)) for k in tdf['id']]
    tdf.to_csv(f'{pdir}registrations/Contacts_MNI_space.csv')

