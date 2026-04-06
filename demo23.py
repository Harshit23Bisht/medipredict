python -c "
import pandas as pd, os

files = {
    'patients':        ('hosp', 'patients'),
    'admissions':      ('hosp', 'admissions'),
    'diagnoses_icd':   ('hosp', 'diagnoses_icd'),
    'd_icd_diagnoses': ('hosp', 'd_icd_diagnoses'),
    'prescriptions':   ('hosp', 'prescriptions'),
    'labevents':       ('hosp', 'labevents'),
    'd_labitems':      ('hosp', 'd_labitems'),
    'icustays':        ('icu',  'icustays'),
    'chartevents':     ('icu',  'chartevents'),
    'd_items':         ('icu',  'd_items'),
}

RAW = 'data/raw'

for name, (folder, fname) in files.items():
    found = False
    for ext in ['.csv', '.csv.gz']:
        path = os.path.join(RAW, folder, fname + ext)
        if os.path.exists(path):
            size = os.path.getsize(path)/(1024*1024)
            cols = pd.read_csv(path, nrows=1).columns.tolist()
            print(f'{name:20s} {size:7.1f}MB  {cols}')
            found = True
            break
    if not found:
        print(f'{name:20s} NOT FOUND')
"