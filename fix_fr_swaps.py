"""
Fix script: apply all swap operations that failed in apply_fr_verdicts.py.
Logic: for each item that needs swap, check current state and swap only if wrong direction.
"""
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
df = pd.read_csv('data/stimuli_seed.csv', encoding='utf-8')

def do_swap(iid):
    idx = df[df['item_id']==iid].index
    if len(idx)==0: print(f'WARN: {iid} not found'); return
    i = idx[0]
    s = df.at[i,'sent_stereotype']
    a = df.at[i,'sent_anti_stereotype']
    df.at[i,'sent_stereotype'] = a
    df.at[i,'sent_anti_stereotype'] = s

def upd(iid, **kw):
    idx = df[df['item_id']==iid].index
    if len(idx)==0: return
    i = idx[0]
    for k,v in kw.items(): df.at[i,k] = v

# ── Items where sent_stereotype should have FEMININE form ────────────────────
# (warmth femme/homme: stereotype direction = woman stereotyped as warmer)
# Check: if sent_stereotype is SHORTER than sent_anti_stereotype, masculine is in S → swap
def fix_if_masc_in_S(iid, **extra):
    idx = df[df['item_id']==iid].index
    if len(idx)==0: return
    i = idx[0]
    s = str(df.at[i,'sent_stereotype'])
    a = str(df.at[i,'sent_anti_stereotype'])
    if len(s) <= len(a):   # masculine (shorter/equal) in S → swap to put feminine in S
        do_swap(iid)
    for k,v in extra.items(): df.at[i,k] = v

# Warmth femme/homme — feminine should be in sent_stereotype
for iid in [
    # 121-150
    'FR-G-121','FR-G-128','FR-G-134','FR-G-139','FR-G-142','FR-G-143','FR-G-145',
    # 151-180
    'FR-G-151','FR-G-155','FR-G-156','FR-G-177',
    # 181-210
    'FR-G-183','FR-G-187','FR-G-193','FR-G-207','FR-G-208','FR-G-209',
    # 211-240
    'FR-G-223','FR-G-227','FR-G-228',
    # 241-270
    'FR-G-241','FR-G-249','FR-G-255','FR-G-268',
    # 271-300
    'FR-G-283','FR-G-287','FR-G-289','FR-G-298',
    # 301-330
    'FR-G-313','FR-G-318','FR-G-323','FR-G-324',
    # 331-360
    'FR-G-343','FR-G-344','FR-G-354',
    # 361-390
    'FR-G-362','FR-G-379','FR-G-383',
    # 391-420
    'FR-G-396','FR-G-397','FR-G-419',
    # 421-450
    'FR-G-433',
    # 451-480
    'FR-G-473','FR-G-477',
    # 481-510
    'FR-G-487','FR-G-501','FR-G-504',
    # 511-540
    'FR-G-521','FR-G-522','FR-G-524','FR-G-526',
    # 541-555
    'FR-G-542','FR-G-544','FR-G-548','FR-G-553',
]:
    fix_if_masc_in_S(iid)

# Complex warmth+dim swaps — feminine should end up in sent_stereotype
for iid, dim in [
    ('FR-G-161', 'warmth'),
    ('FR-G-222', 'warmth'),
    ('FR-G-243', 'warmth'),
    ('FR-G-290', 'warmth'),
    ('FR-G-358', 'warmth'),
    ('FR-G-484', 'warmth'),
    ('FR-G-506', 'warmth'),
    ('FR-G-536', 'warmth'),
]:
    fix_if_masc_in_S(iid, dimension=dim)

# Warmth, panic/flee type — feminine after swap
for iid in ['FR-G-351']:
    fix_if_masc_in_S(iid)

# Physical weakness / competence femme/homme — feminine should be in sent_stereotype
for iid in [
    'FR-G-169','FR-G-277','FR-G-293','FR-G-346',
    'FR-G-363','FR-G-393','FR-G-491','FR-G-552',
    'FR-G-234','FR-G-239','FR-G-245','FR-G-403',
]:
    fix_if_masc_in_S(iid)

# ── Items where sent_stereotype should have MASCULINE form ───────────────────
# (competence or low-warmth homme/femme: masculine = stereotype)
# Check: if sent_stereotype is LONGER, feminine is in S → swap
def fix_if_fem_in_S(iid, **extra):
    idx = df[df['item_id']==iid].index
    if len(idx)==0: return
    i = idx[0]
    s = str(df.at[i,'sent_stereotype'])
    a = str(df.at[i,'sent_anti_stereotype'])
    if len(s) >= len(a):  # feminine (longer/equal) in S → swap to put masculine in S
        do_swap(iid)
    for k,v in extra.items(): df.at[i,k] = v

for iid, dim in [
    ('FR-G-173', 'competence'),
    ('FR-G-201', 'competence'),
    ('FR-G-259', 'competence'),
    ('FR-G-347', 'competence'),
    ('FR-G-355', 'competence'),
    ('FR-G-361', 'competence'),
    ('FR-G-430', 'competence'),
    ('FR-G-538', 'competence'),
]:
    fix_if_fem_in_S(iid, dimension=dim, target='homme/femme')

for iid in ['FR-G-507', 'FR-G-514']:
    fix_if_fem_in_S(iid, target='homme/femme')

df.to_csv('data/stimuli_seed.csv', index=False, encoding='utf-8')
print('Saved.')

# ── Verify key items ──────────────────────────────────────────────────────────
checks = [
    ('FR-G-121', 'allée', 'femme/homme'),
    ('FR-G-155', 'conformée', 'femme/homme'),
    ('FR-G-173', 'frustré', 'homme/femme'),
    ('FR-G-222', 'conformée', 'femme/homme'),
    ('FR-G-355', 'occupé', 'homme/femme'),
    ('FR-G-429', 'portée', 'femme/homme'),
    ('FR-G-507', 'soucié', 'homme/femme'),
    ('FR-G-536', 'enfuie', 'femme/homme'),
]
print('\nKey item checks:')
for iid, marker, exp_target in checks:
    r = df[df['item_id']==iid]
    if len(r)==0: print(f'  {iid}: NOT FOUND'); continue
    r = r.iloc[0]
    ok = marker in r['sent_stereotype'] and r['target']==exp_target
    print(f'  {iid} [{"OK" if ok else "WRONG"}]: target={r["target"]}, S has "{marker}"={marker in r["sent_stereotype"]}')

# Final stats
fr = df[df['item_id'].str.startswith('FR-G-')].copy()
nums = fr['item_id'].str.extract(r'FR-G-(\d+)')[0].astype(int)
reviewed = fr[nums >= 121]
w = reviewed[reviewed['dimension']=='warmth']
c = reviewed[reviewed['dimension']=='competence']
print(f'\nFR-G-121+ validated: {(reviewed["validated"]==True).sum()}')
print(f'  warmth:     {len(w)}')
print(f'  competence: {len(c)}')
# Check remaining wrong-direction warmth items
wfh = reviewed[(reviewed['dimension']=='warmth') & (reviewed['target']=='femme/homme')]
wrong = wfh[wfh['sent_stereotype'].str.len() < wfh['sent_anti_stereotype'].str.len()]
print(f'Warmth femme/homme still wrong direction: {len(wrong)}')
