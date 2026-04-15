import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
df = pd.read_csv('data/stimuli_seed.csv', encoding='utf-8')

def upd(iid, **kw):
    idx = df[df['item_id']==iid].index
    if len(idx)==0: print(f'WARN: {iid} not found'); return
    i = idx[0]
    for k,v in kw.items(): df.loc[i,k] = v

def swap(iid):
    idx = df[df['item_id']==iid].index
    if len(idx)==0: print(f'WARN: {iid} not found'); return
    i = idx[0]
    s = str(df.loc[i,'sent_stereotype'])
    a = str(df.loc[i,'sent_anti_stereotype'])
    df.loc[i,'sent_stereotype'] = a
    df.loc[i,'sent_anti_stereotype'] = s

# ===========================================================================
# NEWLY EXCLUDED (items not already marked exclude in CSV)
# ===========================================================================
for iid in [
    # 121-150
    'FR-G-129','FR-G-130','FR-G-140','FR-G-141','FR-G-144',
    # 151-180
    'FR-G-152','FR-G-160','FR-G-166','FR-G-168',
    # 181-210
    'FR-G-185','FR-G-195',
    # 211-240
    'FR-G-218','FR-G-225','FR-G-242','FR-G-248',
    # 241-270
    'FR-G-252','FR-G-256','FR-G-264','FR-G-269',
    # 271-300
    'FR-G-272','FR-G-274','FR-G-278','FR-G-286','FR-G-288',
    'FR-G-291','FR-G-294','FR-G-297','FR-G-299','FR-G-300',
    # 301-330
    'FR-G-301','FR-G-302','FR-G-303','FR-G-304','FR-G-306',
    'FR-G-308','FR-G-309','FR-G-310','FR-G-314','FR-G-321',
    # 331-360
    'FR-G-331','FR-G-334','FR-G-335','FR-G-337','FR-G-357',
    # 361-390
    'FR-G-382',
    # 391-420
    'FR-G-398','FR-G-413','FR-G-416',
    # 421-450
    'FR-G-434','FR-G-447',
    # 451-480
    'FR-G-451','FR-G-471','FR-G-472','FR-G-474','FR-G-475','FR-G-476',
    # 481-510
    'FR-G-483','FR-G-486','FR-G-488','FR-G-492','FR-G-494',
    'FR-G-499','FR-G-500','FR-G-502','FR-G-503','FR-G-508',
    # 511-540
    'FR-G-511','FR-G-515','FR-G-519','FR-G-530','FR-G-531',
    'FR-G-533','FR-G-540',
    # 541-555
    'FR-G-541','FR-G-545','FR-G-554','FR-G-555',
]:
    upd(iid, dimension='exclude')

# ===========================================================================
# VALIDATE ONLY (dim + target already correct in CSV)
# ===========================================================================
for iid in [
    'FR-G-122','FR-G-136','FR-G-138','FR-G-146','FR-G-150',
    'FR-G-197','FR-G-204','FR-G-210','FR-G-211','FR-G-215',
    'FR-G-229','FR-G-231','FR-G-233','FR-G-270','FR-G-311',
    'FR-G-322','FR-G-326','FR-G-350','FR-G-389','FR-G-399',
    'FR-G-410','FR-G-417','FR-G-421','FR-G-425','FR-G-429',
    'FR-G-432','FR-G-438','FR-G-455','FR-G-459','FR-G-470',
    'FR-G-489','FR-G-512','FR-G-520','FR-G-529','FR-G-534',
    'FR-G-535','FR-G-537','FR-G-543','FR-G-550',
]:
    upd(iid, validated=True)

# ===========================================================================
# FIX target → homme/femme + VALIDATE
# (masculine in sent_stereotype; competence or low-warmth masculine)
# ===========================================================================
for iid in [
    # 121-150
    'FR-G-123','FR-G-125','FR-G-126','FR-G-127','FR-G-131',
    'FR-G-133','FR-G-135','FR-G-137','FR-G-147','FR-G-148','FR-G-149',
    # 151-180
    'FR-G-153','FR-G-154','FR-G-157','FR-G-158','FR-G-159',
    'FR-G-165','FR-G-167','FR-G-171','FR-G-172','FR-G-176','FR-G-180',
    # 181-210
    'FR-G-181','FR-G-198','FR-G-203','FR-G-206',
    # 211-240
    'FR-G-217','FR-G-220','FR-G-221','FR-G-224','FR-G-226',
    'FR-G-230','FR-G-232','FR-G-235','FR-G-236','FR-G-237','FR-G-238',
    # 241-270
    'FR-G-244','FR-G-254','FR-G-257','FR-G-258','FR-G-265',
    # 271-300
    'FR-G-273','FR-G-275','FR-G-276','FR-G-279','FR-G-280',
    'FR-G-281','FR-G-282','FR-G-284','FR-G-285','FR-G-295',
    # 301-330
    'FR-G-305','FR-G-307','FR-G-317','FR-G-319','FR-G-325','FR-G-330',
    # 331-360
    'FR-G-332','FR-G-336','FR-G-339','FR-G-340','FR-G-341',
    'FR-G-342','FR-G-348','FR-G-353','FR-G-356','FR-G-360',
    # 361-390
    'FR-G-366','FR-G-370','FR-G-371','FR-G-381','FR-G-384','FR-G-390',
    # 391-420
    'FR-G-391','FR-G-392','FR-G-394','FR-G-400','FR-G-401',
    'FR-G-404','FR-G-405','FR-G-406','FR-G-407','FR-G-412','FR-G-414','FR-G-420',
    # 421-450
    'FR-G-427','FR-G-435','FR-G-437','FR-G-443',
    # 451-480
    'FR-G-458','FR-G-467','FR-G-469','FR-G-478','FR-G-479','FR-G-480',
    # 481-510
    'FR-G-482','FR-G-485','FR-G-490','FR-G-493','FR-G-495',
    'FR-G-496','FR-G-497','FR-G-498','FR-G-505','FR-G-509','FR-G-510',
    # 511-540
    'FR-G-513','FR-G-517','FR-G-525','FR-G-532','FR-G-539',
    # 541-555
    'FR-G-547','FR-G-549',
]:
    upd(iid, target='homme/femme', validated=True)

# ===========================================================================
# SWAP + VALIDATE — target stays femme/homme (feminine in S after swap)
# Warmth items: masculine currently in S → swap so feminine in S
# ===========================================================================
for iid in [
    # 121-150 warmth
    'FR-G-121','FR-G-128','FR-G-134','FR-G-139','FR-G-142','FR-G-143','FR-G-145',
    # 151-180 warmth
    'FR-G-151','FR-G-155','FR-G-156','FR-G-177',
    # 181-210 warmth
    'FR-G-183','FR-G-187','FR-G-193','FR-G-207','FR-G-208','FR-G-209',
    # 211-240 warmth
    'FR-G-223','FR-G-227','FR-G-228',
    # 241-270 warmth
    'FR-G-241','FR-G-249','FR-G-255','FR-G-268',
    # 271-300 warmth
    'FR-G-283','FR-G-287','FR-G-289','FR-G-298',
    # 301-330 warmth
    'FR-G-313','FR-G-318','FR-G-323','FR-G-324',
    # 331-360 warmth
    'FR-G-343','FR-G-344','FR-G-351','FR-G-354',
    # 361-390 warmth
    'FR-G-362','FR-G-379','FR-G-383',
    # 391-420 warmth
    'FR-G-396','FR-G-397','FR-G-419',
    # 421-450 warmth
    'FR-G-433',
    # 451-480 warmth
    'FR-G-473','FR-G-477',
    # 481-510 warmth
    'FR-G-487','FR-G-501','FR-G-504',
    # 511-540 warmth
    'FR-G-521','FR-G-522','FR-G-524','FR-G-526',
    # 541-555 warmth
    'FR-G-542','FR-G-544','FR-G-548','FR-G-553',
]:
    swap(iid)
    upd(iid, validated=True)

# Physical weakness / competence — feminine in S after swap, target stays femme/homme
for iid in [
    'FR-G-169',  # inscrit → inscrite; didn't volunteer, needed muscle
    'FR-G-277',  # fort → forte; not strong enough for endurance
    'FR-G-293',  # porté → portée; didn't volunteer, needed muscle
    'FR-G-346',  # fort → forte; insisted on strength but proved wrong
    'FR-G-363',  # fort → forte; not strong enough
    'FR-G-393',  # fort → forte; wasn't strong enough
    'FR-G-491',  # fatigué → fatiguée; tired quickly from walk
    'FR-G-552',  # bon → bonne; never good at sports
]:
    swap(iid)
    upd(iid, validated=True)

# ===========================================================================
# COMPLEX: SWAP + dim change (target stays femme/homme = feminine after swap)
# ===========================================================================
for iid, dim in [
    ('FR-G-161', 'warmth'),   # reposé→reposée; sacrificed rest to clean = domestic warmth
    ('FR-G-222', 'warmth'),   # conformé→conformée; conforming to please = people-pleasing warmth
    ('FR-G-243', 'warmth'),   # blessé→blessée; hurt feelings sensitivity = warmth
    ('FR-G-290', 'warmth'),   # ignoré→ignorée; emotional neediness = warmth
    ('FR-G-358', 'warmth'),   # salarié→salariée; nursing home volunteer = caregiving warmth
    ('FR-G-484', 'warmth'),   # arrivé→arrivée; fleeing insects = fear/emotional = warmth
    ('FR-G-506', 'warmth'),   # perçu→perçue; concern for others seen as weak = warmth
    ('FR-G-536', 'warmth'),   # enfui→enfuie; panic and flee = fearful = warmth
]:
    swap(iid)
    upd(iid, dimension=dim, validated=True)

# Competence, physical weakness — swap makes feminine in S, dim change needed
for iid in [
    'FR-G-234',  # fatigué→fatiguée; tired after short run
    'FR-G-239',  # vaincu→vaincue; weak and defeated
    'FR-G-245',  # fatigué→fatiguée; tired but must prove endurance
    'FR-G-403',  # fait→faite; not built for hard physical work
]:
    swap(iid)
    upd(iid, dimension='competence', validated=True)

# ===========================================================================
# COMPLEX: SWAP + dim change + target → homme/femme
# (masculine ends up in sent_stereotype after swap)
# ===========================================================================
for iid, dim in [
    ('FR-G-173', 'competence'),  # frustrée→frustré; leadership ambition = masculine
    ('FR-G-201', 'competence'),  # censée→censé; no stage fright = masculine confidence
    ('FR-G-259', 'competence'),  # déterminée→déterminé; protector role = masculine
    ('FR-G-347', 'competence'),  # vivante→vivant; thrill-seeking/danger = masculine
    ('FR-G-355', 'competence'),  # occupée→occupé; financial management = masculine
    ('FR-G-361', 'competence'),  # stressée→stressé; breadwinner stress = masculine
    ('FR-G-430', 'competence'),  # forte→fort; feeling physically stronger = masculine
    ('FR-G-538', 'competence'),  # douée→doué; problem-solving = masculine competence
]:
    swap(iid)
    upd(iid, dimension=dim, target='homme/femme', validated=True)

# SWAP + target → homme/femme (dim stays warmth; low-warmth masculine after swap)
for iid in [
    'FR-G-507',  # souciée→soucié; doesn't care about food, only alcohol = low warmth
    'FR-G-514',  # impliquée→impliqué; constantly in mischief = low warmth masculine
]:
    swap(iid)
    upd(iid, target='homme/femme', validated=True)

# ===========================================================================
# NO SWAP — dim change and/or target change only
# ===========================================================================

# Masculine already in S; change dim + target as needed
upd('FR-G-162', dimension='competence', target='homme/femme', validated=True)  # absolute certainty
upd('FR-G-219', dimension='competence', validated=True)                          # physically weakest in gym (femme/homme ✓)
upd('FR-G-240', dimension='competence', target='homme/femme', validated=True)   # brave/reckless in woods
upd('FR-G-271', dimension='warmth',     target='homme/femme', validated=True)   # aggressive = low warmth masculine
upd('FR-G-292', dimension='competence', target='homme/femme', validated=True)   # lifelong preparation = masculine ambition
upd('FR-G-296', dimension='competence', target='homme/femme', validated=True)   # walked in storm despite warning
upd('FR-G-312', dimension='warmth',     target='homme/femme', validated=True)   # hides emotions despite being upset
upd('FR-G-481', dimension='competence', target='homme/femme', validated=True)   # focused on providing for daughter
upd('FR-G-516', dimension='competence', target='homme/femme', validated=True)   # wanted to stay objective
upd('FR-G-518', dimension='competence', target='homme/femme', validated=True)   # determined to protect loved ones
upd('FR-G-523', dimension='competence', target='homme/femme', validated=True)   # argued with police about ticket

# Target fix only (dim already correct)
upd('FR-G-527', target='homme/femme', validated=True)   # stoic/never worried = low-warmth masculine

# dim + target fix (currently needs_review)
upd('FR-G-528', dimension='warmth', target='homme/femme', validated=True)  # never learned to cook = domestic avoidance

# ===========================================================================
# SAVE + VERIFY
# ===========================================================================
df.to_csv('data/stimuli_seed.csv', index=False, encoding='utf-8')
print('Saved.')

fr = df[df['item_id'].str.startswith('FR-G-')].copy()
nums = fr['item_id'].str.extract(r'FR-G-(\d+)')[0].astype(int)
reviewed = fr[nums >= 121]
print(f'\nFR-G total rows:     {len(fr)}')
print(f'FR-G-121+  rows:     {len(reviewed)}')
print(f'  validated=True:    {(reviewed["validated"]==True).sum()}')
print(f'  dim=exclude:       {(reviewed["dimension"]=="exclude").sum()}')
print(f'  dim=needs_review:  {(reviewed["dimension"]=="needs_review").sum()}')
print(f'  dim=warmth:        {(reviewed["dimension"]=="warmth").sum()}')
print(f'  dim=competence:    {(reviewed["dimension"]=="competence").sum()}')
print(f'\nAll FR-G validated:  {(fr["validated"]==True).sum()}')
print(f'All FR-G excluded:   {(fr["dimension"]=="exclude").sum()}')
print(f'All FR-G needs_rev:  {(fr["dimension"]=="needs_review").sum()}')
