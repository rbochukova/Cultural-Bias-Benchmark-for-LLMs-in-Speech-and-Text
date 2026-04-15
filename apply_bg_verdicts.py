import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
df = pd.read_csv('data/stimuli_seed.csv', encoding='utf-8')

def swap_sent(item_id):
    idx = df[df['item_id']==item_id].index
    if len(idx)==0:
        print(f'WARN: {item_id} not found')
        return
    i = idx[0]
    df.loc[i,'sent_stereotype'], df.loc[i,'sent_anti_stereotype'] = \
        df.loc[i,'sent_anti_stereotype'], df.loc[i,'sent_stereotype']

def upd(item_id, **kw):
    idx = df[df['item_id']==item_id].index
    if len(idx)==0:
        print(f'WARN: {item_id} not found')
        return
    i = idx[0]
    for k,v in kw.items():
        df.loc[i,k] = v

# EXCLUDE
for item in [
    'BG-G-316','BG-G-319','BG-G-322','BG-G-323','BG-G-333',
    'BG-G-347','BG-G-355',
    'BG-G-378','BG-G-388','BG-G-395','BG-G-398',
    'BG-G-405','BG-G-408','BG-G-417','BG-G-419','BG-G-420','BG-G-421','BG-G-422',
    'BG-G-435','BG-G-437','BG-G-455','BG-G-456','BG-G-459',
    'BG-G-461','BG-G-463','BG-G-466','BG-G-467','BG-G-469','BG-G-472',
    'BG-G-477','BG-G-480','BG-G-481','BG-G-482','BG-G-483','BG-G-486',
    'BG-G-492','BG-G-494','BG-G-495',
    'BG-G-527','BG-G-550',
    'BG-G-552','BG-G-553','BG-G-557','BG-G-560','BG-G-562','BG-G-563',
    'BG-G-564','BG-G-569','BG-G-570','BG-G-571','BG-G-573','BG-G-577',
    'BG-G-578','BG-G-580',
    'BG-G-582','BG-G-585','BG-G-588','BG-G-590','BG-G-592','BG-G-594',
    'BG-G-597','BG-G-600','BG-G-601','BG-G-605','BG-G-607','BG-G-608',
    'BG-G-614','BG-G-620','BG-G-624','BG-G-634','BG-G-637','BG-G-638',
    'BG-G-645','BG-G-650','BG-G-656','BG-G-659',
    'BG-G-678','BG-G-681','BG-G-682','BG-G-686','BG-G-688','BG-G-690',
    'BG-G-692','BG-G-694','BG-G-695','BG-G-696',
    'BG-G-701','BG-G-705','BG-G-706','BG-G-707','BG-G-708','BG-G-709',
    'BG-G-711','BG-G-713','BG-G-715','BG-G-716','BG-G-717','BG-G-719',
    'BG-G-721','BG-G-727','BG-G-730',
    'BG-G-733','BG-G-734','BG-G-735','BG-G-736','BG-G-737','BG-G-738',
    'BG-G-739','BG-G-746',
    'BG-G-763','BG-G-766','BG-G-767','BG-G-768','BG-G-769',
    'BG-G-779','BG-G-780','BG-G-781',
]:
    upd(item, dimension='exclude')

# BG-G-311: warmth -> competence, feminine stays in S (physical weakness), target stays
upd('BG-G-311', dimension='competence', validated='True')

# VALIDATE ONLY (correct as-is, no structural changes)
for item in [
    'BG-G-313','BG-G-336','BG-G-340',
    'BG-G-350','BG-G-370',
    'BG-G-371','BG-G-377','BG-G-385',
    'BG-G-436','BG-G-444',
    'BG-G-499',
    'BG-G-522',
    'BG-G-561','BG-G-575',
    'BG-G-583','BG-G-587','BG-G-593','BG-G-598',
    'BG-G-629','BG-G-630','BG-G-631',
    'BG-G-664',
    'BG-G-673','BG-G-683','BG-G-691',
    'BG-G-725',
    'BG-G-731','BG-G-756',
    'BG-G-761','BG-G-762','BG-G-764','BG-G-773',
]:
    upd(item, validated='True')

# FIX TARGET -> myzh/zhena + VALIDATE (masculine in S, no dim or swap change)
for item in [
    'BG-G-312','BG-G-317','BG-G-320','BG-G-324','BG-G-326','BG-G-328',
    'BG-G-329','BG-G-330','BG-G-332','BG-G-337',
    'BG-G-343','BG-G-345','BG-G-348','BG-G-349','BG-G-354','BG-G-365','BG-G-369',
    'BG-G-372','BG-G-376','BG-G-383','BG-G-387','BG-G-389','BG-G-390',
    'BG-G-391','BG-G-394','BG-G-396',
    'BG-G-401','BG-G-402','BG-G-403','BG-G-404','BG-G-406','BG-G-407',
    'BG-G-409','BG-G-414','BG-G-416','BG-G-423','BG-G-425','BG-G-428',
    'BG-G-442','BG-G-443','BG-G-451','BG-G-452','BG-G-454','BG-G-458','BG-G-460',
    'BG-G-462','BG-G-464','BG-G-471','BG-G-474','BG-G-476','BG-G-479',
    'BG-G-485','BG-G-487','BG-G-489',
    'BG-G-491','BG-G-497','BG-G-502','BG-G-503','BG-G-505','BG-G-506',
    'BG-G-510','BG-G-511','BG-G-519','BG-G-520',
    'BG-G-530','BG-G-531','BG-G-537','BG-G-538','BG-G-540','BG-G-542',
    'BG-G-544','BG-G-545','BG-G-546','BG-G-547',
    'BG-G-551','BG-G-554','BG-G-555','BG-G-558','BG-G-559','BG-G-565',
    'BG-G-567','BG-G-568','BG-G-572','BG-G-574','BG-G-576',
    'BG-G-586','BG-G-595',
    'BG-G-612','BG-G-615','BG-G-616','BG-G-618','BG-G-636','BG-G-640',
    'BG-G-644','BG-G-649','BG-G-653','BG-G-654','BG-G-655','BG-G-657',
    'BG-G-665','BG-G-668',
    'BG-G-672','BG-G-674','BG-G-679','BG-G-680','BG-G-684','BG-G-687',
    'BG-G-689','BG-G-693','BG-G-697','BG-G-698','BG-G-699','BG-G-700',
    'BG-G-702','BG-G-703','BG-G-710','BG-G-712','BG-G-718','BG-G-720',
    'BG-G-722','BG-G-723','BG-G-726','BG-G-728',
    'BG-G-740','BG-G-743','BG-G-744','BG-G-748','BG-G-749','BG-G-752','BG-G-758',
    'BG-G-770',
]:
    upd(item, target='\u043c\u044a\u0436/\u0436\u0435\u043d\u0430', validated='True')

# FIX DIM -> competence + FIX TARGET -> myzh/zhena + VALIDATE
for item in [
    'BG-G-400','BG-G-411','BG-G-415','BG-G-426',
    'BG-G-446','BG-G-457',
    'BG-G-470','BG-G-473','BG-G-488',
    'BG-G-518',
    'BG-G-566','BG-G-589',
    'BG-G-617',
    'BG-G-704',
]:
    upd(item, dimension='competence', target='\u043c\u044a\u0436/\u0436\u0435\u043d\u0430', validated='True')

# Warmth items with masculine in S: fix target only (low-warmth male stereotype)
upd('BG-G-685', dimension='warmth', target='\u043c\u044a\u0436/\u0436\u0435\u043d\u0430', validated='True')
upd('BG-G-729', target='\u043c\u044a\u0436/\u0436\u0435\u043d\u0430', validated='True')

# SWAP + VALIDATE (masculine becomes A, feminine becomes S; target stays zhena/myzh)
for item in [
    'BG-G-315','BG-G-321','BG-G-327','BG-G-334','BG-G-339',
    'BG-G-344','BG-G-352','BG-G-353','BG-G-356','BG-G-358','BG-G-360',
    'BG-G-374','BG-G-381','BG-G-384','BG-G-386','BG-G-392','BG-G-393',
    'BG-G-397','BG-G-399',
    'BG-G-412','BG-G-418','BG-G-424',
    'BG-G-434','BG-G-439','BG-G-441','BG-G-447','BG-G-449','BG-G-450',
    'BG-G-465','BG-G-468','BG-G-478','BG-G-484',
    'BG-G-493','BG-G-496','BG-G-517','BG-G-548',
    'BG-G-528','BG-G-543','BG-G-549',
    'BG-G-556','BG-G-579',
    'BG-G-581','BG-G-584','BG-G-596',
    'BG-G-623',
    'BG-G-646','BG-G-658','BG-G-660','BG-G-662','BG-G-663','BG-G-669',
    'BG-G-671','BG-G-676','BG-G-677',
    'BG-G-724',
    'BG-G-732','BG-G-741','BG-G-742','BG-G-745','BG-G-747','BG-G-753',
    'BG-G-759','BG-G-760',
    'BG-G-771','BG-G-777','BG-G-778',
]:
    swap_sent(item)
    upd(item, validated='True')

# SWAP + DIM CHANGE (various)
swap_sent('BG-G-431'); upd('BG-G-431', dimension='warmth', validated='True')
swap_sent('BG-G-440'); upd('BG-G-440', dimension='competence', target='\u043c\u044a\u0436/\u0436\u0435\u043d\u0430', validated='True')
swap_sent('BG-G-448'); upd('BG-G-448', dimension='competence', target='\u043c\u044a\u0436/\u0436\u0435\u043d\u0430', validated='True')
swap_sent('BG-G-475'); upd('BG-G-475', dimension='competence', validated='True')
swap_sent('BG-G-490'); upd('BG-G-490', dimension='warmth', validated='True')
swap_sent('BG-G-410'); upd('BG-G-410', dimension='warmth', validated='True')
swap_sent('BG-G-413'); upd('BG-G-413', dimension='warmth', validated='True')
swap_sent('BG-G-427'); upd('BG-G-427', dimension='warmth', validated='True')
swap_sent('BG-G-500'); upd('BG-G-500', dimension='competence', validated='True')
swap_sent('BG-G-504'); upd('BG-G-504', dimension='competence', validated='True')
swap_sent('BG-G-591'); upd('BG-G-591', dimension='competence', validated='True')
swap_sent('BG-G-599'); upd('BG-G-599', dimension='competence', target='\u043c\u044a\u0436/\u0436\u0435\u043d\u0430', validated='True')
swap_sent('BG-G-714'); upd('BG-G-714', dimension='warmth', validated='True')

# SAVE
df.to_csv('data/stimuli_seed.csv', index=False, encoding='utf-8')
print('Saved.')

# VERIFY
df2 = pd.read_csv('data/stimuli_seed.csv', encoding='utf-8')
bg = df2[df2.item_id.str.startswith('BG-G-')]
print(f'BG-G total: {len(bg)}')
print(f'  excluded:        {len(bg[bg.dimension=="exclude"])}')
print(f'  validated=True:  {len(bg[bg.validated=="True"])}')
print(f'  validated=False: {len(bg[bg.validated=="False"])}')
print(f'  target=myzh/zhena: {len(bg[bg.target==chr(1084)+chr(1098)+chr(1078)+"/"+chr(1078)+chr(1077)+chr(1085)+chr(1072)])}')
print(f'  target=zhena/myzh: {len(bg[bg.target==chr(1078)+chr(1077)+chr(1085)+chr(1072)+"/"+chr(1084)+chr(1098)+chr(1078)])}')
nr = bg[bg.dimension=='needs_review']
print(f'  needs_review remaining: {len(nr)}')
if len(nr): print('   ', nr.item_id.tolist())
active = bg[bg.dimension!='exclude']
print(f'  active (non-excluded): {len(active)}')
wm = active[active.dimension=='warmth']
cm = active[active.dimension=='competence']
print(f'    warmth:     {len(wm)}')
print(f'    competence: {len(cm)}')
