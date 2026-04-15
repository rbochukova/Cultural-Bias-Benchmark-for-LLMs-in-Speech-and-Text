"""
link_parallel_items.py
~~~~~~~~~~~~~~~~~~~~~~
Links FR-G and BG-G items that share the same EuroGEST Source sentence.

For each matched pair:
  - Sets origin = 'parallel' on both items
  - Sets parallel_group_id = 'PG-NNN' (shared, sequential) on both items

Items with no parallel counterpart keep origin='native' and their existing
parallel_group_id (sequential number suffix of their item_id).

Also links EN-N and FR-N nationality items via shared nationality entity.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from datasets import load_dataset

df = pd.read_csv('data/stimuli_seed.csv', encoding='utf-8')

# ── 1. Load EuroGEST FR and BG ────────────────────────────────────────────────
print("Loading EuroGEST...")
fr_ds = load_dataset('utter-project/EuroGEST', split='French',
                     trust_remote_code=True).to_pandas()
bg_ds = load_dataset('utter-project/EuroGEST', split='Bulgarian',
                     trust_remote_code=True).to_pandas()

fr_usable = fr_ds.dropna(subset=['Masculine', 'Feminine']).copy()
bg_usable = bg_ds.dropna(subset=['Masculine', 'Feminine']).copy()
print(f"EuroGEST usable: FR={len(fr_usable)}, BG={len(bg_usable)}")

# ── 2. Build lookup tables ────────────────────────────────────────────────────

# EuroGEST BG: Source -> list of (Masculine, Feminine) rows
bg_by_source = {}
for _, r in bg_usable.iterrows():
    src = str(r['Source']).strip()
    bg_by_source.setdefault(src, []).append(
        (str(r['Masculine']).strip(), str(r['Feminine']).strip())
    )

# CSV BG-G items: frozenset({S, A}) -> CSV index
bg_items = df[df['item_id'].str.startswith('BG-G-')].copy()
bg_text_to_idx = {}
for idx, r in bg_items.iterrows():
    key = frozenset([str(r['sent_stereotype']).strip(),
                     str(r['sent_anti_stereotype']).strip()])
    bg_text_to_idx[key] = idx

# CSV FR-G items: frozenset({S, A}) -> CSV index
fr_items = df[df['item_id'].str.startswith('FR-G-')].copy()
fr_text_to_idx = {}
for idx, r in fr_items.iterrows():
    key = frozenset([str(r['sent_stereotype']).strip(),
                     str(r['sent_anti_stereotype']).strip()])
    fr_text_to_idx[key] = idx

# ── 3. Match FR ↔ BG via shared Source sentence ───────────────────────────────
pairs = []  # list of (fr_csv_idx, bg_csv_idx, source_en)

for _, fr_row in fr_usable.iterrows():
    src  = str(fr_row['Source']).strip()
    masc = str(fr_row['Masculine']).strip()
    fem  = str(fr_row['Feminine']).strip()

    # Is this FR item in our CSV?
    fr_key = frozenset([masc, fem])
    if fr_key not in fr_text_to_idx:
        continue
    fr_idx = fr_text_to_idx[fr_key]

    # Is there a BG counterpart in EuroGEST?
    if src not in bg_by_source:
        continue

    # Is that BG counterpart in our CSV?
    for bg_masc, bg_fem in bg_by_source[src]:
        bg_key = frozenset([bg_masc, bg_fem])
        if bg_key in bg_text_to_idx:
            bg_idx = bg_text_to_idx[bg_key]
            pairs.append((fr_idx, bg_idx, src))
            break  # one BG match per FR item is enough

print(f"\nMatched FR↔BG pairs: {len(pairs)}")

# ── 4. Apply parallel_group_id and origin to matched pairs ───────────────────
for i, (fr_idx, bg_idx, _) in enumerate(pairs, start=1):
    pgid = f"PG-{i:03d}"
    df.at[fr_idx, 'parallel_group_id'] = pgid
    df.at[fr_idx, 'origin']            = 'parallel'
    df.at[bg_idx, 'parallel_group_id'] = pgid
    df.at[bg_idx, 'origin']            = 'parallel'

# ── 5. Link EN-N and FR-N via shared nationality entity ──────────────────────
# Both came from SHADES; target field holds the nationality name
en_n = df[df['item_id'].str.startswith('EN-N-')].copy()
fr_n = df[df['item_id'].str.startswith('FR-N-')].copy()

en_by_target = {str(r['target']).strip().lower(): idx
                for idx, r in en_n.iterrows()}
fr_by_target = {str(r['target']).strip().lower(): idx
                for idx, r in fr_n.iterrows()}

shared_nations = set(en_by_target) & set(fr_by_target)
print(f"\nShared nationality targets EN∩FR: {len(shared_nations)}")

pg_n_counter = 1
for nation in sorted(shared_nations):
    pgid = f"PN-{pg_n_counter:03d}"
    df.at[en_by_target[nation], 'parallel_group_id'] = pgid
    df.at[en_by_target[nation], 'origin']            = 'parallel'
    df.at[fr_by_target[nation], 'parallel_group_id'] = pgid
    df.at[fr_by_target[nation], 'origin']            = 'parallel'
    pg_n_counter += 1

# ── 6. Save ───────────────────────────────────────────────────────────────────
df.to_csv('data/stimuli_seed.csv', index=False, encoding='utf-8')
print("\nSaved.")

# ── 7. Report ─────────────────────────────────────────────────────────────────
print("\n=== Parallel item summary ===")
par = df[df['origin'] == 'parallel']
nat = df[df['origin'] == 'native']
print(f"origin='parallel' : {len(par)} items  ({par['parallel_group_id'].nunique()} groups)")
print(f"origin='native'   : {len(nat)} items")
print()

pg_g = par[par['parallel_group_id'].str.startswith('PG-')]
pg_n = par[par['parallel_group_id'].str.startswith('PN-')]
print(f"Gender parallel groups  (PG-): {pg_g['parallel_group_id'].nunique()}")
print(f"  FR items : {len(pg_g[pg_g['item_id'].str.startswith('FR-G-')])}")
print(f"  BG items : {len(pg_g[pg_g['item_id'].str.startswith('BG-G-')])}")
print()
print(f"Nationality parallel groups (PN-): {pg_n['parallel_group_id'].nunique()}")
print(f"  EN items : {len(pg_n[pg_n['item_id'].str.startswith('EN-N-')])}")
print(f"  FR items : {len(pg_n[pg_n['item_id'].str.startswith('FR-N-')])}")
print()

# Show a few examples
print("=== Sample parallel gender pairs ===")
for pgid in list(pg_g['parallel_group_id'].unique())[:5]:
    grp = df[df['parallel_group_id'] == pgid]
    print(f"\n{pgid}:")
    for _, r in grp.iterrows():
        print(f"  [{r['item_id']}] dim={r['dimension']} target={r['target']}")
        print(f"    S: {r['sent_stereotype'][:80]}")
        print(f"    A: {r['sent_anti_stereotype'][:80]}")
