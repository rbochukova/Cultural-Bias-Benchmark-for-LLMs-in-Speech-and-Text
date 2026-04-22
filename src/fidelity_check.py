"""
Two-stage fidelity check for parallel FR-BG item pairs.

Stage 1: Classifies whether each sentence contains warmth or competence-relevant content, using expanded language-specific cue word lists. A pair passes if at least one cue is found in both the FR and BG versions.

Stage 2: Each sentence is translated back to English via GPT-4o-mini, then cosine similarity is computed between source and back-translated embeddings using OpenAI text-embedding-3-small. Items below threshold, default 0.70, are flagged.
"""

import argparse
import os
import pathlib
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT     = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "stimuli_seed.csv"
OUT_PATH = ROOT / "data" / "parallel_fidelity.csv"

WARMTH_FR = {
    'chaleureux','chaleureuse','gentil','gentille','aimable','sympa',
    'sympathique','affectueux','affectueuse','sensible','doux','douce',
    'tendre','bienveillant','bienveillante','attentionné','attentionnée',
    'généreux','généreuse','patient','patiente','passionné','passionnée',
    'émotif','émotive','émotionnel','émotionnelle','emotionnel','emotionnelle',
    'conformé','conformée','fier','fière','heureux','heureuse',
    'obligé','obligée','portée','porté','impliqué','impliquée',
    'reposé','reposée','blessé','blessée','arrivé','arrivée',
    'ignoré','ignorée','soucié','souciée','perçu','perçue',
    'bavard','bavarde','communicatif','communicative',
    'famille','familiaux','amis','ami','amie','relationnels','relationnel',
    'aider','aidé','aidée','aide','soutien','soutenu','soutenue',
    'solidaire','coopératif','coopérative','chaleur','empathie',
    'confiance','sincère','honnête','moral','générosité',
    'volunteering','volontaire','bénévole','sauver','sacrifice',
    'pleurer','pleurée','pleuré','larmes','romantique','romantiques',
}
COMPETENCE_FR = {
    'fort','forte','compétent','compétente','intelligent','intelligente',
    'capable','efficace','logique','rationnel','rationnelle','indépendant',
    'indépendante','ambitieux','ambitieuse','confiant','confiante',
    'courageux','courageuse','déterminé','déterminée','frustré','frustrée',
    'occupé','occupée','stressé','stressée','censé','censée',
    'fatigué','fatiguée','vaincu','vaincue','meilleur','meilleure',
    'strict','stricte','doué','douée','solide','sûr','sûre',
    'vivant','vivante','bon','bonne','professionnel','professionnelle',
    'assertif','assertive','décisif','décisive','analytique',
    'leadership','leader','diriger','dirigeant','gérer','gestion',
    'réussir','réussi','réussie','succès','performance','compétition',
    'carrière','travail','emploi','professionnel','technique','science',
    'mathématiques','problème','résoudre','réparer','construire',
    'conduire','sport','athlétique','gagner','gagné','championnat',
    'argent','finances','financier','budget','investissement',
    'protéger','protégé','protégée','défendre','défendu','sécurité',
}

WARMTH_BG = {
    'страстен','страстна','топъл','топла','мил','мила','добър','добра',
    'нежен','нежна','грижлив','грижлива','грижовен','грижовна',
    'емоционален','емоционална','чувствителен','чувствителна',
    'щедър','щедра','търпелив','търпелива','съпричастен','съпричастна',
    'внимателен','внимателна','любящ','любяща','добросърдечен','добросърдечна',
    'сърдечен','сърдечна','комуникативен','комуникативна',
    'искрен','искрена','честен','честна','морален','морална',
    'готов','готова', 
    'семейство','семейства','приятели','приятел','приятелки','приятелка',
    'помогна','помага','помощ','подкрепя','подкрепен','подкрепена',
    'грижа','грижи','грижи се','плача','плакал','плакала','сълзи',
    'романтично','романтична','любов','обичам','близки','роднини',
    'доброволец','жертва','спасявам','спасен','спасена',
}

COMPETENCE_BG = {
    'строг','строга','силен','силна','умен','умна','компетентен','компетентна',
    'способен','способна','ефективен','ефективна','логичен','логична',
    'независим','независима','амбициозен','амбициозна','уверен','уверена',
    'смел','смела','решителен','решителна','настоятелен','настоятелна',
    'по-добър','по-добра','успешен','успешна','надарен','надарена',
    'уморен','уморена','победен','победена','физически','физическа',
    'добър','добра',
    'готов','готова', 
    'ръководен','ръководна','ръководител','лидер','управлявам','управление',
    'успех','постижение','работа','кариера','професионален','професионална',
    'физически','спорт','атлетичен','атлетична','победа','шампион',
    'пари','финанси','финансов','финансова','бюджет','инвестиция',
    'защитавам','защитен','защитена','протектор','сигурност',
    'математика','наука','техника','проблем','решавам',
}

CUES = {
    'fr': {'warmth': WARMTH_FR, 'competence': COMPETENCE_FR},
    'bg': {'warmth': WARMTH_BG, 'competence': COMPETENCE_BG},
}


def _cue_present(text: str, lang: str, dim: str) -> bool:
    tokens = set(text.lower().split())

    text_lower = text.lower()
    cues = CUES.get(lang, {}).get(dim, set())
    return bool(tokens & cues) or any(c in text_lower for c in cues if len(c) > 5)


def _backtranslate_batch(client: OpenAI, texts: list[str], src_lang: str) -> list[str]:
    """Translate a list of texts from src_lang to English using GPT-4o-mini."""
    lang_name = {'fr': 'French', 'bg': 'Bulgarian'}[src_lang]
    results = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        numbered = '\n'.join(f'{j+1}. {t}' for j, t in enumerate(batch))
        prompt = (
            f"Translate each numbered {lang_name} sentence to English. "
            f"Return only the translations, one per line, numbered the same way. "
            f"Keep the meaning as close as possible.\n\n{numbered}"
        )
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model='gpt-4o-mini',
                    temperature=0,
                    messages=[{'role': 'user', 'content': prompt}],
                )
                lines = resp.choices[0].message.content.strip().split('\n')
              
                translations = []
                for line in lines:
                    line = line.strip()
                    if line and line[0].isdigit():
                        # Remove "1. " prefix
                        parts = line.split('.', 1)
                        translations.append(parts[1].strip() if len(parts) > 1 else line)
                    elif line:
                        translations.append(line)
                
                while len(translations) < len(batch):
                    translations.append('')
                results.extend(translations[:len(batch)])
                break
            except Exception as exc:
                if attempt == 2:
                    results.extend([''] * len(batch))
                time.sleep(2)

    print()
    return results


def _get_embeddings(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Get text-embedding-3-small embeddings for a list of texts"""
    vectors = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = [t if t.strip() else ' ' for t in texts[i:i + batch_size]]
        resp = client.embeddings.create(
            model='text-embedding-3-small',
            input=batch,
        )
        vectors.extend([e.embedding for e in resp.data])
        print(f'  Embedded {min(i+batch_size, len(texts))}/{len(texts)}',
              end='\r', flush=True)
    print()
    return np.array(vectors)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.70)
    parser.add_argument('--cue-only', action='store_true',
                        help='Skip back-translation, use cue check only')
    args = parser.parse_args()

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key and not args.cue_only:
        sys.exit('ERROR: OPENAI_API_KEY not set.')
    client = OpenAI(api_key=api_key) if api_key else None

    df = pd.read_csv(CSV_PATH, encoding='utf-8')
    par = df[df['parallel_group_id'].str.startswith('PG-', na=False)].copy()
    print(f"Parallel items: {len(par)}  ({par['parallel_group_id'].nunique()} groups)")

    # Build one row per parallel group
    rows = []
    for pgid, grp in par.groupby('parallel_group_id'):
        fr_row = grp[grp['language'] == 'fr']
        bg_row = grp[grp['language'] == 'bg']
        if fr_row.empty or bg_row.empty:
            continue
        fr_row = fr_row.iloc[0]
        bg_row = bg_row.iloc[0]
        dim = str(fr_row['dimension'])
        rows.append({
            'parallel_group_id': pgid,
            'fr_item_id':        fr_row['item_id'],
            'bg_item_id':        bg_row['item_id'],
            'dimension':         dim,
            'direction_agree':   fr_row['target'] in ('femme/homme','homme/femme') and
                                 _direction(fr_row['target']) == _direction(bg_row['target']),
            'dim_agree':         fr_row['dimension'] == bg_row['dimension'],
            'fr_stereo':         str(fr_row['sent_stereotype']),
            'bg_stereo':         str(bg_row['sent_stereotype']),
            'cue_preserved_fr':  _cue_present(
                str(fr_row['sent_stereotype']) + ' ' + str(fr_row['sent_anti_stereotype']),
                'fr', dim),
            'cue_preserved_bg':  _cue_present(
                str(bg_row['sent_stereotype']) + ' ' + str(bg_row['sent_anti_stereotype']),
                'bg', str(bg_row['dimension'])),
        })

    fid_df = pd.DataFrame(rows)

    print(f"\nStage 1 - Cue-preservation check")
    print(f"  FR cue present : {fid_df['cue_preserved_fr'].sum()} / {len(fid_df)}")
    print(f"  BG cue present : {fid_df['cue_preserved_bg'].sum()} / {len(fid_df)}")
    print(f"  Both present   : {(fid_df['cue_preserved_fr'] & fid_df['cue_preserved_bg']).sum()} / {len(fid_df)}")
    print(f"  Dimension agree: {fid_df['dim_agree'].sum()} / {len(fid_df)}")
    print(f"  Direction agree: {fid_df['direction_agree'].sum()} / {len(fid_df)}")

    if args.cue_only:
        fid_df['sim_fr']       = float('nan')
        fid_df['sim_bg']       = float('nan')
        fid_df['bt_fr']        = ''
        fid_df['bt_bg']        = ''
        fid_df['high_fidelity'] = fid_df['dim_agree'] & fid_df['direction_agree']
        _save(fid_df, args)
        return

    print(f"\nStage 2 - Back-translation + embedding similarity")

    print("  Back-translating FR sentences")
    bt_fr = _backtranslate_batch(client, fid_df['fr_stereo'].tolist(), 'fr')
    print("  Back-translating BG sentences")
    bt_bg = _backtranslate_batch(client, fid_df['bg_stereo'].tolist(), 'bg')

    fid_df['bt_fr'] = bt_fr
    fid_df['bt_bg'] = bt_bg

    print("  Embedding original + back-translated sentences")
    all_texts = (
        fid_df['fr_stereo'].tolist() +
        fid_df['bg_stereo'].tolist() +
        bt_fr + bt_bg
    )
    embs = _get_embeddings(client, all_texts)
    n = len(fid_df)
    emb_fr_orig = embs[:n]
    emb_bg_orig = embs[n:2*n]
    emb_fr_bt   = embs[2*n:3*n]
    emb_bg_bt   = embs[3*n:]

    fid_df['sim_fr'] = [round(_cosine_sim(emb_fr_orig[i], emb_fr_bt[i]), 4) for i in range(n)]
    fid_df['sim_bg'] = [round(_cosine_sim(emb_bg_orig[i], emb_bg_bt[i]), 4) for i in range(n)]

    print(f"\n  Similarity stats (FR back-translation):")
    print(f"    mean={fid_df['sim_fr'].mean():.3f}  "
          f"min={fid_df['sim_fr'].min():.3f}  "
          f"max={fid_df['sim_fr'].max():.3f}")
    print(f"  Similarity stats (BG back-translation):")
    print(f"    mean={fid_df['sim_bg'].mean():.3f}  "
          f"min={fid_df['sim_bg'].min():.3f}  "
          f"max={fid_df['sim_bg'].max():.3f}")
    print(f"  FR above {args.threshold}: {(fid_df['sim_fr'] >= args.threshold).sum()}")
    print(f"  BG above {args.threshold}: {(fid_df['sim_bg'] >= args.threshold).sum()}")

    fid_df['high_fidelity'] = fid_df['dim_agree'] & fid_df['direction_agree']

    _save(fid_df, args)


def _direction(target: str) -> str:
    if target in ('femme/homme', 'жена/мъж'):
        return 'fem_in_S'
    if target in ('homme/femme', 'мъж/жена'):
        return 'masc_in_S'
    return 'unknown'


def _save(fid_df: pd.DataFrame, args) -> None:
    out = fid_df.drop(columns=['fr_stereo', 'bg_stereo'], errors='ignore')
    out.to_csv(OUT_PATH, index=False, encoding='utf-8')

    hf = fid_df['high_fidelity'].sum()
    n  = len(fid_df)
    print(f"\n{'='*55}")
    print(f"High-fidelity subset : {hf} / {n} pairs  ({100*hf/n:.0f}%)")
    w  = fid_df[fid_df['dimension'] == 'warmth']['high_fidelity'].sum()
    c  = fid_df[fid_df['dimension'] == 'competence']['high_fidelity'].sum()
    print(f"  warmth     : {w}")
    print(f"  competence : {c}")
    print(f"Saved: {OUT_PATH.name}")

    print(f"\n{'='*55}")
    print("PARALLEL FIDELITY METHODS TABLE")
    print(f"{'='*55}")
    print(f"{'Criterion':<45} {'N':>5} {'%':>6}")
    print("-" * 57)
    print(f"{'Total parallel pairs':<45} {n:>5} {100:.0f}%")
    dim_ok  = fid_df['dim_agree'].sum()
    dir_ok  = fid_df['direction_agree'].sum()
    both_ok = (fid_df['dim_agree'] & fid_df['direction_agree']).sum()
    print(f"{'  Dimension agreement (FR = BG)':<45} {dim_ok:>5} {100*dim_ok/n:>5.0f}%")
    print(f"{'  Stereotype direction agreement (FR = BG)':<45} {dir_ok:>5} {100*dir_ok/n:>5.0f}%")
    print(f"{'  Both agree -> high-fidelity subset':<45} {both_ok:>5} {100*both_ok/n:>5.0f}%")
    if 'cue_preserved_fr' in fid_df.columns:
        cue_fr = fid_df['cue_preserved_fr'].sum()
        cue_bg = fid_df['cue_preserved_bg'].sum()
        cue_both = (fid_df['cue_preserved_fr'] & fid_df['cue_preserved_bg']).sum()
        print(f"{'  SCM cue present - FR':<45} {cue_fr:>5} {100*cue_fr/n:>5.0f}%  (diagnostic)")
        print(f"{'  SCM cue present - BG':<45} {cue_bg:>5} {100*cue_bg/n:>5.0f}%  (diagnostic)")
        print(f"{'  Cue present in both':<45} {cue_both:>5} {100*cue_both/n:>5.0f}%  (diagnostic)")
    if 'sim_cross' in fid_df.columns and not fid_df['sim_cross'].isna().all():
        sim_mean = fid_df['sim_cross'].mean()
        sim_min  = fid_df['sim_cross'].min()
        above70  = (fid_df['sim_cross'] >= 0.70).sum()
        print(f"{'  Cross-lang semantic sim ≥ 0.70':<45} {above70:>5} {100*above70/n:>5.0f}%  (diagnostic)")
        print(f"  Cross-lang sim: mean={sim_mean:.3f}  min={sim_min:.3f}")
    print("-" * 57)
    print(f"  High-fidelity criterion: dim_agree AND direction_agree")
    print()
    print("Excluded pairs (dim or direction mismatch):")
    low = fid_df[~fid_df['high_fidelity']]
    for _, r in low.iterrows():
        reasons = []
        if not r['dim_agree']:       reasons.append('dim_mismatch')
        if not r['direction_agree']: reasons.append('dir_mismatch')
        print(f"  {r['parallel_group_id']} ({r['dimension']}): {', '.join(reasons)}")


if __name__ == '__main__':
    main()
