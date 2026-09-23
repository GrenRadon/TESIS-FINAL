"""Paso 5-6: metricas OOF (primarias) y test reservado (secundarias) + reporte comparativo.

Metrica primaria : prediccion pooled out-of-fold sobre el conjunto de desarrollo completo.
Metrica secundaria: test reservado de 20 casos, prediccion por ensamble de los 5 folds.
"""
import csv, json, math
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

OUT = Path('resultados_v2')
UMBRAL = 0.5
Z = 1.959963984540054  # z para IC 95%

# Resultado viejo contaminado (Tabla 7 de la tesis / reporte_final.json)
VIEJO = dict(auc=0.9426, sensibilidad=0.8515, especificidad=0.9333,
             tp=86, tn=28, fp=2, fn=15)


def wilson(exitos, n):
    """IC 95% de Wilson para una proporcion."""
    if n == 0:
        return (float('nan'), float('nan'), float('nan'))
    p = exitos / n
    den = 1 + Z ** 2 / n
    centro = (p + Z ** 2 / (2 * n)) / den
    margen = Z * math.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / den
    return (p, max(0.0, centro - margen), min(1.0, centro + margen))


def auc_ic_bootstrap(y, prob, grupos, n_boot=2000, seed=42):
    """IC 95% del AUC por bootstrap agrupado por caso (remuestrea casos, no imagenes)."""
    rng = np.random.default_rng(seed)
    por_grupo = defaultdict(list)
    for i, g in enumerate(grupos):
        por_grupo[g].append(i)
    claves = list(por_grupo)
    vals = []
    for _ in range(n_boot):
        sel = rng.choice(len(claves), size=len(claves), replace=True)
        idx = [i for s in sel for i in por_grupo[claves[s]]]
        yy, pp = y[idx], prob[idx]
        if len(set(yy.tolist())) < 2:
            continue
        vals.append(roc_auc_score(yy, pp))
    if not vals:
        return (float('nan'), float('nan'))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def metricas(y, prob, grupos=None, con_ic_auc=True):
    y = np.asarray(y); prob = np.asarray(prob)
    pred = (prob >= UMBRAL).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens, sens_lo, sens_hi = wilson(tp, tp + fn)
    espec, espec_lo, espec_hi = wilson(tn, tn + fp)
    auc = roc_auc_score(y, prob) if len(set(y.tolist())) > 1 else float('nan')
    d = dict(
        n=int(len(y)), n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()),
        auc=round(float(auc), 4),
        f1=round(float(f1_score(y, pred, zero_division=0)), 4),
        accuracy=round(float((tp + tn) / len(y)), 4),
        sensibilidad=round(sens, 4), sens_ic95=[round(sens_lo, 4), round(sens_hi, 4)],
        especificidad=round(espec, 4), espec_ic95=[round(espec_lo, 4), round(espec_hi, 4)],
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
    )
    if con_ic_auc and grupos is not None and len(set(y.tolist())) > 1:
        lo, hi = auc_ic_bootstrap(y, prob, grupos)
        d['auc_ic95'] = [round(lo, 4), round(hi, 4)]
    return d


def por_caso(rows, prob_col):
    """Agrega a nivel de caso: probabilidad = media de las imagenes del caso."""
    agg = defaultdict(lambda: dict(probs=[], label=None, clase=None))
    for r in rows:
        a = agg[r['clave_agrupacion']]
        a['probs'].append(float(r[prob_col]))
        a['label'] = int(r['label']); a['clase'] = r['clase']
    claves = sorted(agg)
    y = np.array([agg[c]['label'] for c in claves])
    p = np.array([float(np.mean(agg[c]['probs'])) for c in claves])
    return y, p, claves


def fmt(m, ic=True):
    s = (f"AUC={m['auc']:.4f}"
         + (f" [{m['auc_ic95'][0]:.4f}-{m['auc_ic95'][1]:.4f}]" if ic and 'auc_ic95' in m else '')
         + f"  Sens={m['sensibilidad']:.4f}"
         + (f" [{m['sens_ic95'][0]:.3f}-{m['sens_ic95'][1]:.3f}]" if ic else '')
         + f"  Espec={m['especificidad']:.4f}"
         + (f" [{m['espec_ic95'][0]:.3f}-{m['espec_ic95'][1]:.3f}]" if ic else '')
         + f"  F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}")
    return s


def main():
    oof = list(csv.DictReader(open(OUT / 'predicciones_oof.csv', encoding='utf-8')))
    tst = list(csv.DictReader(open(OUT / 'predicciones_test.csv', encoding='utf-8')))

    y_o = np.array([int(r['label']) for r in oof])
    p_o = np.array([float(r['prob']) for r in oof])
    g_o = np.array([r['clave_agrupacion'] for r in oof])
    m_oof_img = metricas(y_o, p_o, g_o)
    yc, pc, cc = por_caso(oof, 'prob')
    m_oof_caso = metricas(yc, pc, np.array(cc))

    y_t = np.array([int(r['label']) for r in tst])
    p_t = np.array([float(r['prob_ensamble']) for r in tst])
    g_t = np.array([r['clave_agrupacion'] for r in tst])
    m_tst_img = metricas(y_t, p_t, g_t)
    ytc, ptc, ctc = por_caso(tst, 'prob_ensamble')
    m_tst_caso = metricas(ytc, ptc, np.array(ctc))

    print('=' * 78)
    print('  PRIMARIA — OOF pooled sobre desarrollo (81 casos)')
    print('=' * 78)
    print(f'  por imagen ({m_oof_img["n"]} img): {fmt(m_oof_img)}')
    print(f'     CM: VP={m_oof_img["tp"]} VN={m_oof_img["tn"]} FP={m_oof_img["fp"]} FN={m_oof_img["fn"]}')
    print(f'  por caso   ({m_oof_caso["n"]} casos): {fmt(m_oof_caso)}')
    print(f'     CM: VP={m_oof_caso["tp"]} VN={m_oof_caso["tn"]} FP={m_oof_caso["fp"]} FN={m_oof_caso["fn"]}')
    print()
    print('=' * 78)
    print('  SECUNDARIA — test reservado, ensamble de 5 folds (20 casos)')
    print('=' * 78)
    print(f'  por imagen ({m_tst_img["n"]} img): {fmt(m_tst_img)}')
    print(f'     CM: VP={m_tst_img["tp"]} VN={m_tst_img["tn"]} FP={m_tst_img["fp"]} FN={m_tst_img["fn"]}')
    print(f'  por caso   ({m_tst_caso["n"]} casos): {fmt(m_tst_caso)}')
    print(f'     CM: VP={m_tst_caso["tp"]} VN={m_tst_caso["tn"]} FP={m_tst_caso["fp"]} FN={m_tst_caso["fn"]}')

    folds = json.load(open(OUT / 'folds_v2.json'))
    res = dict(viejo_contaminado=VIEJO, umbral=UMBRAL,
               primaria_oof=dict(por_imagen=m_oof_img, por_caso=m_oof_caso),
               secundaria_test=dict(por_imagen=m_tst_img, por_caso=m_tst_caso),
               folds=folds['folds'], cfg=folds['cfg'])
    json.dump(res, open(OUT / 'resultados_v2.json', 'w'), indent=2, ensure_ascii=False)
    print(f'\nJSON: {OUT / "resultados_v2.json"}')
    return res


if __name__ == '__main__':
    main()
