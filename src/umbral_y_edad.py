"""M-08: seleccion de umbral sobre OOF (nunca sobre el test).
M-10/M-14: estratificacion de errores por grupo etario.

El umbral se elige SOLO con predicciones out-of-fold del set de desarrollo.
El test reservado se evalua una unica vez con el umbral ya fijado.
"""
import csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from figuras import guardar
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, confusion_matrix, precision_recall_curve
from scipy.stats import fisher_exact, mannwhitneyu

OUT = Path('resultados_v2')
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
CORTE_PEDIATRICO = 18.0
Z = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return (float('nan'),) * 3
    p = k / n
    den = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / den
    m = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / den
    return p, max(0.0, c - m), min(1.0, c + m)


def por_caso(rows, col):
    agg = defaultdict(lambda: {'p': [], 'y': None})
    for r in rows:
        a = agg[r['clave_agrupacion']]
        a['p'].append(float(r[col]))
        a['y'] = int(r['label'])
    claves = sorted(agg)
    return (np.array(claves),
            np.array([agg[c]['y'] for c in claves]),
            np.array([float(np.mean(agg[c]['p'])) for c in claves]))


def met(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens, s_lo, s_hi = wilson(tp, tp + fn)
    esp, e_lo, e_hi = wilson(tn, tn + fp)
    return dict(umbral=round(float(thr), 4), tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                sensibilidad=round(sens, 4), sens_ic95=[round(s_lo, 4), round(s_hi, 4)],
                especificidad=round(esp, 4), espec_ic95=[round(e_lo, 4), round(e_hi, 4)],
                youden=round(sens + esp - 1, 4),
                f1=round(float(f1_score(y, pred, zero_division=0)), 4),
                accuracy=round(float((tp + tn) / len(y)), 4))


def elegir_umbral(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    i = int(np.argmax(j))
    thr_j = float(thr[i])
    cands = np.unique(np.round(p, 4))
    f1s = [(f1_score(y, (p >= t).astype(int), zero_division=0), float(t)) for t in cands]
    thr_f1 = max(f1s)[1]
    return dict(youden=thr_j, f1=thr_f1, fpr=fpr.tolist(), tpr=tpr.tolist(),
                thr=[float(t) for t in thr], j_max=float(j[i]), idx=i)


def edad_num(s):
    s = (s or '').strip()
    if not s or s == '-':
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main():
    oof = list(csv.DictReader(open(OUT / 'predicciones_oof.csv', encoding='utf-8')))
    tst = list(csv.DictReader(open(OUT / 'predicciones_test.csv', encoding='utf-8')))
    man = {r['clave_agrupacion']: r for r in
           csv.DictReader(open(OUT / 'manifiesto_final_casos.csv', encoding='utf-8'))}

    res = {}

    # ============ 1. UMBRAL ============
    print('=' * 78)
    print('  M-08 — SELECCION DE UMBRAL (solo con OOF del desarrollo)')
    print('=' * 78)

    for nivel in ('caso', 'imagen'):
        if nivel == 'caso':
            claves, y, p = por_caso(oof, 'prob')
        else:
            y = np.array([int(r['label']) for r in oof])
            p = np.array([float(r['prob']) for r in oof])
        sel = elegir_umbral(y, p)
        m05 = met(y, p, 0.5)
        mj = met(y, p, sel['youden'])
        mf = met(y, p, sel['f1'])
        auc = roc_auc_score(y, p)
        print(f'\n--- OOF por {nivel} (n={len(y)}, AUC={auc:.4f}) ---')
        print(f'  {"criterio":<22}{"umbral":>8}{"Sens":>9}{"Espec":>9}{"Youden":>9}{"F1":>8}   VP/VN/FP/FN')
        for nom, m in (('0,5 (por defecto)', m05), ('Youden J max', mj), ('F1 max', mf)):
            print(f'  {nom:<22}{m["umbral"]:>8.3f}{m["sensibilidad"]:>9.4f}{m["especificidad"]:>9.4f}'
                  f'{m["youden"]:>9.4f}{m["f1"]:>8.4f}   {m["tp"]}/{m["tn"]}/{m["fp"]}/{m["fn"]}')
        res[f'oof_{nivel}'] = dict(auc=round(float(auc), 4), umbral_youden=round(sel['youden'], 4),
                                   umbral_f1=round(sel['f1'], 4), en_0_5=m05, en_youden=mj, en_f1=mf)
        res[f'roc_{nivel}'] = {k: sel[k] for k in ('fpr', 'tpr', 'thr')}

    # ---- umbral con restriccion de sensibilidad (criterio de cribado) ----
    # Youden pesa sensibilidad y especificidad por igual, lo que contradice la prioridad
    # declarada en la tesis de minimizar falsos negativos. Se preespecifica en su lugar:
    # maxima especificidad sujeta a sensibilidad >= SENS_MIN, elegido SOLO sobre OOF.
    SENS_MIN = 0.95
    claves_c, y_c, p_c = por_caso(oof, 'prob')
    fpr, tpr, thr = roc_curve(y_c, p_c)
    viables = [(t, s, f) for t, s, f in zip(thr, tpr, fpr) if s >= SENS_MIN]
    THR_SENS = float(min(viables, key=lambda x: x[2])[0])
    m_sens = met(y_c, p_c, THR_SENS)
    print(f'\n--- umbral con restriccion Sens >= {SENS_MIN:.2f} (OOF por caso) ---')
    print(f'  umbral={THR_SENS:.4f}  Sens={m_sens["sensibilidad"]:.4f}  '
          f'Espec={m_sens["especificidad"]:.4f}  F1={m_sens["f1"]:.4f}  '
          f'VP/VN/FP/FN={m_sens["tp"]}/{m_sens["tn"]}/{m_sens["fp"]}/{m_sens["fn"]}')
    res['oof_caso']['umbral_sens_min'] = round(THR_SENS, 4)
    res['oof_caso']['en_sens_min'] = m_sens
    res['sens_min'] = SENS_MIN

    THR = THR_SENS
    print(f'\n  >>> UMBRAL RECOMENDADO (max especificidad con Sens>={SENS_MIN:.2f}): {THR:.4f}')
    print(f'      (Youden daria {res["oof_caso"]["umbral_youden"]:.4f}, pero baja la '
          f'sensibilidad a {res["oof_caso"]["en_youden"]["sensibilidad"]:.4f} — '
          f'{res["oof_caso"]["en_youden"]["fn"]} Chiari perdidos frente a {m_sens["fn"]})')

    # ============ aplicar al test ============
    print('\n' + '=' * 78)
    print(f'  TEST RESERVADO — aplicando el umbral {THR:.4f} fijado en OOF')
    print('=' * 78)
    for nivel in ('caso', 'imagen'):
        if nivel == 'caso':
            _, yt, pt = por_caso(tst, 'prob_ensamble')
        else:
            yt = np.array([int(r['label']) for r in tst])
            pt = np.array([float(r['prob_ensamble']) for r in tst])
        thr_y = res['oof_caso']['umbral_youden']
        t05, ty, tj = met(yt, pt, 0.5), met(yt, pt, thr_y), met(yt, pt, THR)
        print(f'\n--- test por {nivel} (n={len(yt)}) ---')
        print(f'  {"umbral":<26}{"Sens":>9}{"Espec":>9}{"F1":>8}   VP/VN/FP/FN')
        for nom, m in (('0,5 (por defecto)', t05), (f'{thr_y:.3f} (Youden)', ty),
                       (f'{THR:.3f} (recomendado)', tj)):
            print(f'  {nom:<26}{m["sensibilidad"]:>9.4f}{m["especificidad"]:>9.4f}'
                  f'{m["f1"]:>8.4f}   {m["tp"]}/{m["tn"]}/{m["fp"]}/{m["fn"]}')
        res[f'test_{nivel}'] = dict(en_0_5=t05, en_youden=ty, en_umbral_recomendado=tj)

    # ============ figura ROC ============
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, nivel in zip(axes, ('caso', 'imagen')):
        r = res[f'roc_{nivel}']
        o = res[f'oof_{nivel}']
        ax.plot(r['fpr'], r['tpr'], lw=2, color='#1D9E75',
                label=f'OOF por {nivel} (AUC={o["auc"]:.4f})')
        ax.plot([0, 1], [0, 1], '--', color='gray', lw=1)
        thr = np.array(r['thr'])
        i = int(np.argmin(np.abs(thr - o['umbral_youden'])))
        ax.plot(r['fpr'][i], r['tpr'][i], 'o', ms=9, color='#D64545',
                label=f'Youden J (t={o["umbral_youden"]:.3f})')
        i5 = int(np.argmin(np.abs(thr - 0.5)))
        ax.plot(r['fpr'][i5], r['tpr'][i5], 's', ms=8, color='#378ADD', label='umbral 0,5')
        ax.set_xlabel('1 - Especificidad'); ax.set_ylabel('Sensibilidad')
        ax.set_title(f'ROC out-of-fold — por {nivel}', fontsize=10)
        ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=.3)
    plt.tight_layout()
    guardar(plt.gcf(), FIG / 'roc_oof_umbral.png')
    print(f'\n  figura: {FIG / "roc_oof_umbral.png"}')

    # ============ 2. ESTRATIFICACION POR EDAD ============
    print('\n' + '=' * 78)
    print(f'  M-10/M-14 — ERRORES POR GRUPO ETARIO (corte {CORTE_PEDIATRICO:.0f} anios)')
    print('=' * 78)
    claves, y, p = por_caso(oof, 'prob')
    filas = []
    for c, yy, pp in zip(claves, y, p):
        e = edad_num(man.get(c, {}).get('edad'))
        filas.append(dict(clave=c, y=int(yy), p=float(pp), edad=e,
                          grupo=('desconocida' if e is None else
                                 ('pediatrico' if e < CORTE_PEDIATRICO else 'adulto'))))
    est = {}
    for thr_nom, thr in (('0,5', 0.5), (f'{THR:.3f}', THR)):
        print(f'\n--- con umbral {thr_nom} ---')
        tabla = {}
        for g in ('pediatrico', 'adulto', 'desconocida'):
            sub = [f for f in filas if f['grupo'] == g]
            neg = [f for f in sub if f['y'] == 0]
            pos = [f for f in sub if f['y'] == 1]
            fp = sum(1 for f in neg if f['p'] >= thr)
            fn = sum(1 for f in pos if f['p'] < thr)
            tfp, lo, hi = wilson(fp, len(neg)) if neg else (float('nan'),) * 3
            tabla[g] = dict(n_neg=len(neg), fp=fp, tasa_fp=None if not neg else round(tfp, 4),
                            fp_ic95=None if not neg else [round(lo, 4), round(hi, 4)],
                            n_pos=len(pos), fn=fn)
            print(f'  {g:<13}: negativos={len(neg):2d}  FP={fp:2d}  '
                  f'tasa_FP={"n/a" if not neg else f"{tfp:.3f} [{lo:.3f}-{hi:.3f}]":<22}'
                  f'  positivos={len(pos):2d}  FN={fn}')
        ped, adu = tabla['pediatrico'], tabla['adulto']
        if ped['n_neg'] and adu['n_neg']:
            tab = [[ped['fp'], ped['n_neg'] - ped['fp']], [adu['fp'], adu['n_neg'] - adu['fp']]]
            odds, pval = fisher_exact(tab, alternative='two-sided')
            print(f'  Fisher exacto (FP pediatrico vs adulto): tabla={tab}  OR={odds:.3f}  p={pval:.4f}')
            tabla['fisher'] = dict(tabla=tab, odds_ratio=None if np.isinf(odds) else round(float(odds), 4),
                                   p_valor=round(float(pval), 4))
        est[thr_nom] = tabla
    res['estratificacion_edad'] = est
    res['corte_pediatrico'] = CORTE_PEDIATRICO

    # --- prueba sobre la probabilidad continua (no depende del umbral) ---
    print('\n--- probabilidad predicha por grupo etario (independiente del umbral) ---')
    dist = {}
    for clase, lab in ((0, 'NORMAL'), (1, 'CHIARI')):
        pe = [f['p'] for f in filas if f['y'] == clase and f['grupo'] == 'pediatrico']
        ad = [f['p'] for f in filas if f['y'] == clase and f['grupo'] == 'adulto']
        u, pv = mannwhitneyu(pe, ad, alternative='two-sided') if pe and ad else (np.nan, np.nan)
        print(f'  {lab:7s} pediatrico n={len(pe):2d} media={np.mean(pe):.3f} mediana={np.median(pe):.3f}'
              f' | adulto n={len(ad):2d} media={np.mean(ad):.3f} mediana={np.median(ad):.3f}'
              f' | Mann-Whitney p={pv:.4f}')
        dist[lab] = dict(n_ped=len(pe), media_ped=round(float(np.mean(pe)), 4),
                         n_adu=len(ad), media_adu=round(float(np.mean(ad)), 4),
                         mannwhitney_p=round(float(pv), 4))
    res['distribucion_prob_por_edad'] = dist

    # --- fragilidad del Fisher: cuanto cambia p si un caso cambia de lado ---
    t0 = est['0,5']['fisher']['tabla']
    frag = {'observado': round(float(fisher_exact(t0)[1]), 4)}
    for etq, tt in (('un_FP_pediatrico_menos', [[t0[0][0] - 1, t0[0][1] + 1], t0[1]]),
                    ('un_FP_pediatrico_mas', [[t0[0][0] + 1, t0[0][1] - 1], t0[1]]),
                    ('un_FP_adulto_mas', [t0[0], [t0[1][0] + 1, t0[1][1] - 1]])):
        if min(min(r) for r in tt) < 0:
            continue
        frag[etq] = round(float(fisher_exact(tt)[1]), 4)
    print(f'  fragilidad del Fisher (p si un caso cambia de lado): {frag}')
    res['fragilidad_fisher'] = frag

    with open(OUT / 'casos_oof_con_edad.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['clave_agrupacion', 'clase', 'edad', 'grupo_etario', 'prob_media',
                    'pred_0_5', f'pred_{THR:.3f}', 'error_0_5', f'error_{THR:.3f}'])
        for f in filas:
            e5 = (f['p'] >= 0.5) != (f['y'] == 1)
            ej = (f['p'] >= THR) != (f['y'] == 1)
            w.writerow([f['clave'], 'chiari' if f['y'] else 'normal', f['edad'] if f['edad'] is not None else '',
                        f['grupo'], round(f['p'], 4), int(f['p'] >= 0.5), int(f['p'] >= THR),
                        'SI' if e5 else 'NO', 'SI' if ej else 'NO'])

    res['umbral_elegido'] = THR
    for k in list(res):
        if k.startswith('roc_'):
            res[k] = {kk: [round(float(x), 5) for x in vv] for kk, vv in res[k].items()}
    json.dump(res, open(OUT / 'umbral_y_edad.json', 'w'), indent=2, ensure_ascii=False)
    print(f'\nJSON: {OUT / "umbral_y_edad.json"}')
    print(f'CSV : {OUT / "casos_oof_con_edad.csv"}')


if __name__ == '__main__':
    main()
