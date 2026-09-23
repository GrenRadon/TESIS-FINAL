"""M-11: comparacion de las 3 arquitecturas bajo el mismo protocolo.

Todo se decide sobre OOF del desarrollo:
  - arquitectura ganadora: mayor AUC OOF por caso
  - umbral: maxima especificidad sujeta a sensibilidad >= SENS_MIN
El test reservado se mira UNA sola vez, al final, con arquitectura y umbral ya fijados.
"""
import csv, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve

V3 = Path('resultados_v3')
ARQS = ['DenseNet121', 'EfficientNetB0', 'ResNet50V2']
SENS_MIN = 0.95
Z = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return (float('nan'),) * 3
    p = k / n
    den = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / den
    m = Z * math.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / den
    return p, max(0.0, c - m), min(1.0, c + m)


def auc_ic(y, p, g, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    por = defaultdict(list)
    for i, k in enumerate(g):
        por[k].append(i)
    cl = list(por)
    vals = []
    for _ in range(n_boot):
        sel = rng.choice(len(cl), size=len(cl), replace=True)
        idx = [i for s in sel for i in por[cl[s]]]
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        vals.append(roc_auc_score(yy, p[idx]))
    return (round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)) if vals else (None, None)


def por_caso(rows, col):
    a = defaultdict(lambda: {'p': [], 'y': None})
    for r in rows:
        d = a[r['clave_agrupacion']]
        d['p'].append(float(r[col]))
        d['y'] = int(r['label'])
    cl = sorted(a)
    return (np.array(cl), np.array([a[c]['y'] for c in cl]),
            np.array([float(np.mean(a[c]['p'])) for c in cl]))


def met(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    s, slo, shi = wilson(tp, tp + fn)
    e, elo, ehi = wilson(tn, tn + fp)
    return dict(umbral=round(float(thr), 4), tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                sensibilidad=round(s, 4), sens_ic95=[round(slo, 4), round(shi, 4)],
                especificidad=round(e, 4), espec_ic95=[round(elo, 4), round(ehi, 4)],
                f1=round(float(f1_score(y, pred, zero_division=0)), 4),
                accuracy=round(float((tp + tn) / len(y)), 4))


def umbral_sens(y, p, smin=SENS_MIN):
    fpr, tpr, thr = roc_curve(y, p)
    v = [(t, f) for t, s, f in zip(thr, tpr, fpr) if s >= smin]
    return float(min(v, key=lambda x: x[1])[0])


def main():
    res, resumen = {}, []
    for arq in ARQS:
        oof = list(csv.DictReader(open(V3 / f'oof_{arq}.csv', encoding='utf-8')))
        cl, y, p = por_caso(oof, 'prob')
        auc = roc_auc_score(y, p)
        lo, hi = auc_ic(y, p, cl)
        thr = umbral_sens(y, p)
        m05, mth = met(y, p, 0.5), met(y, p, thr)
        yi = np.array([int(r['label']) for r in oof])
        pi = np.array([float(r['prob']) for r in oof])
        res[arq] = dict(auc_oof_caso=round(float(auc), 4), auc_ic95=[lo, hi],
                        auc_oof_imagen=round(float(roc_auc_score(yi, pi)), 4),
                        umbral=round(thr, 4), en_0_5=m05, en_umbral=mth)
        resumen.append((arq, auc, lo, hi, thr, mth))

    print('=' * 92)
    print('  M-11 — COMPARACION DE ARQUITECTURAS (OOF por caso, n=81, mismo protocolo)')
    print('=' * 92)
    print(f'  {"Arquitectura":<17}{"AUC OOF":>9}{"IC 95%":>18}{"umbral":>8}'
          f'{"Sens":>8}{"Espec":>8}{"F1":>8}   VP/VN/FP/FN')
    for arq, auc, lo, hi, thr, m in sorted(resumen, key=lambda x: -x[1]):
        print(f'  {arq:<17}{auc:>9.4f}{f"[{lo:.3f}-{hi:.3f}]":>18}{thr:>8.3f}'
              f'{m["sensibilidad"]:>8.4f}{m["especificidad"]:>8.4f}{m["f1"]:>8.4f}'
              f'   {m["tp"]}/{m["tn"]}/{m["fp"]}/{m["fn"]}')

    ganadora = max(resumen, key=lambda x: x[1])[0]
    thr_g = res[ganadora]['umbral']
    print(f'\n  >>> GANADORA por AUC OOF: {ganadora}  (umbral {thr_g:.4f})')

    print('\n  Comparacion pareada (bootstrap agrupado por caso, 2000 remuestreos):')
    oofs = {a: por_caso(list(csv.DictReader(open(V3 / f"oof_{a}.csv", encoding="utf-8"))), 'prob')
            for a in ARQS}
    rng = np.random.default_rng(42)
    cl0 = oofs[ARQS[0]][0]
    for i in range(len(ARQS)):
        for j in range(i + 1, len(ARQS)):
            a, b = ARQS[i], ARQS[j]
            difs = []
            for _ in range(2000):
                sel = rng.choice(len(cl0), size=len(cl0), replace=True)
                ya = oofs[a][1][sel]
                if len(set(ya.tolist())) < 2:
                    continue
                difs.append(roc_auc_score(ya, oofs[a][2][sel]) -
                            roc_auc_score(oofs[b][1][sel], oofs[b][2][sel]))
            lo_, hi_ = np.percentile(difs, [2.5, 97.5])
            sig = 'difieren' if (lo_ > 0 or hi_ < 0) else 'NO distinguibles'
            print(f'    {a:<16} - {b:<16} dAUC={np.mean(difs):+.4f} '
                  f'IC95 [{lo_:+.4f}, {hi_:+.4f}]  -> {sig}')
            res[f'dif_{a}_vs_{b}'] = dict(d_auc=round(float(np.mean(difs)), 4),
                                          ic95=[round(float(lo_), 4), round(float(hi_), 4)],
                                          distinguibles=bool(lo_ > 0 or hi_ < 0))

    print('\n' + '=' * 92)
    print(f'  TEST RESERVADO — solo {ganadora} con umbral {thr_g:.4f} (fijados en OOF)')
    print('=' * 92)
    tst = list(csv.DictReader(open(V3 / f'test_{ganadora}.csv', encoding='utf-8')))
    _, yt, pt = por_caso(tst, 'prob_ensamble')
    mt = met(yt, pt, thr_g)
    print(f'  por caso (n={len(yt)}): AUC={roc_auc_score(yt, pt):.4f}  '
          f'Sens={mt["sensibilidad"]:.4f} {mt["sens_ic95"]}  '
          f'Espec={mt["especificidad"]:.4f} {mt["espec_ic95"]}  '
          f'F1={mt["f1"]:.4f}   VP/VN/FP/FN={mt["tp"]}/{mt["tn"]}/{mt["fp"]}/{mt["fn"]}')
    res['ganadora'] = ganadora
    res['umbral_final'] = thr_g
    res['test_ganadora'] = dict(auc=round(float(roc_auc_score(yt, pt)), 4), **mt)
    json.dump(res, open(V3 / 'resultados_v3.json', 'w'), indent=2, ensure_ascii=False)
    print(f'\nJSON: {V3 / "resultados_v3.json"}')


if __name__ == '__main__':
    main()
