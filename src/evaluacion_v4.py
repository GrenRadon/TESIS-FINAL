"""Evaluacion de la corrida v4: M-11, M-05 (control ES), M-16 (flip), M-15 (CLAHE).

Todo sobre OOF agrupado por caso. El test sellado se mira UNA vez, al final,
con arquitectura y umbral ya fijados.
"""
import csv, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve

V4 = Path('resultados_v4')
SENS_MIN = 0.95
Z = 1.959963984540054
SEED = 42
N_BOOT = 2000
BASES = ['base_DenseNet121', 'base_EfficientNetB0', 'base_ResNet50V2']
ABLACIONES = {'control_es': 'M-05 · con early stopping',
              'sin_flip': 'M-16 · sin volteo horizontal',
              'sin_clahe': 'M-15 · sin CLAHE'}
REFERENCIA = 'base_DenseNet121'


def wilson(k, n):
    if n == 0:
        return (float('nan'),) * 3
    p = k / n
    d = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / d
    m = Z * math.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def por_caso(archivo, col='prob'):
    a = defaultdict(lambda: {'p': [], 'y': None})
    for r in csv.DictReader(open(archivo, encoding='utf-8')):
        d = a[r['clave_agrupacion']]
        d['p'].append(float(r[col]))
        d['y'] = int(r['label'])
    cl = sorted(a)
    return (np.array(cl), np.array([a[c]['y'] for c in cl]),
            np.array([float(np.mean(a[c]['p'])) for c in cl]))


def auc_ic(y, p, seed=SEED):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(N_BOOT):
        i = rng.choice(len(y), size=len(y), replace=True)
        if len(set(y[i].tolist())) > 1:
            v.append(roc_auc_score(y[i], p[i]))
    return (round(float(np.percentile(v, 2.5)), 4),
            round(float(np.percentile(v, 97.5)), 4)) if v else (None, None)


def met(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    s, slo, shi = wilson(tp, tp + fn)
    e, elo, ehi = wilson(tn, tn + fp)
    return dict(umbral=round(float(thr), 4),
                sensibilidad=round(s, 4), sens_ic95=[round(slo, 4), round(shi, 4)],
                especificidad=round(e, 4), espec_ic95=[round(elo, 4), round(ehi, 4)],
                f1=round(float(f1_score(y, pred, zero_division=0)), 4),
                accuracy=round(float((tp + tn) / len(y)), 4),
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def umbral_sens(y, p, smin=SENS_MIN):
    fpr, tpr, thr = roc_curve(y, p)
    v = [(t, f) for t, s, f in zip(thr, tpr, fpr) if s >= smin]
    return float(min(v, key=lambda x: x[1])[0])


def delta_auc(da, db, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(da[1])
    d = []
    for _ in range(N_BOOT):
        i = rng.choice(n, size=n, replace=True)
        if len(set(da[1][i].tolist())) < 2:
            continue
        d.append(roc_auc_score(da[1][i], da[2][i]) - roc_auc_score(db[1][i], db[2][i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return round(float(np.mean(d)), 4), round(float(lo), 4), round(float(hi), 4), bool(lo > 0 or hi < 0)


def fila(nombre, y, p, thr):
    auc = roc_auc_score(y, p)
    lo, hi = auc_ic(y, p)
    m = met(y, p, thr)
    ic = '[{:.3f}-{:.3f}]'.format(lo, hi)
    ics = '[{:.3f}-{:.3f}]'.format(*m['sens_ic95'])
    ice = '[{:.3f}-{:.3f}]'.format(*m['espec_ic95'])
    cm = '{}/{}/{}/{}'.format(m['tp'], m['tn'], m['fp'], m['fn'])
    print(f'  {nombre:<30}{auc:>8.4f}{ic:>18}{m["sensibilidad"]:>8.4f}{ics:>16}'
          f'{m["especificidad"]:>8.4f}{ice:>16}{m["f1"]:>8.4f}   {cm}')
    return dict(auc=round(float(auc), 4), auc_ic95=[lo, hi], **m)


def main():
    datos = {n: por_caso(V4 / f'oof_{n}.csv') for n in BASES + list(ABLACIONES)}
    res = {'sens_min': SENS_MIN, 'semilla': SEED, 'n_bootstrap': N_BOOT}
    cab = (f'  {"":<30}{"AUC":>8}{"IC 95%":>18}{"Sens":>8}{"IC 95%":>16}'
           f'{"Espec":>8}{"IC 95%":>16}{"F1":>8}   VP/VN/FP/FN')

    # ---------- M-11 ----------
    print('=' * 118)
    print('  M-11 — ARQUITECTURAS (OOF por caso, epocas fijas, sin early stopping)')
    print('=' * 118)
    print(cab)
    res['arquitecturas'] = {}
    for n in BASES:
        _, y, p = datos[n]
        res['arquitecturas'][n] = fila(n.replace('base_', ''), y, p, umbral_sens(y, p))
    ganadora = max(BASES, key=lambda n: res['arquitecturas'][n]['auc'])
    print(f'\n  Comparacion pareada del AUC:')
    res['comp_arq'] = {}
    for i in range(len(BASES)):
        for j in range(i + 1, len(BASES)):
            a, b = BASES[i], BASES[j]
            d, lo, hi, dist = delta_auc(datos[a], datos[b])
            print(f'    {a.replace("base_",""):<16} - {b.replace("base_",""):<16} '
                  f'dAUC={d:+.4f} IC95 [{lo:+.4f}, {hi:+.4f}] -> '
                  f'{"DIFIEREN" if dist else "NO distinguibles"}')
            res['comp_arq'][f'{a}_vs_{b}'] = dict(d_auc=d, ic95=[lo, hi], distinguibles=dist)
    print(f'\n  >>> Ganadora por regla preespecificada (mayor AUC OOF): {ganadora}')

    # ---------- umbral ----------
    _, yg, pg = datos[ganadora]
    THR = umbral_sens(yg, pg)
    res['ganadora'] = ganadora
    res['umbral'] = round(THR, 4)
    print(f'  >>> Umbral (max especificidad con Sens >= {SENS_MIN}): {THR:.4f}')

    # ---------- ablaciones ----------
    print('\n' + '=' * 118)
    print(f'  ABLACIONES vs {REFERENCIA} — todas al umbral comun {THR:.4f}')
    print('=' * 118)
    print(cab)
    _, yr, pr = datos[REFERENCIA]
    res['ablaciones'] = {'referencia': fila('referencia (base DenseNet121)', yr, pr, THR)}
    for n, lab in ABLACIONES.items():
        _, y, p = datos[n]
        res['ablaciones'][n] = fila(lab, y, p, THR)
        d, lo, hi, dist = delta_auc(datos[REFERENCIA], datos[n])
        res['ablaciones'][n]['delta_vs_ref'] = dict(d_auc=d, ic95=[lo, hi], distinguibles=dist)
    print('\n  Delta de AUC frente a la referencia (positivo = la referencia es mejor):')
    for n, lab in ABLACIONES.items():
        d = res['ablaciones'][n]['delta_vs_ref']
        print(f'    {lab:<34} dAUC={d["d_auc"]:+.4f} IC95 '
              f'[{d["ic95"][0]:+.4f}, {d["ic95"][1]:+.4f}] -> '
              f'{"DIFIEREN" if d["distinguibles"] else "NO distinguibles"}')

    # ---------- test sellado ----------
    print('\n' + '=' * 118)
    print(f'  TEST SELLADO — solo {ganadora}, umbral {THR:.4f} (ambos fijados en OOF)')
    print('=' * 118)
    _, yt, pt = por_caso(V4 / f'test_{ganadora}.csv', 'prob_ensamble')
    mt = met(yt, pt, THR)
    auct = roc_auc_score(yt, pt)
    print(f'  n={len(yt)} casos  AUC={auct:.4f}  '
          f'Sens={mt["sensibilidad"]:.4f} {mt["sens_ic95"]}  '
          f'Espec={mt["especificidad"]:.4f} {mt["espec_ic95"]}  '
          f'VP/VN/FP/FN={mt["tp"]}/{mt["tn"]}/{mt["fp"]}/{mt["fn"]}')
    res['test'] = dict(auc=round(float(auct), 4), n_casos=int(len(yt)), **mt)

    json.dump(res, open(V4 / 'resultados_v4.json', 'w'), indent=2, ensure_ascii=False)
    print(f'\nJSON: {V4 / "resultados_v4.json"}')


if __name__ == '__main__':
    main()
