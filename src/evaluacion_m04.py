"""M-04 — comparacion de estrategias frente al desbalance. Descriptiva, no selectiva."""
import csv, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score

V4 = Path('resultados_v4')
Z = 1.959963984540054
SEED, N_BOOT, SENS_MIN = 42, 2000, 0.95
VAR = {'class_weight (DESPLEGADA)': V4 / 'oof_sin_flip.csv',
       'focal loss (a=0,25 g=2)': V4 / 'oof_focal_loss.csv',
       'muestreo balanceado': V4 / 'oof_batch_balanceado.csv'}
REF = 'class_weight (DESPLEGADA)'


def wilson(k, n):
    p = k / n; d = 1 + Z**2 / n; c = (p + Z**2 / (2*n)) / d
    m = Z * math.sqrt(p*(1-p)/n + Z**2/(4*n**2)) / d
    return p, max(0.0, c-m), min(1.0, c+m)


def caso(f):
    a = defaultdict(lambda: {'p': [], 'y': None})
    for r in csv.DictReader(open(f, encoding='utf-8')):
        d = a[r['clave_agrupacion']]; d['p'].append(float(r['prob'])); d['y'] = int(r['label'])
    cl = sorted(a)
    return np.array([a[c]['y'] for c in cl]), np.array([float(np.mean(a[c]['p'])) for c in cl])


def umbral(y, p, s=SENS_MIN):
    fpr, tpr, thr = roc_curve(y, p)
    return float(min([(t, f) for t, ss, f in zip(thr, tpr, fpr) if ss >= s], key=lambda x: x[1])[0])


def auc_ic(y, p):
    rng = np.random.default_rng(SEED); v = []
    for _ in range(N_BOOT):
        i = rng.choice(len(y), size=len(y), replace=True)
        if len(set(y[i].tolist())) > 1:
            v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    d = {k: caso(v) for k, v in VAR.items() if v.exists()}
    res = {'esquema': 'comparacion descriptiva sobre el mismo OOF de 5 pliegues',
           'semilla': SEED, 'referencia': REF, 'variantes': {}}
    print('=' * 100)
    print('  M-04 — ESTRATEGIAS FRENTE AL DESBALANCE (OOF por caso, n=97)')
    print('  Cada una en SU umbral (max especificidad con Sens >= 0,95)')
    print('=' * 100)
    print(f'  {"estrategia":<28}{"umbral":>9}{"AUC":>9}{"IC 95%":>18}{"Sens":>9}{"Espec":>9}{"F1":>9}   VP/VN/FP/FN')
    for k, (y, p) in d.items():
        t = umbral(y, p); pr = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
        s, slo, shi = wilson(tp, tp+fn); e, elo, ehi = wilson(tn, tn+fp)
        lo, hi = auc_ic(y, p); auc = roc_auc_score(y, p)
        ic = '[{:.3f}-{:.3f}]'.format(lo, hi)
        print(f'  {k:<28}{t:>9.4f}{auc:>9.4f}{ic:>18}{s:>9.4f}{e:>9.4f}'
              f'{f1_score(y,pr):>9.4f}   {tp}/{tn}/{fp}/{fn}')
        res['variantes'][k] = dict(umbral=round(t, 4), auc=round(float(auc), 4),
                                   auc_ic95=[round(float(lo), 4), round(float(hi), 4)],
                                   sensibilidad=round(s, 4), sens_ic95=[round(slo, 4), round(shi, 4)],
                                   especificidad=round(e, 4), espec_ic95=[round(elo, 4), round(ehi, 4)],
                                   f1=round(float(f1_score(y, pr)), 4),
                                   cm=f'{tp}/{tn}/{fp}/{fn}')
    print('\n  Comparacion pareada del AUC frente a la referencia:')
    rng = np.random.default_rng(SEED); ya, pa = d[REF]
    for k, (y, p) in d.items():
        if k == REF:
            continue
        difs = []
        for _ in range(N_BOOT):
            i = rng.choice(len(ya), size=len(ya), replace=True)
            if len(set(ya[i].tolist())) > 1:
                difs.append(roc_auc_score(ya[i], pa[i]) - roc_auc_score(y[i], p[i]))
        lo, hi = np.percentile(difs, [2.5, 97.5])
        dist = bool(lo > 0 or hi < 0)
        print(f'    {REF} - {k:<28} dAUC={np.mean(difs):+.4f} '
              f'IC95 [{lo:+.4f}, {hi:+.4f}] -> {"DIFIEREN" if dist else "NO distinguibles"}')
        res['variantes'][k]['delta_vs_ref'] = dict(d_auc=round(float(np.mean(difs)), 4),
                                                   ic95=[round(float(lo), 4), round(float(hi), 4)],
                                                   distinguibles=dist)
    alg = any(v.get('delta_vs_ref', {}).get('distinguibles') for v in res['variantes'].values())
    res['conclusion'] = ('Al menos una estrategia difiere; revisar.' if alg else
        'Los datos no permiten distinguir entre class_weight, focal loss y muestreo balanceado '
        'por lote con n=97 casos. Se retiene class_weight por ser la preespecificada y la '
        'desplegada, NO como resultado de una comparacion concluyente.')
    print(f'\n  CONCLUSION: {res["conclusion"]}')
    json.dump(res, open(V4 / 'resultados_m04.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\nJSON: {V4 / "resultados_m04.json"}')


if __name__ == '__main__':
    main()
