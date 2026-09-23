"""M-12 — tabla comparativa de la ablacion Dense(16/32/64).

Entrega, para las tres variantes y en el mismo formato que el resto del proyecto:
  - AUC OOF pooled por caso + IC bootstrap 2000 remuestreos AGRUPADO POR CASO
  - Sensibilidad y especificidad + IC de Wilson, en dos puntos de operacion:
      (a) el umbral comun ya fijado sobre el OOF de Dense(32)  <- comparacion principal
      (b) el umbral propio de cada variante (max especificidad con Sens >= 0,95)
  - Comparacion pareada del AUC entre variantes, con IC del delta

ESQUEMA DE EVALUACION: comparacion directa entre variantes mediante
predicciones out-of-fold de cinco pliegues. No se utiliza validacion
anidada; esta limitacion se declara en el informe.
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
VARIANTES = {16: V4 / 'oof_dense16.csv',
             32: V4 / 'oof_base_DenseNet121.csv',   # preespecificado
             64: V4 / 'oof_dense64.csv'}
PREESPECIFICADO = 32


def wilson(k, n):
    if n == 0:
        return (float('nan'),) * 3
    p = k / n
    d = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / d
    m = Z * math.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def por_caso(archivo):
    a = defaultdict(lambda: {'p': [], 'y': None})
    for r in csv.DictReader(open(archivo, encoding='utf-8')):
        d = a[r['clave_agrupacion']]
        d['p'].append(float(r['prob']))
        d['y'] = int(r['label'])
    cl = sorted(a)
    return (np.array(cl), np.array([a[c]['y'] for c in cl]),
            np.array([float(np.mean(a[c]['p'])) for c in cl]))


def auc_ic(y, p, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)   # remuestreo de CASOS
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return (round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)) if vals else (None, None)


def met(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    s, slo, shi = wilson(tp, tp + fn)
    e, elo, ehi = wilson(tn, tn + fp)
    return dict(umbral=round(float(thr), 4),
                sensibilidad=round(s, 4), sens_ic95=[round(slo, 4), round(shi, 4)],
                especificidad=round(e, 4), espec_ic95=[round(elo, 4), round(ehi, 4)],
                f1=round(float(f1_score(y, pred, zero_division=0)), 4),
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def umbral_sens(y, p, smin=SENS_MIN):
    fpr, tpr, thr = roc_curve(y, p)
    v = [(t, f) for t, s, f in zip(thr, tpr, fpr) if s >= smin]
    return float(min(v, key=lambda x: x[1])[0])


def main():
    faltan = [u for u, f in VARIANTES.items() if not f.exists()]
    if faltan:
        raise SystemExit(f'faltan las corridas de Dense{faltan}')

    datos = {u: por_caso(f) for u, f in VARIANTES.items()}
    _, y_ref, p_ref = datos[PREESPECIFICADO]
    THR_COMUN = umbral_sens(y_ref, p_ref)

    print('=' * 96)
    print('  M-12 — ABLACION DESCRIPTIVA DE LA CAPA DENSA (analisis de sensibilidad)')
    print(f'  Esquema: comparacion directa sobre el mismo OOF de 5 folds. NO es validacion anidada.')
    print(f'  Semilla {SEED} · bootstrap {N_BOOT} remuestreos agrupado por caso · '
          f'umbral comun {THR_COMUN:.4f} (fijado sobre Dense({PREESPECIFICADO}))')
    print('=' * 96)

    res = {'esquema': 'comparacion directa sobre el mismo OOF de 5 folds (no anidada)',
           'semilla': SEED, 'n_bootstrap': N_BOOT, 'umbral_comun': round(THR_COMUN, 4),
           'preespecificado': PREESPECIFICADO, 'variantes': {}}

    print(f'\n  --- (a) umbral comun {THR_COMUN:.4f} ---')
    print(f'  {"Dense":>7}{"AUC OOF":>10}{"IC 95%":>19}{"Sens":>9}{"IC 95%":>17}'
          f'{"Espec":>9}{"IC 95%":>17}   VP/VN/FP/FN')
    for u in sorted(VARIANTES):
        _, y, p = datos[u]
        auc = roc_auc_score(y, p)
        lo, hi = auc_ic(y, p)
        m = met(y, p, THR_COMUN)
        marca = ' *' if u == PREESPECIFICADO else '  '
        ic_auc = '[{:.3f}-{:.3f}]'.format(lo, hi)
        ic_s = '[{:.3f}-{:.3f}]'.format(*m['sens_ic95'])
        ic_e = '[{:.3f}-{:.3f}]'.format(*m['espec_ic95'])
        cm = '{}/{}/{}/{}'.format(m['tp'], m['tn'], m['fp'], m['fn'])
        print(f'  {u:>5}{marca}{auc:>10.4f}{ic_auc:>19}'
              f'{m["sensibilidad"]:>9.4f}{ic_s:>17}'
              f'{m["especificidad"]:>9.4f}{ic_e:>17}   {cm}')
        res['variantes'][str(u)] = dict(auc_oof_caso=round(float(auc), 4), auc_ic95=[lo, hi],
                                        n_casos=int(len(y)), en_umbral_comun=m)

    print(f'\n  --- (b) umbral propio de cada variante (max espec. con Sens >= {SENS_MIN}) ---')
    print(f'  {"Dense":>7}{"umbral":>9}{"Sens":>9}{"Espec":>9}{"F1":>9}   VP/VN/FP/FN')
    for u in sorted(VARIANTES):
        _, y, p = datos[u]
        t = umbral_sens(y, p)
        m = met(y, p, t)
        print(f'  {u:>7}{t:>9.4f}{m["sensibilidad"]:>9.4f}{m["especificidad"]:>9.4f}'
              f'{m["f1"]:>9.4f}   {m["tp"]}/{m["tn"]}/{m["fp"]}/{m["fn"]}')
        res['variantes'][str(u)]['en_umbral_propio'] = m

    print('\n  --- comparacion pareada del AUC (bootstrap agrupado por caso) ---')
    rng = np.random.default_rng(SEED)
    us = sorted(VARIANTES)
    n = len(datos[us[0]][1])
    res['comparaciones'] = {}
    for i in range(len(us)):
        for j in range(i + 1, len(us)):
            a, b = us[i], us[j]
            difs = []
            for _ in range(N_BOOT):
                idx = rng.choice(n, size=n, replace=True)
                ya = datos[a][1][idx]
                if len(set(ya.tolist())) < 2:
                    continue
                difs.append(roc_auc_score(ya, datos[a][2][idx]) -
                            roc_auc_score(datos[b][1][idx], datos[b][2][idx]))
            lo, hi = np.percentile(difs, [2.5, 97.5])
            dist = bool(lo > 0 or hi < 0)
            print(f'    Dense({a}) - Dense({b}): dAUC={np.mean(difs):+.4f} '
                  f'IC95 [{lo:+.4f}, {hi:+.4f}]  -> '
                  f'{"DIFIEREN" if dist else "NO distinguibles"}')
            res['comparaciones'][f'{a}_vs_{b}'] = dict(
                d_auc=round(float(np.mean(difs)), 4),
                ic95=[round(float(lo), 4), round(float(hi), 4)], distinguibles=dist)

    algun = any(v['distinguibles'] for v in res['comparaciones'].values())
    res['conclusion'] = (
        'Al menos un par difiere; revisar antes de concluir.' if algun else
        f'Los datos no permiten distinguir entre Dense(16/32/64) con n={n} casos. '
        f'Se retiene Dense({PREESPECIFICADO}) como decision heuristica preespecificada, '
        f'NO como resultado de una comparacion concluyente. Esta comparacion no esta '
        f'validada de forma anidada (misma limitacion que M-05).')
    print(f'\n  CONCLUSION: {res["conclusion"]}')
    json.dump(res, open(V4 / 'ablacion_m12.json', 'w'), indent=2, ensure_ascii=False)
    print(f'\nJSON: {V4 / "ablacion_m12.json"}')


if __name__ == '__main__':
    main()
