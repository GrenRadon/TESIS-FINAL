"""Puntos de operacion del modelo desplegado, para que la app los muestre sin recalcular.

La especificidad de 0,528 no es un limite del modelo: es consecuencia del punto de corte
elegido (maxima especificidad con sensibilidad >= 0,95). Este script mide el mismo modelo en
varios cortes para que esa decision quede visible y auditable, y no parezca una carencia.

Escribe app/models/curva_operacion.json, junto al .onnx y al config, para que no pueda
desincronizarse del modelo al que describe.

Uso:  python src/curva_operacion.py
"""
import csv, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

V4 = Path('resultados_v4')
APP = Path('app/models')
OOF = V4 / 'oof_sin_flip.csv'
SALIDA = APP / 'curva_operacion.json'
CORTES = [0.25, 0.35, 0.50, 0.65, 0.80, 0.90]
PREVALENCIAS = [0.01, 0.10, 0.30, 0.64]
Z = 1.959963984540054
N_BOOT = 2000
SEMILLA = 42


def wilson(k, n):
    """Intervalo de Wilson para una proporcion. Preferido sobre el normal porque no se
    sale de [0,1] ni colapsa cuando la proporcion se acerca a los extremos."""
    if n == 0:
        return (float('nan'), float('nan'))
    ph = k / n
    d = 1 + Z ** 2 / n
    centro = (ph + Z ** 2 / (2 * n)) / d
    margen = Z * math.sqrt(ph * (1 - ph) / n + Z ** 2 / (4 * n ** 2)) / d
    return (round(max(0.0, centro - margen), 4), round(min(1.0, centro + margen), 4))


def auc_ic(y, p):
    """Bootstrap remuestreando CASOS, no imagenes: las imagenes del mismo caso no son
    observaciones independientes."""
    rng = np.random.default_rng(SEMILLA)
    v = []
    for _ in range(N_BOOT):
        i = rng.choice(len(y), size=len(y), replace=True)
        if len(set(y[i].tolist())) > 1:
            v.append(roc_auc_score(y[i], p[i]))
    return (round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4))


def por_caso(archivo):
    a = defaultdict(lambda: {'p': [], 'y': None})
    for r in csv.DictReader(open(archivo, encoding='utf-8')):
        d = a[r['clave_agrupacion']]
        d['p'].append(float(r['prob']))
        d['y'] = int(r['label'])
    cl = sorted(a)
    return (np.array([a[c]['y'] for c in cl]),
            np.array([float(np.mean(a[c]['p'])) for c in cl]))


def punto(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if tp + fn else 0.0
    es = tn / (tn + fp) if tn + fp else 0.0
    pr = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * pr * se / (pr + se) if pr + se else 0.0
    return dict(umbral=round(float(t), 4), sensibilidad=round(se, 4),
                sens_ic95=wilson(tp, tp + fn), especificidad=round(es, 4),
                espec_ic95=wilson(tn, tn + fp), precision=round(pr, 4),
                precision_ic95=wilson(tp, tp + fp), f1=round(f1, 4),
                exactitud=round((tp + tn) / (tp + tn + fp + fn), 4),
                vp=int(tp), vn=int(tn), fp=int(fp), fn=int(fn),
                vpp=[round(float(se * v / (se * v + (1 - es) * (1 - v))), 4)
                     for v in PREVALENCIAS])


def main():
    umbral = json.loads((APP / 'config_modelo.json').read_text(encoding='utf-8'))['umbral']
    y, p = por_caso(OOF)
    cortes = sorted({round(umbral, 4)} | set(CORTES))
    d = dict(
        fuente=str(OOF).replace('\\', '/'),
        n_casos=int(len(y)), n_positivos=int(y.sum()), n_negativos=int((1 - y).sum()),
        prevalencia=round(float(y.mean()), 4),
        auc=round(float(roc_auc_score(y, p)), 4), auc_ic95=auc_ic(y, p),
        n_bootstrap=N_BOOT, semilla=SEMILLA, z=Z,
        average_precision=round(float(average_precision_score(y, p)), 4),
        ap_linea_base=round(float(y.mean()), 4),
        umbral_desplegado=round(float(umbral), 4),
        regla='maxima especificidad sujeta a sensibilidad >= 0,95, elegida SOLO con las '
              'predicciones out-of-fold del conjunto de desarrollo',
        prevalencias=PREVALENCIAS,
        puntos=[punto(y, p, t) for t in cortes])
    SALIDA.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'  {d["n_casos"]} casos · AUC {d["auc"]:.4f} · AP {d["average_precision"]:.4f} '
          f'(linea base {d["ap_linea_base"]:.4f})')
    print(f'  {"umbral":>9}{"Sens":>8}{"Espec":>8}{"Precis":>8}{"F1":>8}')
    for q in d['puntos']:
        marca = '  <- desplegado' if abs(q['umbral'] - d['umbral_desplegado']) < 1e-9 else ''
        print(f'  {q["umbral"]:>9.4f}{q["sensibilidad"]:>8.4f}{q["especificidad"]:>8.4f}'
              f'{q["precision"]:>8.4f}{q["f1"]:>8.4f}{marca}')
    print(f'  JSON: {SALIDA}')


if __name__ == '__main__':
    main()
