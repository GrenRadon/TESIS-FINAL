"""C-04 / C17: evalua sobre el test sellado EL ARTEFACTO QUE ESTA EN PRODUCCION.

Hasta ahora el test sellado se habia predicho con el ENSAMBLE de los cinco modelos de
pliegue (columna prob_ensamble de resultados_v4/test_*.csv). Pero lo que se despliega es un
modelo UNICO, reentrenado sobre los 97 casos de desarrollo completos por
entrenamiento_final_v4.py y exportado a ONNX. Son objetos distintos, y el artefacto
desplegado no se habia evaluado nunca sobre el conjunto reservado.

Este script cierra ese hueco. Ademas compara las dos escalas de probabilidad, porque el
umbral 0,1484 se eligio sobre las predicciones out-of-fold del ensamble y se aplica al modelo
unico: si las escalas no coinciden, el punto de operacion no es trasladable sin declararlo.

Se ejecuta el .onnx que esta en app/models/, no un modelo equivalente: es el mismo binario
que responde en la aplicacion.

Escribe resultados_v4/test_desplegado.json.

Uso:  python src/test_modelo_desplegado.py
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocesamiento as PP

V2 = Path('resultados_v2')
V4 = Path('resultados_v4')
APP = Path('app/models')
SALIDA = V4 / 'test_desplegado.json'
CSV_SALIDA = V4 / 'test_desplegado.csv'
Z = 1.959963984540054
N_BOOT = 2000
SEMILLA = 42


def wilson(k, n):
    if n == 0:
        return (float('nan'), float('nan'))
    ph = k / n
    d = 1 + Z ** 2 / n
    c = (ph + Z ** 2 / (2 * n)) / d
    m = Z * math.sqrt(ph * (1 - ph) / n + Z ** 2 / (4 * n ** 2)) / d
    return (round(max(0.0, c - m), 4), round(min(1.0, c + m), 4))


def boot_auc(y, p, grupos):
    """Bootstrap remuestreando CASOS. Con n=20 el intervalo sale ancho; ese es el punto."""
    rng = np.random.default_rng(SEMILLA)
    idx = defaultdict(list)
    for i, g in enumerate(grupos):
        idx[g].append(i)
    claves = list(idx)
    v = []
    for _ in range(N_BOOT):
        sel = []
        for c in rng.choice(len(claves), size=len(claves), replace=True):
            sel.extend(idx[claves[c]])
        ys, ps = y[sel], p[sel]
        if len(set(ys.tolist())) > 1:
            v.append(roc_auc_score(ys, ps))
    if not v:
        return (float('nan'), float('nan')), 0
    return (round(float(np.percentile(v, 2.5)), 4),
            round(float(np.percentile(v, 97.5)), 4)), len(v)


def metricas(y, p, thr, grupos):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    se, slo, shi = tp / (tp + fn) if tp + fn else 0.0, *wilson(tp, tp + fn)
    es, elo, ehi = tn / (tn + fp) if tn + fp else 0.0, *wilson(tn, tn + fp)
    ic, n_ok = boot_auc(y, p, grupos)
    return dict(auc=round(float(roc_auc_score(y, p)), 4), auc_ic95=ic,
                n_bootstrap_validos=n_ok,
                sensibilidad=round(se, 4), sens_ic95=(slo, shi),
                especificidad=round(es, 4), espec_ic95=(elo, ehi),
                f1=round(float(f1_score(y, pred, zero_division=0)), 4),
                vp=int(tp), vn=int(tn), fp=int(fp), fn=int(fn))


def por_caso(y, p, grupos):
    a = defaultdict(lambda: {'p': [], 'y': None})
    for g, pp, yy in zip(grupos, p, y):
        a[g]['p'].append(pp)
        a[g]['y'] = yy
    cl = sorted(a)
    return (np.array([a[c]['y'] for c in cl]),
            np.array([float(np.mean(a[c]['p'])) for c in cl]), cl)


def main():
    cfg = json.loads((APP / 'config_modelo.json').read_text(encoding='utf-8'))
    umbral = cfg['umbral']
    onnx = next(APP.glob('*.onnx'))
    ses = ort.InferenceSession(str(onnx))
    ent = ses.get_inputs()[0].name

    ens = {r['ruta']: float(r['prob_ensamble'])
           for r in csv.DictReader(open(V4 / 'test_sin_flip.csv', encoding='utf-8'))}
    filas = [r for r in csv.DictReader(open(V2 / 'dataset_nivel_imagen.csv', encoding='utf-8'))
             if r['split'] == 'test']

    y, p, g, det = [], [], [], []
    for r in filas:
        t = PP.cargar_imagen(r['ruta'])[None, ...].astype(np.float32)
        pr = float(ses.run(None, {ent: t})[0][0][0])
        y.append(int(r['label']))
        p.append(pr)
        g.append(r['clave_agrupacion'])
        det.append(dict(archivo=r['archivo'], ruta=r['ruta'], clase=r['clase'],
                        label=int(r['label']), clave_agrupacion=r['clave_agrupacion'],
                        prob_desplegado=round(pr, 6),
                        prob_ensamble=round(ens.get(r['ruta'], float('nan')), 6)))
    y, p = np.array(y), np.array(p)

    with open(CSV_SALIDA, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(det[0]))
        w.writeheader()
        w.writerows(det)

    # --- el mismo test, con las dos formas de predecir ---
    pe = np.array([ens[r['ruta']] for r in filas])
    yc, pc, claves = por_caso(y, p, g)
    yce, pce, _ = por_caso(y, pe, g)

    comunes = [d for d in det if not math.isnan(d['prob_ensamble'])]
    dif = np.array([abs(d['prob_desplegado'] - d['prob_ensamble']) for d in comunes])
    corr = float(np.corrcoef([d['prob_desplegado'] for d in comunes],
                             [d['prob_ensamble'] for d in comunes])[0, 1])
    coincide = sum(1 for d in comunes
                   if (d['prob_desplegado'] >= umbral) == (d['prob_ensamble'] >= umbral))

    d = dict(
        artefacto=onnx.name, umbral=umbral,
        entrenamiento_del_artefacto='entrenamiento_final_v4.py --sin-flip: UN modelo '
                                    'reentrenado sobre los 97 casos de desarrollo completos',
        prediccion_del_ensamble='promedio de los 5 modelos de pliegue (columna '
                                'prob_ensamble de test_sin_flip.csv)',
        n_imagenes=len(y), n_casos=len(claves),
        desplegado_por_imagen=metricas(y, p, umbral, g),
        desplegado_por_caso=metricas(yc, pc, umbral, list(claves)),
        ensamble_por_imagen=metricas(y, pe, umbral, g),
        ensamble_por_caso=metricas(yce, pce, umbral, list(claves)),
        comparacion_escalas=dict(
            n=len(comunes), correlacion_pearson=round(corr, 4),
            dif_media=round(float(dif.mean()), 4),
            dif_mediana=round(float(np.median(dif)), 4),
            dif_maxima=round(float(dif.max()), 4),
            clasificaciones_coincidentes=f'{coincide}/{len(comunes)}'),
        nota='El artefacto desplegado y el ensamble son objetos distintos. El umbral 0,1484 '
             'se eligio sobre las predicciones out-of-fold, que produce el ensamble. Esta '
             'comparacion mide si trasladarlo al modelo unico es defendible.')
    SALIDA.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'  artefacto: {onnx.name}  ·  umbral {umbral}')
    print(f'  test sellado: {len(y)} imagenes / {len(claves)} casos')
    for k in ('desplegado_por_imagen', 'desplegado_por_caso',
              'ensamble_por_imagen', 'ensamble_por_caso'):
        m = d[k]
        print(f'  {k:<24} AUC {m["auc"]:.4f} [{m["auc_ic95"][0]:.3f}-{m["auc_ic95"][1]:.3f}]  '
              f'sens {m["sensibilidad"]:.4f}  espec {m["especificidad"]:.4f}  '
              f'VP/VN/FP/FN {m["vp"]}/{m["vn"]}/{m["fp"]}/{m["fn"]}')
    c = d['comparacion_escalas']
    print(f'  escalas: r={c["correlacion_pearson"]:.4f}  |dif| mediana '
          f'{c["dif_mediana"]:.4f}  max {c["dif_maxima"]:.4f}  '
          f'mismas clasificaciones {c["clasificaciones_coincidentes"]}')
    print(f'  JSON: {SALIDA}   CSV: {CSV_SALIDA}')


if __name__ == '__main__':
    main()
