"""Analisis que no requieren reentrenar. Cierran C-05, M-03, M-18 y controlan M-13.

  1. Curva precision-recall (mas informativa que ROC con clases desbalanceadas) — C-05
  2. Calibracion: curva de fiabilidad + Brier score — C-05
     Importa porque la app muestra al usuario un porcentaje como si fuera probabilidad.
  3. PPV/NPV para un rango de prevalencias plausibles — M-03
  4. Especificidad desagregada: negativos sanos vs con patologia no-MC-I — M-18
  5. Control del atajo por tamano de recorte: correlacion prob vs area — M-13
"""
import csv, json, math, re
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
from PIL import Image
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             brier_score_loss, roc_auc_score)

V4 = Path('resultados_v4')
FIG = Path('resultados_v2') / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
# Se analiza la configuracion DESPLEGADA, que es la que reporta el libro.
OOF = V4 / 'oof_sin_flip.csv'
APP = Path('app/models')
Z = 1.959963984540054
PREVALENCIAS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.64]
CASO_SANO_MAX = 25          # normal_1..25 son cerebros sanos; 26+ llevan patologia no-MC-I


def wilson(k, n):
    if n == 0:
        return (float('nan'),) * 3
    p = k / n
    d = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / d
    m = Z * math.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def cargar():
    a = defaultdict(lambda: {'p': [], 'y': None, 'n': None, 'rutas': []})
    for r in csv.DictReader(open(OOF, encoding='utf-8')):
        d = a[r['clave_agrupacion']]
        d['p'].append(float(r['prob']))
        d['y'] = int(r['label'])
        d['n'] = int(re.match(r'^(?:chiari|normal)_(\d+)', r['archivo']).group(1))
        d['rutas'].append(r['ruta'])
    cl = sorted(a)
    return (cl, np.array([a[c]['y'] for c in cl]),
            np.array([float(np.mean(a[c]['p'])) for c in cl]),
            np.array([a[c]['n'] for c in cl]),
            [a[c]['rutas'] for c in cl])



def correlacion_tamano_v2():
    """Correlacion prob-area en la corrida v2 (la contaminada), por imagen y por caso.

    Sirve de referencia "antes" para M-13. Se calcula aqui en vez de citarla de memoria
    porque la cifra que circulaba en los documentos (+0,368) no reproduce desde ningun
    artefacto.
    """
    import os
    from collections import defaultdict as _dd
    ruta = Path('resultados_v2') / 'predicciones_oof.csv'
    if not ruta.exists():
        return {}
    filas = [r for r in csv.DictReader(open(ruta, encoding='utf-8'))
             if os.path.exists(r['ruta'])]
    out = {}
    for clase, val in (('chiari', '1'), ('normal', '0')):
        sub = [r for r in filas if r['label'] == val]
        if len(sub) < 4:
            continue
        ar = [float(np.prod(Image.open(r['ruta']).size)) for r in sub]
        pr = [float(r['prob']) for r in sub]
        rho, pv = spearmanr(ar, pr)
        out[f'{clase}_por_imagen'] = dict(n=len(sub), rho=round(float(rho), 4),
                                          p=round(float(pv), 4),
                                          significativo=bool(pv < 0.05))
        agg = _dd(lambda: {'a': [], 'p': []})
        for r, a_ in zip(sub, ar):
            agg[r['clave_agrupacion']]['a'].append(a_)
            agg[r['clave_agrupacion']]['p'].append(float(r['prob']))
        a2 = [float(np.mean(v['a'])) for v in agg.values()]
        p2 = [float(np.mean(v['p'])) for v in agg.values()]
        rho, pv = spearmanr(a2, p2)
        out[f'{clase}_por_caso'] = dict(n=len(a2), rho=round(float(rho), 4),
                                        p=round(float(pv), 4),
                                        significativo=bool(pv < 0.05))
    out['nota'] = ('Referencia de la corrida v2 (con fuga, 132 imagenes). La correlacion es '
                   'distinguible de cero POR IMAGEN y deja de serlo POR CASO: agregar por '
                   'caso elimina la pseudorreplicacion que la inflaba.')
    return out


def main():
    cl, y, p, nums, rutas = cargar()
    # Umbral EXACTO recalculado, no el redondeado del JSON: redondear a 4 decimales
    # desplaza un caso limitrofe y falsea la sensibilidad (0,9344 en vez de 0,9508).
    from sklearn.metrics import roc_curve
    fpr, tpr, thrs = roc_curve(y, p)
    thr = float(min([(t, f) for t, s_, f in zip(thrs, tpr, fpr) if s_ >= 0.95],
                    key=lambda x: x[1])[0])
    cfg = json.load(open(APP / 'config_modelo.json', encoding='utf-8'))
    print(f'  configuracion desplegada: {cfg["onnx"]}  (config dice umbral {cfg["umbral"]})')
    res = {'umbral': thr, 'n_casos': int(len(y))}
    print(f'  base DenseNet121 · OOF por caso · n={len(y)} · umbral {thr}')

    # ---------- 1. PR ----------
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    base_pr = float(y.mean())
    print('\n=== 1. PRECISION-RECALL (C-05) ===')
    print(f'  Average precision = {ap:.4f}   (linea base = prevalencia = {base_pr:.4f})')
    res['pr'] = dict(average_precision=round(float(ap), 4), linea_base=round(base_pr, 4))

    # ---------- 2. calibracion ----------
    print('\n=== 2. CALIBRACION (C-05) ===')
    brier = brier_score_loss(y, p)
    bins = np.linspace(0, 1, 6)
    idx = np.digitize(p, bins) - 1
    obs, pred_m, cuenta = [], [], []
    for b in range(5):
        m = idx == b
        if m.sum() == 0:
            continue
        obs.append(float(y[m].mean())); pred_m.append(float(p[m].mean())); cuenta.append(int(m.sum()))
    print(f'  Brier score = {brier:.4f}  (0 = perfecto; una prediccion constante da {base_pr*(1-base_pr):.4f})')
    print(f'  {"bin":>18}{"n":>5}{"p media":>10}{"frac. real":>12}')
    for pm, ob, c, b in zip(pred_m, obs, cuenta, range(5)):
        print(f'  [{bins[b]:.1f}-{bins[b+1]:.1f}]{"":>8}{c:>5}{pm:>10.3f}{ob:>12.3f}')
    res['calibracion'] = dict(brier=round(float(brier), 4),
                              referencia_constante=round(base_pr * (1 - base_pr), 4),
                              bins=[dict(p_media=round(a, 3), frac_real=round(b_, 3), n=c)
                                    for a, b_, c in zip(pred_m, obs, cuenta)])

    # ---------- 3. PPV / NPV ----------
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn); espec = tn / (tn + fp)
    print('\n=== 3. PPV / NPV POR PREVALENCIA (M-03) ===')
    print(f'  con Sens={sens:.4f} y Espec={espec:.4f}')
    print(f'  {"prevalencia":>13}{"PPV":>9}{"NPV":>9}')
    ppvnpv = []
    for pr_ in PREVALENCIAS:
        ppv = sens * pr_ / (sens * pr_ + (1 - espec) * (1 - pr_))
        npv = espec * (1 - pr_) / (espec * (1 - pr_) + (1 - sens) * pr_)
        nota = '  <- desarrollo' if abs(pr_ - 0.64) < 1e-9 else ''
        print(f'  {pr_:>13.2f}{ppv:>9.3f}{npv:>9.3f}{nota}')
        ppvnpv.append(dict(prevalencia=pr_, ppv=round(float(ppv), 4), npv=round(float(npv), 4)))
    res['ppv_npv'] = dict(sensibilidad=round(sens, 4), especificidad=round(espec, 4), tabla=ppvnpv)

    # ---------- 4. especificidad desagregada ----------
    print('\n=== 4. ESPECIFICIDAD DESAGREGADA (M-18) ===')
    res['desagregado'] = {}
    grupos = {}
    for lab, mask in (('sanos', (y == 0) & (nums <= CASO_SANO_MAX)),
                      ('patologia_no_MC1', (y == 0) & (nums > CASO_SANO_MAX))):
        pp = p[mask]
        vn = int((pp < thr).sum())
        e, lo, hi = wilson(vn, len(pp))
        print(f'  negativos {lab:<18} n={len(pp):2d}  VN={vn:2d}  '
              f'espec={e:.4f} [{lo:.3f}-{hi:.3f}]  p_media={pp.mean():.3f}')
        res['desagregado'][lab] = dict(n=int(len(pp)), vn=vn, especificidad=round(e, 4),
                                       ic95=[round(lo, 4), round(hi, 4)],
                                       p_media=round(float(pp.mean()), 4))
        grupos[lab] = pp
    u, pv = mannwhitneyu(grupos['sanos'], grupos['patologia_no_MC1'], alternative='two-sided')
    print(f'  Mann-Whitney de la probabilidad entre subgrupos: p={pv:.4f}')
    res['desagregado']['mannwhitney_p'] = round(float(pv), 4)
    for lab, mask in (('solo negativos sanos', (y == 1) | (nums <= CASO_SANO_MAX)),
                      ('solo negativos patologicos', (y == 1) | (nums > CASO_SANO_MAX))):
        sub_y, sub_p = y[mask], p[mask]
        a = roc_auc_score(sub_y, sub_p)
        print(f'  AUC restringido a {lab:<28} n={len(sub_y):3d}  AUC={a:.4f}')
        res['desagregado'][f'auc_{lab.replace(" ", "_")}'] = round(float(a), 4)

    # ---------- 5. control del atajo por tamano ----------
    print('\n=== 5. CONTROL DEL ATAJO POR TAMANO DE RECORTE (M-13) ===')
    res['control_tamano'] = {}
    for clase, val in (('chiari', 1), ('normal', 0)):
        areas, probs = [], []
        for i in range(len(y)):
            if y[i] != val:
                continue
            a_ = [Image.open(r).size for r in rutas[i]]
            areas.append(float(np.mean([w * h for w, h in a_]))); probs.append(p[i])
        rho, pv2 = spearmanr(areas, probs)
        print(f'  {clase:7s} n={len(areas):3d}  rho={rho:+.3f}  p={pv2:.4f}  -> '
              f'{"CORRELACIONA" if pv2 < 0.05 else "sin relacion detectable"}')
        res['control_tamano'][clase] = dict(n=len(areas), rho=round(float(rho), 4),
                                            p=round(float(pv2), 4), significativo=bool(pv2 < 0.05))
    # La "referencia antes de esta corrida" estaba ESCRITA A MANO (+0,368 / p=0,0001) y no
    # reproducia desde ningun artefacto. Se recalcula aqui sobre las predicciones v2, que si
    # existen, y se guarda. Ademas se hace por imagen y por caso: la correlacion pierde
    # significacion al agregar por caso, que es justo el efecto de la pseudorreplicacion.
    res['control_tamano_v2'] = correlacion_tamano_v2()
    for clave, d2 in res['control_tamano_v2'].items():
        if isinstance(d2, dict):
            print(f'  v2 contaminada · {clave:<18} n={d2["n"]:3d} rho={d2["rho"]:+.4f} '
                  f'p={d2["p"]:.4f}')

    # ---------- figuras ----------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax[0].plot(rec, prec, lw=2, color='#1D9E75', label=f'AP={ap:.4f}')
    ax[0].axhline(base_pr, ls='--', color='gray', lw=1, label=f'línea base={base_pr:.3f}')
    ax[0].set_xlabel('Recall (sensibilidad)'); ax[0].set_ylabel('Precisión (PPV)')
    ax[0].set_title('Curva precisión-recall — OOF por caso', fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_ylim(0, 1.02)

    ax[1].plot([0, 1], [0, 1], '--', color='gray', lw=1, label='calibración perfecta')
    ax[1].plot(pred_m, obs, 'o-', color='#D64545', lw=2, ms=7, label=f'Brier={brier:.4f}')
    for xm, ym, c in zip(pred_m, obs, cuenta):
        ax[1].annotate(f'n={c}', (xm, ym), textcoords='offset points',
                       xytext=(6, -10), fontsize=7, color='gray')
    ax[1].set_xlabel('Probabilidad predicha (media del bin)')
    ax[1].set_ylabel('Fracción real de positivos')
    ax[1].set_title('Curva de calibración', fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout()
    guardar(plt.gcf(), FIG / 'pr_y_calibracion.png')
    print(f'\nfigura: {FIG / "pr_y_calibracion.png"}')

    json.dump(res, open(V4 / 'analisis_finales.json', 'w'), indent=2, ensure_ascii=False)
    print(f'JSON  : {V4 / "analisis_finales.json"}')


if __name__ == '__main__':
    main()
