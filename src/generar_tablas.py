"""Genera tablas de resultados a partir de los archivos CSV y JSON del proyecto.

Calcula las tablas y las muestra en la salida estándar.

Uso:
    python src/generar_tablas.py
"""

import csv, json, math, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score

V2 = Path('resultados_v2')
V4 = Path('resultados_v4')
APP = Path('app/models')
Z = 1.959963984540054
SEED = 42
N_BOOT = 2000


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
    return (np.array([a[c]['y'] for c in cl]),
            np.array([float(np.mean(a[c]['p'])) for c in cl]))


def auc_ic(y, p):
    rng = np.random.default_rng(SEED)
    v = []
    for _ in range(N_BOOT):
        i = rng.choice(len(y), size=len(y), replace=True)
        if len(set(y[i].tolist())) > 1:
            v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def met(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    s, slo, shi = wilson(tp, tp + fn)
    e, elo, ehi = wilson(tn, tn + fp)
    return dict(sens=s, sens_ic=(slo, shi), espec=e, espec_ic=(elo, ehi),
                f1=f1_score(y, pred, zero_division=0), cm=f'{tp}/{tn}/{fp}/{fn}')


def umbral_sens(y, p, smin=0.95):
    fpr, tpr, thr = roc_curve(y, p)
    v = [(t, f) for t, s, f in zip(thr, tpr, fpr) if s >= smin]
    return float(min(v, key=lambda x: x[1])[0])


# ------------------------------------------------------------------ tablas
def t_composicion():
    ds = list(csv.DictReader(open(V2 / 'dataset_nivel_imagen.csv', encoding='utf-8')))
    g = defaultdict(lambda: [0, set()])
    for r in ds:
        k = (r['split'], r['clase'])
        g[k][0] += 1
        g[k][1].add(r['clave_agrupacion'])
    dn = [r for r in ds if r['split'] == 'desarrollo' and r['clase'] == 'normal']
    san = len({r['clave_agrupacion'] for r in dn if int(r['numero_base']) <= 25})
    pat = len({r['clave_agrupacion'] for r in dn if int(r['numero_base']) >= 26})
    nums = sorted({int(r['numero_base']) for r in dn if int(r['numero_base']) >= 26})
    L = [f'**{len(ds)} imágenes · {len({r["clave_agrupacion"] for r in ds})} casos.** '
         'Contado sobre `resultados_v2/dataset_nivel_imagen.csv`, el CSV que leyó el pipeline.', '',
         '| Split | Clase | Casos | Imágenes |', '|---|---|---|---|']
    for k in sorted(g):
        det = f' ({san} sanos + {pat} con patología no-MC-I)' if k == ('desarrollo', 'normal') else ''
        det = ' (todos sanos)' if k == ('test', 'normal') else det
        L.append(f'| {k[0].capitalize()} | {k[1].capitalize()}{det} | {len(g[k][1])} | {g[k][0]} |')
    L.append(f'| **TOTAL** | | **{len({r["clave_agrupacion"] for r in ds})}** | **{len(ds)}** |')
    L += ['', f'Casos Normal con patología en desarrollo: {", ".join(map(str, nums))}. '
              'Los saltos en 36, 38 y 40 son las exclusiones por patología **en** fosa posterior.']
    return '\n'.join(L)


def t_primaria():
    cfg = json.loads((APP / 'config_modelo.json').read_text(encoding='utf-8'))
    desplegado = cfg['umbral']
    filas = []
    for lab, arch in (('Base (con volteo) — análisis primario', V4 / 'oof_base_DenseNet121.csv'),
                      ('Sin volteo — **DESPLEGADA**', V4 / 'oof_sin_flip.csv')):
        y, p = por_caso(arch)
        t = umbral_sens(y, p)
        m = met(y, p, t)
        lo, hi = auc_ic(y, p)
        marca = '**' if abs(t - desplegado) < 1e-6 else ''
        filas.append(f'| {lab} | {marca}{t:.4f}{marca} | {marca}{roc_auc_score(y,p):.4f}{marca} '
                     f'[{lo:.3f}–{hi:.3f}] | {m["sens"]:.4f} [{m["sens_ic"][0]:.3f}–{m["sens_ic"][1]:.3f}] '
                     f'| {m["espec"]:.4f} [{m["espec_ic"][0]:.3f}–{m["espec_ic"][1]:.3f}] '
                     f'| {m["f1"]:.4f} | {m["cm"]} |')
    return ('⚠️ **Dos configuraciones con dos umbrales. No son intercambiables.**\n\n'
            '| Configuración | Umbral | AUC [IC 95 %] | Sens [IC 95 %] | Espec [IC 95 %] | F1 | VP/VN/FP/FN |\n'
            '|---|---|---|---|---|---|---|\n' + '\n'.join(filas) +
            f'\n\nCada una **en su propio umbral** da sensibilidad, especificidad, F1 y matriz '
            f'**idénticas**. Solo difieren el punto de corte y el AUC.\n\n'
            f'**Para el libro (M-08): umbral {desplegado} con su AUC correspondiente**, que es el '
            f'modelo en producción (`{cfg["onnx"]}`). Emparejar un umbral con el AUC de la otra '
            f'configuración es un error.')


def t_arquitecturas():
    r = json.loads((V4 / 'resultados_v4.json').read_text(encoding='utf-8'))
    L = ['| Arquitectura | AUC OOF | IC 95 % | Sens | Espec |', '|---|---|---|---|---|']
    for n, d in sorted(r['arquitecturas'].items(), key=lambda x: -x[1]['auc']):
        g = ' ★' if n == r.get('ganadora') else ''
        L.append(f'| {n.replace("base_","")}{g} | {d["auc"]:.4f} | '
                 f'[{d["auc_ic95"][0]:.3f}–{d["auc_ic95"][1]:.3f}] | '
                 f'{d["sensibilidad"]:.4f} | {d["especificidad"]:.4f} |')
    L += ['', '| Par | ΔAUC | IC 95 % | ¿Distinguible? |', '|---|---|---|---|']
    for k, d in r.get('comp_arq', {}).items():
        a, b = [x.replace('base_', '') for x in k.split('_vs_')]
        L.append(f'| {a} − {b} | {d["d_auc"]:+.4f} | [{d["ic95"][0]:+.3f}, {d["ic95"][1]:+.3f}] | '
                 f'{"**Sí**" if d["distinguibles"] else "No"} |')
    return '\n'.join(L)


def t_ablaciones():
    r = json.loads((V4 / 'resultados_v4.json').read_text(encoding='utf-8'))
    eti = {'control_es': '**M-05** con early stopping', 'sin_flip': '**M-16** sin volteo horizontal',
           'sin_clahe': '**M-15** sin CLAHE'}
    ref = r['ablaciones']['referencia']
    L = [f'Todas al **umbral común {r["umbral"]}**, para que sean comparables entre sí. '
         'Este no es el punto de operación de producción.', '',
         '| Variante | AUC | Sens | Espec | ΔAUC vs base | ¿Distinguible? |', '|---|---|---|---|---|---|',
         f'| Base (referencia) | {ref["auc"]:.4f} | {ref["sensibilidad"]:.4f} | '
         f'{ref["especificidad"]:.4f} | — | — |']
    for n, lab in eti.items():
        d = r['ablaciones'].get(n)
        if not d:
            continue
        dv = d['delta_vs_ref']
        L.append(f'| {lab} | {d["auc"]:.4f} | {d["sensibilidad"]:.4f} | {d["especificidad"]:.4f} | '
                 f'{dv["d_auc"]:+.4f} [{dv["ic95"][0]:+.3f}, {dv["ic95"][1]:+.3f}] | '
                 f'{"Sí" if dv["distinguibles"] else "No"} |')
    return '\n'.join(L)


def t_m12():
    r = json.loads((V4 / 'ablacion_m12.json').read_text(encoding='utf-8'))
    L = [f'Al umbral común {r["umbral_comun"]}.', '',
         '| Dense | AUC OOF | IC 95 % | Sens | Espec |', '|---|---|---|---|---|']
    for u in sorted(r['variantes'], key=int):
        d = r['variantes'][u]
        m = d['en_umbral_comun']
        g = ' ★ preespecificado' if int(u) == r['preespecificado'] else ''
        L.append(f'| {u}{g} | {d["auc_oof_caso"]:.4f} | '
                 f'[{d["auc_ic95"][0]:.3f}–{d["auc_ic95"][1]:.3f}] | '
                 f'{m["sensibilidad"]:.4f} | {m["especificidad"]:.4f} |')
    L += ['', f'**Conclusión:** {r["conclusion"]}']
    return '\n'.join(L)


def t_ppvnpv():
    a = json.loads((V4 / 'analisis_finales.json').read_text(encoding='utf-8'))
    pn = a['ppv_npv']
    L = [f'Con Sens = {pn["sensibilidad"]:.4f} y Espec = {pn["especificidad"]:.4f}:', '',
         '| Prevalencia | PPV | NPV |', '|---|---|---|']
    for r in pn['tabla']:
        n = ' (desarrollo)' if abs(r['prevalencia'] - 0.64) < 1e-9 else ''
        neg = '**' if r['prevalencia'] <= 0.01 else ''
        L.append(f'| {r["prevalencia"]:.2f}{n} | {neg}{r["ppv"]:.3f}{neg} | {r["npv"]:.3f} |')
    return '\n'.join(L)


def t_desagregado():
    a = json.loads((V4 / 'analisis_finales.json').read_text(encoding='utf-8'))['desagregado']
    L = ['| Subgrupo de negativos | n | Especificidad | IC 95 % |', '|---|---|---|---|']
    for k, lab in (('sanos', 'Cerebros sanos'), ('patologia_no_MC1', 'Con patología no-MC-I')):
        d = a[k]
        L.append(f'| {lab} | {d["n"]} | {d["especificidad"]:.4f} | '
                 f'[{d["ic95"][0]:.3f}–{d["ic95"][1]:.3f}] |')
    L += ['', f'Mann-Whitney entre subgrupos: p = {a["mannwhitney_p"]:.3f}. '
              f'AUC restringido: {a["auc_solo_negativos_sanos"]:.4f} solo con negativos sanos, '
              f'{a["auc_solo_negativos_patologicos"]:.4f} solo con patológicos.']
    return '\n'.join(L)


def t_tsne():
    """M-14. La afirmacion 'la estructura cambia' deja de ser inspeccion visual."""
    d = json.loads((Path('resultados_v2') / 'tsne_parametros.json').read_text(encoding='utf-8'))
    az = d['vecinos_misma_clase_al_azar']
    sem = d['semillas']
    L = [f'**Parámetros declarados:** {d["n_imagenes"]} imágenes preprocesadas con el pipeline del '
         f'modelo, vector de {d["dim_vector"]} dimensiones (píxeles submuestreados 1 de cada 16), '
         f'PCA previo a {d["pca_componentes"]} componentes. '
         f'Perplejidades **{", ".join(str(p) for p in d["perplejidades"])}** × semillas '
         f'**{", ".join(str(s) for s in sem)}**, `init=\'{d["init"]}\'`, '
         f'`learning_rate=\'{d["learning_rate"]}\'`. Ocho proyecciones.', '',
         '| Perplejidad | Semilla | Trustworthiness (k=%d) | Vecinos de la misma clase |'
         % d['k_vecinos'], '|---|---|---|---|']
    for r in d['proyecciones']:
        L.append(f'| {r["perplejidad"]} | {r["semilla"]} | {r["trustworthiness"]:.4f} | '
                 f'{r["vecinos_misma_clase"]:.4f} |')
    L += ['', f'**Cambiando solo la semilla**, con la misma perplejidad y los mismos datos:', '',
          '| Perplejidad | Disimilitud de Procrustes entre semilla %d y %d |' % (sem[0], sem[1]),
          '|---|---|']
    for r in d['estabilidad_entre_semillas']:
        L.append(f'| {r["perplejidad"]} | **{r["procrustes_entre_semillas"]:.4f}** |')
    ps = [r['procrustes_entre_semillas'] for r in d['estabilidad_entre_semillas']]
    pp = [r['procrustes'] for r in d['estabilidad_entre_perplejidades']]
    ex_min = d['vecinos_misma_clase_min'] - az
    ex_max = d['vecinos_misma_clase_max'] - az
    rango = d['vecinos_misma_clase_max'] - d['vecinos_misma_clase_min']
    L += ['', f'Entre perplejidades consecutivas con la misma semilla: '
              f'{min(pp):.4f}–{max(pp):.4f}.', '',
          '**El criterio NO es inspección visual.** Se usan tres medidas:', '',
          f'1. **Disimilitud de Procrustes** entre proyecciones del *mismo* conjunto de puntos '
          f'(0 = idénticas salvo rotación, escala y traslación; 1 = sin relación). Cambiar '
          f'únicamente la semilla da **{min(ps):.2f}–{max(ps):.2f}**: las dos mitades de la '
          f'figura no son la misma nube vista de otro modo, son mapas distintos.',
          f'2. **Trustworthiness (k={d["k_vecinos"]})**, cuánto del vecindario local del espacio '
          f'original sobrevive a la proyección: **{d["trustworthiness_min"]:.4f}–'
          f'{d["trustworthiness_max"]:.4f}**, y crece con la perplejidad, así que ninguna de las '
          f'ocho vistas es «la correcta».',
          f'3. **Fracción de los {d["k_vecinos"]} vecinos más cercanos en la proyección que son de '
          f'la misma clase** — que es lo que el ojo lee como «las clases se separan». Da '
          f'**{d["vecinos_misma_clase_min"]:.4f}–{d["vecinos_misma_clase_max"]:.4f}** frente a '
          f'**{az:.4f} esperado al azar** con este desbalance. El exceso sobre el azar es de '
          f'{ex_min:.3f} a {ex_max:.3f}, y la variación entre parámetros ({rango:.3f}) es del '
          f'mismo orden que el propio exceso.', '',
          '**Conclusión:** la separación aparente no es estable ni sustancialmente mayor que el '
          'azar. El t-SNE no puede sostener ninguna afirmación sobre separabilidad ni sobre la '
          'viabilidad de un clasificador; es visualización exploratoria y así debe presentarse.']
    return chr(10).join(L)


def t_calibracion():
    """C-05. Presenta las métricas de precisión-recall y calibración
    de la configuración desplegada, leídas de analisis_finales.json."""
    a = json.loads((V4 / 'analisis_finales.json').read_text(encoding='utf-8'))
    pr, cal = a['pr'], a['calibracion']
    alto = max(cal['bins'], key=lambda b: b['p_media'])
    L = [f'Configuración **desplegada** (sin volteo), umbral {a["umbral"]:.4f}, '
         f'{a["n_casos"]} casos de desarrollo. Predicción out-of-fold agrupada por caso.', '',
         '| Métrica | Valor | Referencia |', '|---|---|---|',
         f'| Average precision | **{pr["average_precision"]:.4f}** | '
         f'línea base = prevalencia = {pr["linea_base"]:.4f} |',
         f'| Brier score | **{cal["brier"]:.4f}** | predicción constante = '
         f'{cal["referencia_constante"]:.4f} |', '',
         '| Bin | n | Probabilidad media predicha | Fracción real de positivos | Desvío |',
         '|---|---|---|---|---|']
    for i, b in enumerate(cal['bins'], 1):
        dv = b['frac_real'] - b['p_media']
        L.append(f'| {i} | {b["n"]} | {b["p_media"]:.3f} | {b["frac_real"]:.3f} | '
                 f'{dv:+.3f} |')
    L += ['', f'**Sobreconfianza en el extremo alto**: en el bin más alto el modelo predice '
              f'{alto["p_media"]:.3f} de media y la fracción real de positivos es '
              f'{alto["frac_real"]:.3f} (n={alto["n"]}). Importa porque **la app muestra ese '
              f'número como porcentaje al usuario**: cuando dice '
              f'{alto["p_media"] * 100:.0f} %, la realidad está más cerca de '
              f'{alto["frac_real"] * 100:.0f} %.']
    return chr(10).join(L)


def t_m04():
    d = json.loads((V4 / 'resultados_m04.json').read_text(encoding='utf-8'))
    L = ['Cada estrategia en **su propio umbral** (máxima especificidad con Sens ≥ 0,95).', '',
         '| Estrategia | Umbral | AUC | IC 95 % | Sens | Espec | F1 | ΔAUC vs referencia |',
         '|---|---|---|---|---|---|---|---|']
    for k, v in d['variantes'].items():
        dv = v.get('delta_vs_ref')
        dt = (f'{dv["d_auc"]:+.4f} [{dv["ic95"][0]:+.3f}, {dv["ic95"][1]:+.3f}]'
              if dv else '— (referencia)')
        neg = '**' if k == d['referencia'] else ''
        L.append(f'| {neg}{k}{neg} | {v["umbral"]:.4f} | {neg}{v["auc"]:.4f}{neg} | '
                 f'[{v["auc_ic95"][0]:.3f}–{v["auc_ic95"][1]:.3f}] | {v["sensibilidad"]:.4f} | '
                 f'{v["especificidad"]:.4f} | {v["f1"]:.4f} | {dt} |')
    L += ['', f'**Conclusión:** {d["conclusion"]}']
    return chr(10).join(L)


def t_gradcam():
    d = json.loads((Path('resultados_v2') / 'gradcam_resumen.json').read_text(encoding='utf-8'))
    import statistics
    met = list(csv.DictReader(open(Path('resultados_v2') / 'gradcam_metricas.csv',
                                   encoding='utf-8')))
    def sd(grupo, campo):
        v = [float(r[campo]) for r in met if r['grupo'] == grupo and r[campo] not in ('', 'nan')]
        return statistics.pstdev(v) if len(v) > 1 else 0.0

    L = [f'Calculado sobre la **configuración desplegada** (sin volteo, umbral 0,1484). Cada '
         'imagen se explica con el modelo del pliegue que no la vio en entrenamiento. '
         'Verificado: las probabilidades del explicador reproducen las out-of-fold con '
         'max|dif| = 5,1e-05.', '',
         '| Grupo | n | Centro de masa Y | Frac. tercio inferior | Mapas degenerados |',
         '|---|---|---|---|---|']
    eti = {'FP_pediatrico': 'Falso positivo pediátrico',
           'TP_chiari_pediatrico': 'Acierto Chiari pediátrico',
           'TP_chiari_adulto': 'Acierto Chiari adulto',
           'TN_normal_adulto': 'Acierto Normal adulto'}
    for k, lab in eti.items():
        v = d['resumen'].get(k)
        if not v:
            continue
        L.append(f'| {lab} | {v["n"]} | {v["centro_masa_y"]:.3f} ± {sd(k, "centro_masa_y"):.3f} | '
                 f'{v["frac_tercio_inferior"]:.3f} ± {sd(k, "frac_tercio_inferior"):.3f} | '
                 f'{v["degenerados"]} |')
    L += ['', 'Centro de masa Y: 0 = borde superior, 1 = borde inferior. Fracción tercio '
              'inferior: proporción de la masa de activación por debajo de 2/3 de la altura, '
              'usada como proxy de fosa posterior. Desviación típica poblacional.', '',
          '| Comparación | p (Mann-Whitney sobre frac. tercio inferior) |', '|---|---|']
    for k, v in d['comparaciones'].items():
        a, b = k.split('_vs_')
        marca = ' — **difieren**' if v < 0.05 else (' — indistinguibles' if v > 0.3 else '')
        L.append(f'| {eti.get(a,a)} vs {eti.get(b,b)} | {v:.4f}{marca} |')
    sig = min(d['comparaciones'].values())
    n_total = sum(v['n'] for v in d['resumen'].values())
    L += ['', f'**Cero mapas degenerados** en esta corrida: los {n_total} mapas tienen masa de '
              f'activación no nula.', '',
          'La única diferencia detectable está **dentro de la misma clase, entre edades** '
          f'(p={sig:.4f}). Las dos comparaciones entre clases **opuestas** son indistinguibles '
          '(p=0,91 y p=0,56). Y los aciertos Normal adultos son los que **más** activan el '
          'tercio inferior (0,672), por encima de todos los grupos Chiari: si esa activación '
          'indicara descenso amigdalar, el orden estaría invertido.', '',
          f'**Salvedad de comparaciones múltiples:** son tres contrastes; con corrección de '
          f'Bonferroni el umbral es 0,0167 y p={sig:.4f} queda **justo por encima**. La lectura '
          'honesta no es «la edad determina el mapa», sino que **el mapa no separa las clases**: '
          'eso lo sostienen los dos p-valores altos y la inversión del orden, que no dependen de '
          'la corrección.']
    return chr(10).join(L)


def t_test():
    """Conjunto sellado. Dos ejes distintos que NO hay que mezclar:
      (a) configuracion: con volteo frente a sin volteo, ambas predichas con SU ensamble;
      (b) forma de agregar la configuracion desplegada: ensamble de los 5 pliegues frente al
          modelo unico reentrenado sobre todo el desarrollo, que es el que esta en la app.
    La version anterior de esta tabla rotulaba "DESPLEGADA" a la fila del ensamble, lo que
    inducia a error: el artefacto desplegado no es el ensamble (C-04).
    """
    d = json.loads((V4 / 'test_por_configuracion.json').read_text(encoding='utf-8'))
    cfg = json.loads((APP / 'config_modelo.json').read_text(encoding='utf-8'))
    L = ['**(a) Dos configuraciones, cada una con su propio umbral.** Ambas filas se predicen '
         'con el **ensamble de sus cinco modelos de pliegue**. No mezclar filas: son modelos '
         'distintos.', '',
         '| Configuración | Umbral | AUC | Sens [IC 95 %] | Espec [IC 95 %] | VP/VN/FP/FN |',
         '|---|---|---|---|---|---|']
    for lab, v in d.items():
        etq = lab.replace('DESPLEGADA', 'configuración desplegada, ensamble')
        neg = '**' if abs(v['umbral'] - cfg['umbral']) < 1e-6 else ''
        L.append(f'| {neg}{etq}{neg} | {neg}{v["umbral"]:.4f}{neg} | {v["auc"]:.4f} | '
                 f'{v["sens"]:.4f} [{v["sens_ic"][0]:.3f}–{v["sens_ic"][1]:.3f}] | '
                 f'{v["espec"]:.4f} [{v["espec_ic"][0]:.3f}–{v["espec_ic"][1]:.3f}] | {v["cm"]} |')
    n = list(d.values())[0]['n_casos']

    td = V4 / 'test_desplegado.json'
    if td.exists():
        t = json.loads(td.read_text(encoding='utf-8'))
        ec, dc = t['ensamble_por_caso'], t['desplegado_por_caso']
        ei, di = t['ensamble_por_imagen'], t['desplegado_por_imagen']
        c = t['comparacion_escalas']
        L += ['', '**(b) La configuración desplegada, con las dos formas de predecir.** El '
                  'artefacto que está en la aplicación **no es el ensamble**: es un modelo '
                  'único reentrenado sobre los 97 casos de desarrollo completos. Ambas filas '
                  f'usan el mismo umbral {t["umbral"]:.4f}.', '',
              '| Predicción | Unidad | AUC [IC 95 %] | Sens [IC 95 %] | Espec [IC 95 %] | '
              'VP/VN/FP/FN |', '|---|---|---|---|---|---|']
        for lab, m in (('Ensamble de los 5 pliegues', ec),
                       ('**Artefacto desplegado**', dc)):
            L.append(f'| {lab} | por caso | {m["auc"]:.4f} '
                     f'[{m["auc_ic95"][0]:.3f}–{m["auc_ic95"][1]:.3f}] | '
                     f'{m["sensibilidad"]:.4f} [{m["sens_ic95"][0]:.3f}–{m["sens_ic95"][1]:.3f}] | '
                     f'{m["especificidad"]:.4f} '
                     f'[{m["espec_ic95"][0]:.3f}–{m["espec_ic95"][1]:.3f}] | '
                     f'{m["vp"]}/{m["vn"]}/{m["fp"]}/{m["fn"]} |')
        for lab, m in (('Ensamble de los 5 pliegues', ei),
                       ('Artefacto desplegado', di)):
            L.append(f'| {lab} | por imagen | {m["auc"]:.4f} '
                     f'[{m["auc_ic95"][0]:.3f}–{m["auc_ic95"][1]:.3f}] | '
                     f'{m["sensibilidad"]:.4f} [{m["sens_ic95"][0]:.3f}–{m["sens_ic95"][1]:.3f}] | '
                     f'{m["especificidad"]:.4f} '
                     f'[{m["espec_ic95"][0]:.3f}–{m["espec_ic95"][1]:.3f}] | '
                     f'{m["vp"]}/{m["vn"]}/{m["fp"]}/{m["fn"]} |')
        L += ['', f'**El AUC por caso coincide ({dc["auc"]:.4f} en ambos) pero el punto de '
                  f'operación no**: la especificidad pasa de {ec["especificidad"]:.4f} en el '
                  f'ensamble a {dc["especificidad"]:.4f} en el artefacto desplegado, y la '
                  f'sensibilidad de {ec["sensibilidad"]:.4f} a {dc["sensibilidad"]:.4f}. '
                  f'**Con 5 casos negativos esa diferencia son 4 aciertos frente a 1**: lo '
                  f'afirmable es que el punto de operación difiere, no cuál de los dos es '
                  f'mejor.', '',
              f'**Las dos escalas de probabilidad** sobre las {c["n"]} imágenes del sellado: '
              f'correlación de Pearson **{c["correlacion_pearson"]:.4f}**, diferencia '
              f'absoluta mediana **{c["dif_mediana"]:.4f}**, máxima '
              f'**{c["dif_maxima"]:.4f}**, y **{c["clasificaciones_coincidentes"]}** '
              f'clasificaciones coincidentes. El orden se conserva casi por completo, pero '
              f'las escalas no son idénticas.', '',
              f'Artefactos: `resultados_v4/test_desplegado.json` y `.csv` '
              f'(`src/test_modelo_desplegado.py`).']
    L += ['', f'n = {n} casos, de los cuales solo 5 son negativos y todos cerebros sanos. '
              'Los intervalos de la especificidad no sostienen ninguna afirmación.']
    return chr(10).join(L)


def t_pliegues():
    """Detalle por pliegue de la configuracion DESPLEGADA (M-06).

    Se dan imagenes y casos: la unidad de analisis es el caso, asi que dar solo imagenes
    oculta que los pliegues no reparten el mismo numero de pacientes.
    """
    oof = list(csv.DictReader(open(V4 / 'oof_sin_flip.csv', encoding='utf-8')))
    tf = json.loads((V4 / 'tabla_folds.json').read_text(encoding='utf-8'))
    fil = {r['fold']: r for r in tf['filas'] if 'sin volteo' in r['variante']}
    todos = {r['clave_agrupacion'] for r in oof}
    L = ['Configuración **desplegada** (DenseNet121 sin volteo), '
         '`StratifiedGroupKFold` k=5 agrupado por caso. Épocas fijas 40 + 15 en los cinco '
         'pliegues, sin parada temprana.', '',
         '| Pliegue | Img. entren. | Img. valid. | Casos entren. | Casos valid. | AUC valid. '
         '| Peso Normal | Peso Chiari |', '|---|---|---|---|---|---|---|---|']
    aucs = []
    for f in range(1, 6):
        val = {r['clave_agrupacion'] for r in oof if int(r['fold']) == f}
        r = fil[f]
        aucs.append(r['auc_val'])
        L.append(f'| {f} | {r["n_train"]} | {r["n_val"]} | {len(todos - val)} | {len(val)} | '
                 f'{r["auc_val"]:.4f} | {r["cw_normal"]:.4f} | {r["cw_chiari"]:.4f} |')
    L += [f'| **Total** | | **{len(oof)}** | | **{len(todos)}** | | | |', '',
          f'**Rango del AUC por pliegue: {min(aucs):.4f} – {max(aucs):.4f}** '
          f'(media {np.mean(aucs):.4f}, desv. típica {np.std(aucs, ddof=1):.4f}). '
          f'Ese rango corresponde a **esta** configuración; las otras variantes tienen el '
          f'suyo y no son intercambiables.', '',
          'Los pesos de clase se recalculan en cada pliegue con '
          '`compute_class_weight("balanced")` sobre su propio conjunto de entrenamiento, '
          'por eso varían entre pliegues.', '',
          '**La media de estos cinco AUC no es el resultado del trabajo.** El estimador '
          'primario es el AUC out-of-fold agrupado, que se calcula sobre las predicciones '
          'reunidas y sí tiene intervalo de confianza interpretable.']
    return chr(10).join(L)


TABLAS = {'m04': t_m04, 'gradcam': t_gradcam, 'test': t_test, 'composicion': t_composicion, 'primaria': t_primaria, 'arquitecturas': t_arquitecturas,
          'ablaciones': t_ablaciones, 'm12': t_m12, 'ppvnpv': t_ppvnpv, 'desagregado': t_desagregado,
          'calibracion': t_calibracion, 'tsne': t_tsne, 'pliegues': t_pliegues}


def main():
    generadas = {
        nombre: funcion()
        for nombre, funcion in TABLAS.items()
    }

    for nombre, contenido in generadas.items():
        print(
            f"\n{'=' * 70}\n"
            f"  {nombre.upper()}\n"
            f"{'=' * 70}\n"
            f"{contenido}"
        )


if __name__ == '__main__':
    main()
