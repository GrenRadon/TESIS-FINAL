"""Paso 6: reporte comparativo viejo (contaminado) vs nuevo (limpio)."""
import json
from pathlib import Path

OUT = Path('resultados_v2')


def ic(v):
    return f"[{v[0]:.3f}–{v[1]:.3f}]"


def main():
    r = json.load(open(OUT / 'resultados_v2.json', encoding='utf-8'))
    v = r['viejo_contaminado']
    oi, oc = r['primaria_oof']['por_imagen'], r['primaria_oof']['por_caso']
    ti, tc = r['secundaria_test']['por_imagen'], r['secundaria_test']['por_caso']
    L = []
    A = L.append

    A('# Resultado v2 — reentrenamiento sobre datos limpios\n')
    A('Corrección de C-01 (fuga train↔"externo") y C-03 (unidad de análisis) del informe del comité.')
    A('Todo lo demás se mantuvo idéntico a nb_03 para aislar el efecto de arreglar la fuga:')
    A('mismo `cargar_imagen()` (CLAHE→resize→/255), misma DenseNet121, mismos hiperparámetros.\n')

    A('## Comparación cabeza a cabeza\n')
    A('| Métrica | Viejo (contaminado) | Nuevo — OOF por imagen | Nuevo — OOF por caso |')
    A('|---|---|---|---|')
    A(f"| AUC | {v['auc']:.4f} | {oi['auc']:.4f} {ic(oi['auc_ic95']) if 'auc_ic95' in oi else ''} "
      f"| {oc['auc']:.4f} {ic(oc['auc_ic95']) if 'auc_ic95' in oc else ''} |")
    A(f"| Sensibilidad | {v['sensibilidad']:.4f} | {oi['sensibilidad']:.4f} {ic(oi['sens_ic95'])} "
      f"| {oc['sensibilidad']:.4f} {ic(oc['sens_ic95'])} |")
    A(f"| Especificidad | {v['especificidad']:.4f} | {oi['especificidad']:.4f} {ic(oi['espec_ic95'])} "
      f"| {oc['especificidad']:.4f} {ic(oc['espec_ic95'])} |")
    A(f"| F1 | — | {oi['f1']:.4f} | {oc['f1']:.4f} |")
    A(f"| Accuracy | — | {oi['accuracy']:.4f} | {oc['accuracy']:.4f} |")
    A(f"| Matriz (VP/VN/FP/FN) | {v['tp']}/{v['tn']}/{v['fp']}/{v['fn']} "
      f"| {oi['tp']}/{oi['tn']}/{oi['fp']}/{oi['fn']} | {oc['tp']}/{oc['tn']}/{oc['fp']}/{oc['fn']} |")
    A(f"| n evaluado | 131 img (contaminadas) | {oi['n']} img ({oi['n_pos']}+/{oi['n_neg']}−) "
      f"| {oc['n']} casos ({oc['n_pos']}+/{oc['n_neg']}−) |")
    A('')
    A('**El resultado viejo no es comparable**: sus 131 imágenes incluían 27 copias exactas del')
    A('entrenamiento y 3 imágenes Chiari rotuladas como Normal, de modo que su clase negativa no')
    A('contenía ni un solo negativo válido no visto. Se muestra solo como referencia de lo que se')
    A('reportó, no como línea base legítima.\n')

    A('## Métrica primaria — OOF pooled sobre desarrollo\n')
    A(f"Cada uno de los {oc['n']} casos de desarrollo fue predicho por el fold que no lo vio en")
    A('entrenamiento (StratifiedGroupKFold k=5 agrupado por `clave_agrupacion`).\n')
    A('| Nivel | n | AUC | Sens | Espec | F1 | Acc | VP/VN/FP/FN |')
    A('|---|---|---|---|---|---|---|---|')
    for nom, m in (('por imagen', oi), ('por caso', oc)):
        A(f"| {nom} | {m['n']} | {m['auc']:.4f} {ic(m['auc_ic95']) if 'auc_ic95' in m else ''} "
          f"| {m['sensibilidad']:.4f} {ic(m['sens_ic95'])} | {m['especificidad']:.4f} {ic(m['espec_ic95'])} "
          f"| {m['f1']:.4f} | {m['accuracy']:.4f} | {m['tp']}/{m['tn']}/{m['fp']}/{m['fn']} |")
    A('')
    A('IC de sensibilidad/especificidad por método de Wilson; IC del AUC por bootstrap de 2000')
    A('remuestreos **agrupados por caso** (se remuestrean casos, no imágenes).\n')

    A('## Métrica secundaria — test reservado\n')
    A(f"{tc['n']} casos ({tc['n_pos']} Chiari + {tc['n_neg']} Normal) nunca vistos, predichos por el")
    A('ensamble promedio de los 5 modelos de fold.\n')
    A('| Nivel | n | AUC | Sens | Espec | F1 | Acc | VP/VN/FP/FN |')
    A('|---|---|---|---|---|---|---|---|')
    for nom, m in (('por imagen', ti), ('por caso', tc)):
        A(f"| {nom} | {m['n']} | {m['auc']:.4f} {ic(m['auc_ic95']) if 'auc_ic95' in m else ''} "
          f"| {m['sensibilidad']:.4f} {ic(m['sens_ic95'])} | {m['especificidad']:.4f} {ic(m['espec_ic95'])} "
          f"| {m['f1']:.4f} | {m['accuracy']:.4f} | {m['tp']}/{m['tn']}/{m['fp']}/{m['fn']} |")
    A('')
    A(f"⚠️ Con {tc['n_neg']} casos Normal, el IC de Wilson de la especificidad es "
      f"{ic(tc['espec_ic95'])} — un intervalo demasiado ancho para sostener ninguna afirmación.")
    A('Debe declararse explícitamente en el libro, no omitirse.\n')

    A('## Detalle por fold\n')
    A('| Fold | n_train | n_val | class_weight (0/1) | épocas F1+F2 | AUC val | min |')
    A('|---|---|---|---|---|---|---|')
    for f in r['folds']:
        cw = f['class_weight']
        A(f"| {f['fold']} | {f['n_train']} | {f['n_val']} | {cw['0']:.2f}/{cw['1']:.2f} "
          if isinstance(list(cw)[0], str) else
          f"| {f['fold']} | {f['n_train']} | {f['n_val']} | {cw[0]:.2f}/{cw[1]:.2f} ")
        L[-1] += (f"| {f['epocas_f1']}+{f['epocas_f2']} | "
                  f"{f['auc_val'] if f['auc_val'] is not None else 'n/a'} | {f['minutos']} |")
    A('')

    A('## Composición del dataset limpio\n')
    A('| | Casos | Imágenes |')
    A('|---|---|---|')
    A(f"| Desarrollo Chiari | 61 | {oi['n_pos']} |")
    A(f"| Desarrollo Normal | 20 | {oi['n_neg']} |")
    A(f"| Test Chiari | 15 | {ti['n_pos']} |")
    A(f"| Test Normal | 5 | {ti['n_neg']} |")
    A('')
    A('Partición agrupada por `clave_agrupacion` (rID de Radiopaedia para Chiari, URL de caso para')
    A('Normal), semilla 42. Verificación registrada en `verificacion_particion.json`:')
    A('intersección de claves dev∩test vacía **y** intersección de sha256 dev∩test = 0.\n')

    # --- analisis de errores OOF ---
    import csv as _csv
    from collections import defaultdict as _dd
    import numpy as _np
    filas = list(_csv.DictReader(open(OUT / 'predicciones_oof.csv', encoding='utf-8')))
    agg = _dd(lambda: {'p': [], 'y': None})
    for f in filas:
        a = agg[f['clave_agrupacion']]
        a['p'].append(float(f['prob'])); a['y'] = int(f['label'])
    err = [(k, float(_np.mean(v['p'])), v['y'], len(v['p']))
           for k, v in agg.items() if (_np.mean(v['p']) >= 0.5) != (v['y'] == 1)]
    err.sort(key=lambda t: t[1])
    fps = [e for e in err if e[2] == 0]
    ped = [e for e in fps if any(t in e[0].lower() for t in
                                 ('paediatric', 'pediatric', 'month', 'year-old', 'myelination'))]
    borde = [e for e in fps if 0.5 <= e[1] < 0.60]

    A('## Análisis de errores (OOF, por caso)\n')
    A(f'{len(err)} casos mal clasificados de {len(agg)}: {len(fps)} falsos positivos y '
      f'{len(err) - len(fps)} falsos negativos.\n')
    A('| Tipo | p media | n img | Caso |')
    A('|---|---|---|---|')
    for k, p, y, n in err:
        tipo = 'FN' if y == 1 else 'FP'
        A(f"| {tipo} | {p:.3f} | {n} | `{k[:58]}` |")
    A('')
    A(f'Dos patrones a revisar antes de sacar conclusiones clínicas:\n')
    A(f'1. **{len(ped)} de los {len(fps)} falsos positivos son casos pediátricos** '
      '(lactantes, mielinización, 2 y 9 años). El modelo puede estar respondiendo a la anatomía '
      'infantil y no al descenso tonsilar. Los positivos Chiari incluyen niños, así que hay una '
      'correlación de edad que el diseño actual no controla — se cruza con M-10 y M-14.')
    A(f'2. **{len(borde)} de los {len(fps)} falsos positivos caen entre 0,50 y 0,60**, justo por '
      'encima del umbral. El umbral de 0,5 no está justificado (M-08); moverlo cambia mucho la '
      'especificidad. Debe elegirse dentro de validación, nunca mirando el test.\n')

    A('## Archivos\n')
    A('- `manifiesto_final_casos.csv` — 102 filas de caso con rID, clave, split y metadatos clínicos')
    A('- `dataset_nivel_imagen.csv` — dataset limpio a nivel de imagen con `clave_agrupacion` y split')
    A('- `predicciones_oof.csv` / `predicciones_test.csv` — predicciones crudas')
    A('- `resultados_v2.json` — todas las métricas e IC')
    A('- `models_v2/densenet121_v2_fold{1..5}.keras` — los 5 modelos de fold')
    A('- `src/particion.py`, `src/entrenamiento.py`, `src/evaluacion.py` — código reproducible\n')

    (OUT / 'REPORTE_V2.txt').write_text('\n'.join(L), encoding='utf-8')
    print(f'Reporte escrito: {OUT / "REPORTE_V2.txt"}')


if __name__ == '__main__':
    main()
