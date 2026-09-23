"""M-06 — tabla de detalle por pliegue de la corrida final, para el libro."""
import json
from pathlib import Path
import numpy as np

V4 = Path('resultados_v4')
ETI = {'base_DenseNet121': 'DenseNet121 (base)', 'base_EfficientNetB0': 'EfficientNetB0 (base)',
       'base_ResNet50V2': 'ResNet50V2 (base)', 'control_es': 'DenseNet121 + early stopping',
       'sin_flip': 'DenseNet121 sin volteo (DESPLEGADA)', 'sin_clahe': 'DenseNet121 sin CLAHE'}


def main():
    d = json.loads((V4 / 'folds_v4.json').read_text(encoding='utf-8'))
    folds = d['folds']
    print(f'Entrenamientos registrados: {len(folds)}  ·  validación cruzada k=5 agrupada por caso\n')
    print(f'| Variante | Pliegue | n_train | n_val | AUC val | class_weight (normal / chiari) |')
    print(f'|---|---|---|---|---|---|')
    filas = []
    for nom, lab in ETI.items():
        for f in [x for x in folds if x['variante'] == nom]:
            cw = f['class_weight']
            c0, c1 = (cw['0'], cw['1']) if isinstance(list(cw)[0], str) else (cw[0], cw[1])
            print(f'| {lab} | {f["fold"]} | {f["n_train"]} | {f["n_val"]} | '
                  f'{f["auc_val"]:.4f} | {c0:.2f} / {c1:.2f} |')
            filas.append(dict(variante=lab, fold=f['fold'], n_train=f['n_train'],
                              n_val=f['n_val'], auc_val=f['auc_val'], cw_normal=c0, cw_chiari=c1))
    cw0 = [r['cw_normal'] for r in filas]; cw1 = [r['cw_chiari'] for r in filas]
    ntr = [r['n_train'] for r in filas]; nva = [r['n_val'] for r in filas]
    print(f'\nRANGOS sobre los {len(filas)} entrenamientos:')
    print(f'  n_train      : {min(ntr)}–{max(ntr)} imágenes')
    print(f'  n_val        : {min(nva)}–{max(nva)} imágenes')
    print(f'  class_weight normal : {min(cw0):.2f}–{max(cw0):.2f}')
    print(f'  class_weight chiari : {min(cw1):.2f}–{max(cw1):.2f}')
    print(f'\n  El class_weight se recalcula en CADA pliegue con `compute_class_weight("balanced")`')
    print(f'  sobre las etiquetas de ese pliegue, por eso varía entre pliegues.')
    ent = d.get('entorno', {})
    print(f'\nEntorno: TF {ent.get("tensorflow")} · Keras {ent.get("keras")} · '
          f'GPUs {ent.get("gpus")} · semilla {ent.get("semilla")} · pipeline v{ent.get("pipeline")}')
    print(f'Épocas fijadas a priori (sin EarlyStopping salvo en la variante de control): {d["epocas"]}')
    json.dump(dict(filas=filas, rangos=dict(
        n_train=[min(ntr), max(ntr)], n_val=[min(nva), max(nva)],
        cw_normal=[round(min(cw0), 2), round(max(cw0), 2)],
        cw_chiari=[round(min(cw1), 2), round(max(cw1), 2)]), entorno=ent, epocas=d['epocas']),
        open(V4 / 'tabla_folds.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\nJSON: {V4 / "tabla_folds.json"}')


if __name__ == '__main__':
    main()
