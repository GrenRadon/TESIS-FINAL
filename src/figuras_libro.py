"""Reemplazos de las figuras del libro que quedaron invalidadas por las correcciones.

  muestra_dataset.png  -> Figura 5  (§4.2.2)  el muestrario venia del inventario antiguo
  efecto_clahe.png     -> Figura 7  (§4.2.4)  el orden CLAHE/resize cambio (C-06/C-08)
  auc_por_fold.png     -> Figura 14 (§5.1)    el K-Fold antiguo tenia fuga y parada temprana
"""
import sys, csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from figuras import guardar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifiesto import leer_manifiesto
import preprocesamiento as PP

OUT = Path('resultados_v2')
V4 = Path('resultados_v4')
FIG = OUT / 'figures'
RNG = np.random.default_rng(42)
CLAVE_ABL = {'control_es': 'DenseNet121 + early stopping',
             'sin_flip': 'DenseNet121 sin volteo (DESPLEGADA)',
             'sin_clahe': 'DenseNet121 sin CLAHE'}


def filas():
    return list(csv.DictReader(open(OUT / 'dataset_nivel_imagen.csv', encoding='utf-8')))


def muestra_dataset(ds, reg):
    """Cuatro por clase, elegidas al azar con semilla, con la edad y el split."""
    f, ax = plt.subplots(2, 4, figsize=(12, 6.4))
    for i, clase in enumerate(('chiari', 'normal')):
        cand = [r for r in ds if r['clase'] == clase]
        for j, r in enumerate(RNG.choice(cand, 4, replace=False)):
            e = (reg.get((clase, int(r['numero_base']))) or {}).get('edad', '?')
            ax[i, j].imshow(cv2.imread(r['ruta'], cv2.IMREAD_GRAYSCALE), cmap='gray')
            ax[i, j].set_title(f"{r['archivo']}\n{e} años · {r['split']}", fontsize=8)
            ax[i, j].axis('off')
        ax[i, 0].text(-0.08, 0.5, clase.upper(), transform=ax[i, 0].transAxes, rotation=90,
                      va='center', ha='center', fontsize=11, fontweight='bold')
    n_c = len({r['clave_agrupacion'] for r in ds})
    f.suptitle(f'Muestra del conjunto final: {len(ds)} imágenes de {n_c} casos '
               f'(recortes de fosa posterior, sagital T1)', fontsize=12, fontweight='bold')
    f.tight_layout()
    guardar(f, FIG / 'muestra_dataset.png')
    plt.close(f)
    print('  muestra_dataset.png')


def efecto_clahe(ds):
    """Documenta el orden canónico y cuantifica cuánto se aparta del orden del libro.

    El libro aplicaba resize y DESPUES CLAHE. Al ecualizar sobre la imagen ya reducida,
    los tiles de 8x8 cubren una región anatómica distinta, así que el resultado no es el
    mismo. La cuarta viñeta muestra la diferencia absoluta entre ambos órdenes.
    """
    r = next(x for x in ds if x['clase'] == 'chiari')
    orig = cv2.imread(r['ruta'], cv2.IMREAD_GRAYSCALE)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    from PIL import Image
    bien = PP.cargar_imagen(r['ruta'])[:, :, 0]                     # CLAHE -> resize
    chico = np.array(Image.fromarray(orig).resize(PP.IMG_SIZE, Image.LANCZOS))
    mal = cl.apply(chico).astype(np.float32)                        # resize -> CLAHE
    dif = np.abs(bien - mal)

    f, ax = plt.subplots(1, 4, figsize=(15, 4.2))
    for a, (im, t, cm) in zip(ax, [
            (orig, f'a) original {orig.shape[1]}×{orig.shape[0]}', 'gray'),
            (bien, 'b) CLAHE en resolución original\n→ resize 224 LANCZOS  '
                   '(orden vigente)', 'gray'),
            (mal, 'c) resize 224 → CLAHE\n(orden del libro)', 'gray'),
            (dif, f'd) |b − c|   máx {dif.max():.0f} · media {dif.mean():.1f}', 'inferno')]):
        im_ = a.imshow(im, cmap=cm)
        a.set_title(t, fontsize=9)
        a.axis('off')
        if cm == 'inferno':
            f.colorbar(im_, ax=a, fraction=0.046)
    f.suptitle('Preprocesamiento: el orden de CLAHE y del redimensionado no es intercambiable',
               fontsize=12, fontweight='bold', y=1.06)
    f.tight_layout()
    guardar(f, FIG / 'efecto_clahe.png')
    plt.close(f)
    print(f'  efecto_clahe.png  (diferencia media entre órdenes: {dif.mean():.2f}/255)')
    return float(dif.mean()), float(dif.max())


def auc_por_fold():
    """Dispersión del AUC entre pliegues frente al AUC agrupado out-of-fold.

    Reemplaza la 'media ± desviación' del libro: con 28-32 imágenes de validación por
    pliegue, la media de cinco AUC no tiene un intervalo interpretable. El estimador
    primario es el OOF agrupado, marcado con la línea.
    """
    t = json.load(open(V4 / 'tabla_folds.json', encoding='utf-8'))
    res = json.load(open(V4 / 'resultados_v4.json', encoding='utf-8'))
    oof = {**{f'{k.split("_", 1)[1]} (base)': v for k, v in res['arquitecturas'].items()},
           **{CLAVE_ABL[k]: v for k, v in res['ablaciones'].items() if k in CLAVE_ABL}}
    g = defaultdict(list)
    for r in t['filas']:
        g[r['variante']].append(r['auc_val'])
    orden = [k for k in g if '(base)' in k] + [k for k in g if '(base)' not in k]

    f, ax = plt.subplots(figsize=(11, 5.6))
    for i, v in enumerate(orden):
        y = g[v]
        cont = 'early stopping' in v
        col = '#c44' if cont else '#378ADD'
        ax.scatter([i] * len(y), y, s=54, color=col, zorder=3, alpha=.85)
        ax.plot([i - .22, i + .22], [np.mean(y)] * 2, color=col, lw=2, ls=':', zorder=4)
        o = oof.get(v)
        if o:
            ax.plot([i - .22, i + .22], [o['auc']] * 2, color='#111', lw=2.6, zorder=5)
            ax.vlines(i, *o['auc_ic95'], color='#111', lw=1.4, zorder=5)
            ic = f"[{o['auc_ic95'][0]:.2f}–{o['auc_ic95'][1]:.2f}]"
            ax.text(i + .3, o['auc'], f"OOF {o['auc']:.3f}\n{ic}", fontsize=7.2, va='center')
        med = f"media pliegues\n{np.mean(y):.3f}±{np.std(y, ddof=1):.3f}"
        ax.text(i, .485, med, fontsize=7, ha='center', color=col)
    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels([v.replace(' (base)', '') for v in orden], rotation=18, ha='right',
                       fontsize=8.5)
    ax.set_xlim(-.5, len(orden) - .12)
    ax.set_ylabel('AUC')
    ax.set_ylim(0.45, 1.03)
    ax.grid(axis='y', alpha=.3)
    ax.plot([], [], 'o', color='#378ADD', label='AUC de cada pliegue (28–32 imágenes)')
    ax.plot([], [], ls=':', color='#378ADD', label='media de los cinco pliegues')
    ax.plot([], [], color='#111', lw=2.6, label='AUC out-of-fold agrupado, IC 95 % '
            '(bootstrap por caso) — estimador primario')
    ax.plot([], [], 'o', color='#c44', label='variante con parada temprana: el pliegue de '
            'validación eligió la época (no comparable)')
    ax.legend(fontsize=7.6, loc='upper center', bbox_to_anchor=(.5, -.22),
              ncol=2, frameon=False)
    ax.set_title('AUC por pliegue frente al AUC out-of-fold agrupado\n'
                 'la media de cinco AUC no tiene intervalo interpretable; el estimador '
                 'primario es el OOF', fontsize=11, fontweight='bold')
    f.tight_layout()
    guardar(f, FIG / 'auc_por_fold.png')
    plt.close(f)
    todos = [x for v in g.values() for x in v]
    print(f'  auc_por_fold.png  ({len(orden)} variantes · AUC por pliegue entre '
          f'{min(todos):.3f} y {max(todos):.3f})')


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ds = filas()
    muestra_dataset(ds, leer_manifiesto())
    efecto_clahe(ds)
    auc_por_fold()


if __name__ == '__main__':
    main()
