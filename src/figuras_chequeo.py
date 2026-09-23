"""Reconstruye las dos figuras de verificacion visual que se habian generado sin script.

  1. chequeo_flip.png       (M-16) — el volteo horizontal sobre un corte SAGITAL invierte el
                                     eje antero-posterior, no la lateralidad
  2. chequeo_alineacion.png (M-01) — comprobacion visual de que la fila N del manifiesto
                                     corresponde al archivo normal_N y no a normal_(N-1)

Ambas son evidencia citada en el texto, asi que deben ser regenerables.
"""
import sys, csv
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifiesto import leer_manifiesto

OUT = Path('resultados_v2')
FIG = OUT / 'figures'
EJEMPLOS_FLIP = ['chiari_13a.jpg', 'normal_2a.png', 'chiari_50a.jpg']
RANGO_ALINEACION = range(8, 27)   # tramo donde la hipotesis A y la B difieren


def rutas():
    return {r['archivo']: r['ruta'] for r in
            csv.DictReader(open(OUT / 'dataset_nivel_imagen.csv', encoding='utf-8'))}


def fig_flip(rt):
    disp = [n for n in EJEMPLOS_FLIP if n in rt]
    if not disp:
        print('  chequeo_flip: ninguno de los ejemplos esta en el dataset actual')
        return
    f, ax = plt.subplots(2, len(disp), figsize=(3.4 * len(disp), 7))
    ax = np.atleast_2d(ax).reshape(2, len(disp))
    for j, nom in enumerate(disp):
        im = np.array(Image.open(rt[nom]).convert('L'))
        ax[0, j].imshow(im, cmap='gray')
        ax[0, j].set_title(nom, fontsize=9)
        ax[1, j].imshow(np.fliplr(im), cmap='gray')
        ax[1, j].set_title('np.fliplr() — lo que ve el modelo\nen el 50% de las epocas',
                           fontsize=8, color='#c44')
        for i in (0, 1):
            ax[i, j].axis('off')
    f.suptitle('Aumento de datos: volteo horizontal sobre cortes SAGITALES',
               fontsize=12, fontweight='bold')
    f.tight_layout()
    guardar(f, FIG / 'chequeo_flip.png')
    plt.close(f)
    print(f'  chequeo_flip.png: {len(disp)} ejemplos')


def fig_alineacion(rt, reg):
    """Muestra normal_N junto a las dos edades candidatas del manifiesto.

    Hipotesis A: el archivo normal_N corresponde a la fila N.
    Hipotesis B: corresponde a la fila N-1 (desplazamiento a partir del caso 7).
    Que el aspecto de la imagen (lactante o adulto) coincida con una sola de las dos
    resuelve la ambiguedad sin abrir el Excel.
    """
    items = []
    for n in RANGO_ALINEACION:
        nom = next((f'normal_{n}{s}.{e}' for s in 'abc' for e in ('png', 'jpg', 'jpeg')
                    if f'normal_{n}{s}.{e}' in rt), None)
        if nom is None:
            continue
        ea = (reg.get(('normal', n)) or {}).get('edad', '?')
        eb = (reg.get(('normal', n - 1)) or {}).get('edad', '?')
        items.append((n, rt[nom], ea or '?', eb or '?'))
    if not items:
        print('  chequeo_alineacion: sin casos en el rango')
        return
    f, ax = plt.subplots(1, len(items), figsize=(2.1 * len(items), 3.2))
    ax = np.atleast_1d(ax)
    for a, (n, ruta, ea, eb) in zip(ax, items):
        a.imshow(np.array(Image.open(ruta).convert('L')), cmap='gray')
        a.set_title(f'normal_{n}\nA:{ea}a  B:{eb}a', fontsize=7)
        a.axis('off')
    f.suptitle('¿Lactante o adulto?  A = archivo N ↔ fila N    B = archivo N ↔ fila N−1',
               fontsize=9)
    f.tight_layout()
    guardar(f, FIG / 'chequeo_alineacion.png')
    plt.close(f)
    print(f'  chequeo_alineacion.png: {len(items)} casos ({items[0][0]}–{items[-1][0]})')


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rt = rutas()
    fig_flip(rt)
    fig_alineacion(rt, leer_manifiesto())


if __name__ == '__main__':
    main()
