"""Figura de la seccion 5.2: de donde sale el umbral.

Distribucion de la probabilidad out-of-fold por clase real, con los dos puntos de corte
marcados. Un solo panel, sin curva ROC: el AUC ya aparece en tres tablas.

La figura tiene que dejar ver tres cosas, y cada una tiene su elemento:
  1. Por que 0,1484 y no 0,5 -> banda sombreada entre ambos, con el recuento de positivos
     que quedan dentro y se perderian al subir el corte.
  2. Por que la especificidad es 0,5278 -> los negativos a la derecha del corte, contados.
  3. Que el solapamiento es real -> los puntos individuales de las dos clases, que se
     entrelazan en todo el rango.

Legibilidad para impresion: el ancho se fija EXACTAMENTE en 16,5 cm (6,496 pulgadas), de
modo que los tamanos de fuente en puntos de matplotlib son los puntos reales en la pagina y
FUENTE = 9 se imprime como 9 pt. La regla de 1/52 del ancho se cumple por construccion.
Subir el dpi no cambia el tamano aparente del texto, solo su nitidez.

Por eso se guarda SIN bbox_inches='tight': el recorte automatico se come entre medio y un
centimetro de ancho, y al recolocar la imagen a 16,5 cm en la pagina el texto se escala y
deja de medir los 9 pt previstos. Los margenes se fijan a mano.

Escribe resultados_v2/figures/distribucion_umbral.png (300 ppp) y su PDF vectorial.

Uso:  python src/figura_umbral.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figuras import guardar

V2 = Path('resultados_v2')
V4 = Path('resultados_v4')
APP = Path('app/models')
FIG = V2 / 'figures'
SALIDA = FIG / 'distribucion_umbral.png'

ANCHO_CM = 16.5
ANCHO_PULG = ANCHO_CM / 2.54
ALTO_PULG = 4.05
FUENTE = 9
UMBRAL_OMISION = 0.50
N_BINS = 20
COL_POS = '#C1553C'      # casos con MC-I
COL_NEG = '#2F7D5F'      # casos sin hallazgos compatibles


def datos():
    """Las 97 probabilidades out-of-fold agrupadas por caso, configuracion desplegada."""
    agg = defaultdict(lambda: {'p': [], 'y': None})
    for r in csv.DictReader(open(V4 / 'oof_sin_flip.csv', encoding='utf-8')):
        a = agg[r['clave_agrupacion']]
        a['p'].append(float(r['prob']))
        a['y'] = int(r['label'])
    cl = sorted(agg)
    y = np.array([agg[c]['y'] for c in cl])
    p = np.array([float(np.mean(agg[c]['p'])) for c in cl])
    return y, p


def main():
    y, p = datos()
    umbral = json.loads((APP / 'config_modelo.json').read_text(encoding='utf-8'))['umbral']
    pos, neg = p[y == 1], p[y == 0]

    fn = int((pos < umbral).sum())
    fp = int((neg >= umbral).sum())
    perdidos = int(((pos >= umbral) & (pos < UMBRAL_OMISION)).sum())
    fn_omision = int((pos < UMBRAL_OMISION).sum())

    plt.rcParams.update({'font.size': FUENTE, 'axes.labelsize': FUENTE,
                         'xtick.labelsize': FUENTE - 0.5,
                         'ytick.labelsize': FUENTE - 0.5,
                         'legend.fontsize': FUENTE - 0.5})
    fig, ax = plt.subplots(figsize=(ANCHO_PULG, ALTO_PULG))
    bins = np.linspace(0, 1, N_BINS + 1)

    # banda entre los dos cortes: es el territorio en disputa
    ax.axvspan(umbral, UMBRAL_OMISION, color='#F0E4CE', alpha=.55, zorder=0, lw=0)

    ax.hist(neg, bins=bins, color=COL_NEG, alpha=.55, zorder=2,
            label=f'Sin hallazgos compatibles (n = {len(neg)})')
    ax.hist(pos, bins=bins, color=COL_POS, alpha=.55, zorder=2,
            label=f'Con MC-I (n = {len(pos)})')
    ax.hist(neg, bins=bins, histtype='step', color=COL_NEG, lw=1.1, zorder=3)
    ax.hist(pos, bins=bins, histtype='step', color=COL_POS, lw=1.1, zorder=3)

    # --- puntos individuales: el solapamiento caso a caso ---
    rng = np.random.default_rng(42)
    alto = max(np.histogram(pos, bins=bins)[0].max(),
               np.histogram(neg, bins=bins)[0].max())
    fila_pos, fila_neg = -alto * 0.10, -alto * 0.22
    ax.scatter(pos, fila_pos + rng.normal(0, alto * 0.015, len(pos)), s=7,
               color=COL_POS, alpha=.8, lw=0, zorder=4)
    ax.scatter(neg, fila_neg + rng.normal(0, alto * 0.015, len(neg)), s=7,
               color=COL_NEG, alpha=.8, lw=0, zorder=4)
    ax.axhline(0, color='#B9C0C9', lw=.7, zorder=1)

    # llave sobre los negativos que caen a la derecha del corte: son los falsos positivos
    yb = fila_neg - alto * 0.075
    ax.plot([umbral, 1.0], [yb, yb], color='#1F5C46', lw=.8, zorder=4)
    for xv in (umbral, 1.0):
        ax.plot([xv, xv], [yb, yb + alto * 0.022], color='#1F5C46', lw=.8, zorder=4)

    # --- los dos cortes ---
    ax.axvline(umbral, color='#1C2430', lw=1.7, zorder=5)
    ax.axvline(UMBRAL_OMISION, color='#1C2430', lw=1.2, ls=(0, (4, 3)), zorder=5)

    tope = alto * 1.38
    ax.text(umbral + 0.013, tope * 0.985,
            f'umbral operativo  {umbral:.4f}'.replace('.', ','),
            fontsize=FUENTE - 1, ha='left', va='top', color='#1C2430',
            fontweight='semibold')
    ax.text(UMBRAL_OMISION + 0.013, tope * 0.835, 'umbral por omisión  0,50',
            fontsize=FUENTE - 1, ha='left', va='top', color='#5A6675')

    # --- las tres lecturas ---
    txt_banda = (f'{perdidos} casos con MC-I\n'
                 f'se perderían al subir\nel corte a 0,50')
    ax.text((umbral + UMBRAL_OMISION) / 2, alto * 0.52, txt_banda,
            fontsize=FUENTE - 1.5, ha='center', va='center', color='#7A4A1E',
            linespacing=1.35)
    ax.text(0.985, yb - alto * 0.055,
            f'{fp} falsos positivos  ·  especificidad 0,5278',
            fontsize=FUENTE - 1.5, ha='right', va='top', color='#1F5C46')
    txt_fn = f'{fn} falsos negativos,\ntodos por debajo de 0,06'
    ax.annotate(txt_fn,
                xy=(0.03, fila_pos + alto * 0.03), xytext=(0.195, alto * 1.00),
                fontsize=FUENTE - 1.5, ha='left', va='center', color='#8C3B26',
                linespacing=1.35,
                arrowprops=dict(arrowstyle='->', lw=.7, color='#8C3B26',
                                connectionstyle='arc3,rad=0.3'))

    ax.set_xlabel('Probabilidad predicha out-of-fold (media por caso)')
    ax.set_ylabel('Número de casos')
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(fila_neg - alto * 0.30, tope)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xticklabels([f'{v:.1f}'.replace('.', ',') for v in np.arange(0, 1.01, 0.1)])
    paso = 10 if alto > 18 else 5
    ax.set_yticks(np.arange(0, alto + paso, paso))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.22, zorder=0)
    ax.plot([], [], 'o', color='#5A6675', ms=3.2, lw=0,
            label='un punto por caso')
    ax.legend(loc='upper center', frameon=False, ncol=3,
              bbox_to_anchor=(0.5, -0.19), handletextpad=.6, columnspacing=1.8)
    # margenes a mano, no tight_layout: el ancho tiene que quedar en los 16,5 cm exactos
    fig.subplots_adjust(left=0.098, right=0.995, top=0.985, bottom=0.295)
    salidas = guardar(fig, SALIDA, bbox_inches=None)
    plt.close(fig)

    print(f'  n = {len(y)} casos · {len(pos)} con MC-I · {len(neg)} sin hallazgos')
    print(f'  umbral operativo {umbral}: VP {int((pos >= umbral).sum())} · FN {fn} · '
          f'FP {fp} · VN {int((neg < umbral).sum())}')
    print(f'  umbral 0,50: FN subiria de {fn} a {fn_omision} '
          f'({perdidos} positivos en la banda)')
    print(f'  ancho {ANCHO_CM} cm ({ANCHO_PULG:.3f} pulg) · fuente base {FUENTE} pt · '
          f'{ANCHO_PULG / 52:.3f} pulg por caracter al criterio de 1/52')
    for s in salidas:
        print(f'  {s}')


if __name__ == '__main__':
    main()
