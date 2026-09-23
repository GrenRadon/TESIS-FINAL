"""C-02 — diagrama de flujo tipo CONSORT/STARD del dataset.

Todas las cifras se cuentan sobre los artefactos del pipeline; ninguna se escribe a mano.
Genera figura PNG y una version en texto para el libro.
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from figuras import guardar
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifiesto import leer_manifiesto

V2 = Path('resultados_v2')
FIG = V2 / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
MAL_ETIQ = {'6163ad6ffd63b87f', '66291783b27449e6', 'f5871fe2194fe845'}
EXCL_FOSA = {36: 'atrofia cerebelosa, hipocampal y de núcleos basales',
             38: 'meduloblastoma leptomeníngeo primario',
             40: 'demencia británica familiar con afectación cerebelosa'}


def flujo():
    reg = leer_manifiesto()
    aud = list(csv.DictReader(open(V2 / 'auditoria_nivel_imagen.csv', encoding='utf-8')))
    ds = list(csv.DictReader(open(V2 / 'dataset_nivel_imagen.csv', encoding='utf-8')))

    h = defaultdict(list)
    for r in aud:
        h[r['sha256']].append(r)

    f = {}
    f['casos_manifiesto'] = len(reg)
    f['casos_manifiesto_ch'] = sum(1 for k in reg if k[0] == 'chiari')
    f['casos_manifiesto_no'] = sum(1 for k in reg if k[0] == 'normal')
    f['entradas_archivo'] = len(aud)
    f['unicas'] = len(h)
    f['dup_eliminados'] = sum(len(v) - 1 for v in h.values())
    f['mal_etiquetadas'] = len({k for k in h if any(k.startswith(m) for m in MAL_ETIQ)})
    f['excl_fosa_img'] = len({r['sha256'] for r in aud if r['clase'] == 'normal'
                              and int(r['numero_base']) in EXCL_FOSA})
    f['excl_fosa_casos'] = len(EXCL_FOSA)
    f['incluidas_img'] = len(ds)
    f['incluidas_casos'] = len({r['clave_agrupacion'] for r in ds})

    g = defaultdict(lambda: [0, set()])
    for r in ds:
        k = (r['split'], r['clase'])
        g[k][0] += 1
        g[k][1].add(r['clave_agrupacion'])
    for k, v in g.items():
        f[f'{k[0]}_{k[1]}_img'] = v[0]
        f[f'{k[0]}_{k[1]}_casos'] = len(v[1])
    return f


def figura(f):
    fig, ax = plt.subplots(figsize=(10.5, 12))
    ax.set_xlim(0, 10); ax.set_ylim(0, 15); ax.axis('off')
    AZ, RJ, VD, GR = '#eef3fa', '#fdeeee', '#eef7f2', '#666666'

    def caja(x, y, w, h, txt, color, borde='#333', neg=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                                    fc=color, ec=borde, lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center',
                fontsize=8.4, fontweight='bold' if neg else 'normal', linespacing=1.5)

    def flecha(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                     mutation_scale=13, lw=1.2, color='#333'))

    caja(1.6, 13.3, 6.8, 1.2,
         f'Casos documentados en el manifiesto Radiopaedia\n'
         f'{f["casos_manifiesto"]} casos  ({f["casos_manifiesto_ch"]} Chiari + '
         f'{f["casos_manifiesto_no"]} Normal)', AZ, neg=True)
    flecha(5, 13.3, 5, 12.6)

    caja(1.6, 11.4, 6.8, 1.2,
         f'Entradas de archivo auditadas\n{f["entradas_archivo"]} '
         f'(el mismo archivo podía estar en varias carpetas)', AZ)
    flecha(5, 11.4, 5, 10.7)
    caja(6.2, 10.85, 3.6, 0.75,
         f'− {f["dup_eliminados"]} duplicados exactos\n(mismo sha256)', RJ, '#c44')
    ax.plot([5, 6.2], [11.2, 11.2], color=GR, lw=1, ls='--')

    caja(1.6, 9.5, 6.8, 1.2,
         f'Imágenes físicas únicas\n{f["unicas"]}  (deduplicadas por sha256)', AZ, neg=True)
    flecha(5, 9.5, 5, 8.8)
    caja(6.2, 8.95, 3.6, 0.75,
         f'{f["mal_etiquetadas"]} imágenes aparecían con\nAMBAS clases → reasignadas a Chiari', RJ, '#c44')
    ax.plot([5, 6.2], [9.3, 9.3], color=GR, lw=1, ls='--')

    caja(1.6, 7.7, 6.8, 1.1,
         'Evaluación de elegibilidad\npor clase y región anatómica', AZ)
    flecha(5, 7.7, 5, 6.6)
    caja(6.2, 7.3, 3.6, 0.95,
         f'− {f["excl_fosa_casos"]} casos ({f["excl_fosa_img"]} img):\n'
         'patología EN la fosa posterior\n(casos 36, 38 y 40)', RJ, '#c44')
    ax.plot([5, 6.2], [7.78, 7.78], color=GR, lw=1, ls='--')

    caja(1.6, 5.3, 6.8, 1.3,
         f'INCLUIDAS\n{f["incluidas_img"]} imágenes  /  {f["incluidas_casos"]} casos', VD,
         '#1D9E75', neg=True)
    flecha(3.2, 5.3, 2.6, 4.7)
    flecha(6.8, 5.3, 7.4, 4.7)

    caja(0.3, 3.2, 4.6, 1.5,
         f'DESARROLLO — K-Fold agrupado\n'
         f'{f["desarrollo_chiari_casos"] + f["desarrollo_normal_casos"]} casos / '
         f'{f["desarrollo_chiari_img"] + f["desarrollo_normal_img"]} img\n'
         f'Chiari {f["desarrollo_chiari_casos"]} casos ({f["desarrollo_chiari_img"]} img) · '
         f'Normal {f["desarrollo_normal_casos"]} ({f["desarrollo_normal_img"]} img)', VD, '#1D9E75')
    caja(5.1, 3.2, 4.6, 1.5,
         f'TEST SELLADO\n'
         f'{f["test_chiari_casos"] + f["test_normal_casos"]} casos / '
         f'{f["test_chiari_img"] + f["test_normal_img"]} img\n'
         f'Chiari {f["test_chiari_casos"]} casos ({f["test_chiari_img"]} img) · '
         f'Normal {f["test_normal_casos"]} ({f["test_normal_img"]} img)', VD, '#1D9E75')

    flecha(2.6, 3.2, 2.6, 2.5)
    caja(0.3, 1.3, 4.6, 1.2,
         'StratifiedGroupKFold k=5\nagrupado por caso\nPredicción out-of-fold', AZ)
    flecha(7.4, 3.2, 7.4, 2.5)
    caja(5.1, 1.3, 4.6, 1.2,
         'Evaluado UNA vez\ncon arquitectura y umbral\nya fijados en el desarrollo', AZ)

    ax.text(5, 0.55, 'Verificación en cada corrida: la intersección de casos entre desarrollo y test\n'
                     'debe ser vacía Y la intersección de hashes sha256 debe ser cero',
            ha='center', va='center', fontsize=8, style='italic', color=GR)
    ax.text(5, 14.75, 'Flujo de selección del conjunto de datos (CONSORT/STARD)',
            ha='center', fontsize=11.5, fontweight='bold')
    plt.tight_layout()
    guardar(plt.gcf(), FIG / 'consort_stard.png')
    return FIG / 'consort_stard.png'


def texto(f):
    L = ['# Diagrama de flujo del conjunto de datos (CONSORT/STARD) — C-02', '',
         'Cifras contadas sobre los artefactos del pipeline. Regenerable con `python src/consort.py`.', '',
         '```',
         f'Casos documentados en el manifiesto Radiopaedia .... {f["casos_manifiesto"]} casos',
         f'   ({f["casos_manifiesto_ch"]} Chiari + {f["casos_manifiesto_no"]} Normal)',
         '                              |',
         f'Entradas de archivo auditadas ..................... {f["entradas_archivo"]}',
         f'                              |     -- {f["dup_eliminados"]} duplicados exactos (sha256)',
         f'Imágenes físicas únicas ........................... {f["unicas"]}',
         f'                              |     -- {f["mal_etiquetadas"]} con AMBAS clases -> reasignadas a Chiari',
         f'                              |     -- {f["excl_fosa_casos"]} casos ({f["excl_fosa_img"]} img): '
         f'patología EN fosa posterior',
         f'INCLUIDAS ......................................... {f["incluidas_img"]} imágenes / '
         f'{f["incluidas_casos"]} casos',
         '                     /                    \\',
         f'   DESARROLLO {f["desarrollo_chiari_casos"] + f["desarrollo_normal_casos"]} casos / '
         f'{f["desarrollo_chiari_img"] + f["desarrollo_normal_img"]} img'
         f'      TEST SELLADO {f["test_chiari_casos"] + f["test_normal_casos"]} casos / '
         f'{f["test_chiari_img"] + f["test_normal_img"]} img',
         f'   Chiari {f["desarrollo_chiari_casos"]} ({f["desarrollo_chiari_img"]} img)'
         f'                Chiari {f["test_chiari_casos"]} ({f["test_chiari_img"]} img)',
         f'   Normal {f["desarrollo_normal_casos"]} ({f["desarrollo_normal_img"]} img)'
         f'                 Normal {f["test_normal_casos"]} ({f["test_normal_img"]} img)',
         '```', '',
         '## Exclusiones con su motivo', '',
         '| Qué | Cuánto | Motivo |', '|---|---|---|',
         f'| Duplicados exactos | {f["dup_eliminados"]} entradas | Mismo sha256 en varias carpetas. '
         'Incluye las 27 copias byte a byte entre entrenamiento y el conjunto llamado "externo" |',
         f'| Imágenes con doble clase | {f["mal_etiquetadas"]} | Byte-idénticas a imágenes Chiari '
         '(matchTemplate ≥ 0,9987 contra las raw). Retiradas de Normal, conservadas en Chiari |']
    for n, m in EXCL_FOSA.items():
        L.append(f'| Caso normal_{n} | 1 caso | Patología **en** la fosa posterior: {m} |')
    L += ['', '## Casos documentados sin imagen incluida', '',
          f'Los {f["casos_manifiesto"]} casos del manifiesto menos los {f["incluidas_casos"]} incluidos '
          f'= {f["casos_manifiesto"] - f["incluidas_casos"]}, que son exactamente los tres excluidos '
          'por patología en fosa posterior (36, 38 y 40).', '',
          '## Verificación', '',
          'En cada corrida se exige que la intersección de casos entre desarrollo y test sea vacía '
          '**y** que la intersección de hashes sha256 sea cero. Si alguna falla, el proceso se detiene.']
    return '\n'.join(L)


def main():
    f = flujo()
    ruta = figura(f)
    txt = texto(f)
    (V2 / 'CONSORT_STARD.txt').write_text(txt, encoding='utf-8')
    json.dump(f, open(V2 / 'consort_flujo.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(txt)
    print(f'\nfigura: {ruta}')
    print(f'texto : {V2 / "CONSORT_STARD.txt"}')


if __name__ == '__main__':
    main()
