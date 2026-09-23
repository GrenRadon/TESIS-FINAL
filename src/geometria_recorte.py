"""Mide donde recortaron a mano la fosa posterior, para que la app proponga un recuadro util.

La app necesita un punto de partida razonable (RF02). En vez de inventar la geometria, se
mide: para cada imagen cruda se localiza su recorte manual con correlacion de plantilla, se
delimita la cabeza separandola del fondo y se expresa el recorte en fracciones de esa caja.
Las medianas resultantes son la sugerencia inicial.

Esto NO es deteccion automatica de la region de interes: es una regularidad geometrica del
conjunto con el que se trabajo, y solo sirve como punto de partida que el usuario corrige.
La limitacion C-07 sigue abierta.

Escribe app/models/sugerencia_recorte.json.

Uso:  python src/geometria_recorte.py
"""
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np

RAW = 'data/vf/raw'
RECORTES = ['data/vf/cropped', 'data/vf/val_externo/cropped']
SALIDA = Path('app/models/sugerencia_recorte.json')
CORR_MINIMA = 0.90        # por debajo, la localizacion del recorte no es fiable
UMBRAL_FONDO_REL = 0.10   # fraccion del maximo para separar la cabeza del fondo


def bbox_cabeza(a):
    """Caja que contiene la cabeza, separandola del fondo oscuro."""
    u = max(12.0, float(a.max()) * UMBRAL_FONDO_REL)
    m = a > u
    fil = np.where(m.any(axis=1))[0]
    col = np.where(m.any(axis=0))[0]
    if len(fil) < 8 or len(col) < 8:
        return None
    return int(col[0]), int(fil[0]), int(col[-1]), int(fil[-1])


EXT = {'.jpg', '.jpeg', '.png'}


def pares():
    """Cada imagen cruda con su recorte manual, localizado por correlacion.

    Se recorre el directorio de imagenes CRUDAS, no el dataset final. Los dos conjuntos no
    coinciden y conviene tenerlo presente al citar el numero:
      - hay 5 imagenes del dataset que comparten nombre de archivo con otra (la misma imagen
        catalogada en dos carpetas), asi que 177 imagenes son 172 nombres distintos;
      - hay 3 crudas de casos EXCLUIDOS por patologia en fosa posterior que conservan su
        recorte y entran aqui.
    172 + 3 = 175. Ninguna falla al emparejar.
    """
    for clase in ('chiari', 'normal'):
        for raw in sorted(glob.glob(f'{RAW}/{clase}/*.*')):
            n = os.path.basename(raw)
            if os.path.splitext(n)[1].lower() not in EXT:
                continue          # el directorio contiene algun archivo que no es imagen
            cand = [f'{d}/{clase}/{n}' for d in RECORTES if os.path.exists(f'{d}/{clase}/{n}')]
            if not cand:
                continue
            big = cv2.imread(raw, cv2.IMREAD_GRAYSCALE)
            small = cv2.imread(cand[0], cv2.IMREAD_GRAYSCALE)
            if big is None or small is None:
                continue
            if small.shape[0] > big.shape[0] or small.shape[1] > big.shape[1]:
                continue
            res = cv2.matchTemplate(big, small, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx < CORR_MINIMA:
                continue
            bb = bbox_cabeza(big.astype(np.float32))
            if bb is None:
                continue
            x0, y0, x1, y1 = bb
            aw, ah = x1 - x0, y1 - y0
            if aw < 20 or ah < 20:
                continue
            yield dict(
                fx=(loc[0] + small.shape[1] / 2 - x0) / aw,
                fy=(loc[1] + small.shape[0] / 2 - y0) / ah,
                fw=small.shape[1] / aw, fh=small.shape[0] / ah)


def conciliacion():
    """Deja por escrito de donde sale el numero de pares, que no es 177."""
    import csv as _csv
    ds = list(_csv.DictReader(open('resultados_v2/dataset_nivel_imagen.csv', encoding='utf-8')))
    nom_ds = {r['archivo'] for r in ds}
    crudas = {os.path.basename(f) for c in ('chiari', 'normal')
              for f in glob.glob(f'{RAW}/{c}/*.*')
              if os.path.splitext(f)[1].lower() in EXT}
    return dict(imagenes_dataset=len(ds), nombres_unicos_dataset=len(nom_ds),
                imagenes_crudas=len(crudas),
                crudas_fuera_del_dataset=sorted(crudas - nom_ds),
                dataset_sin_cruda=sorted(nom_ds - crudas))


def main():
    d = list(pares())
    if not d:
        raise SystemExit('no se localizo ningun recorte: revisa las rutas')
    conc = conciliacion()
    r = {}
    for k in ('fx', 'fy', 'fw', 'fh'):
        v = np.array([x[k] for x in d])
        r[k] = dict(mediana=round(float(np.median(v)), 4),
                    p25=round(float(np.percentile(v, 25)), 4),
                    p75=round(float(np.percentile(v, 75)), 4))
    salida = dict(
        n_pares=len(d), correlacion_minima=CORR_MINIMA,
        conciliacion=conc,
        pares_fallidos=conc['imagenes_crudas'] - len(d),
        umbral_fondo_relativo=UMBRAL_FONDO_REL,
        fraccion_centro_x=r['fx'], fraccion_centro_y=r['fy'],
        fraccion_ancho=r['fw'], fraccion_alto=r['fh'],
        nota='Fracciones respecto a la caja que contiene la cabeza. El centro X mediano '
             'de 0,66 refleja que en este conjunto la cabeza mira hacia la izquierda, de '
             'modo que la fosa posterior queda a la derecha. Con una imagen espejada la '
             'sugerencia caera del lado contrario y el usuario debera moverla.',
        limitacion='Regularidad geometrica del conjunto de trabajo, NO deteccion de la '
                   'region de interes. C-07 sigue abierto.',
        nota_recuento='Los 175 pares NO son "175 de 177 con 2 fallos": son TODAS las '
                      'imagenes crudas disponibles, y ninguna fallo al emparejar. La '
                      'diferencia con 177 tiene dos causas: 5 imagenes del dataset comparten '
                      'nombre con otra (177 imagenes = 172 nombres distintos) y 3 crudas '
                      'pertenecen a casos excluidos por patologia en fosa posterior que '
                      'conservan su recorte. 172 + 3 = 175.')
    SALIDA.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'  {len(d)} pares imagen cruda / recorte manual localizados '
          f'(de {conc["imagenes_crudas"]} crudas; fallos: '
          f'{conc["imagenes_crudas"] - len(d)})')
    print(f'  dataset: {conc["imagenes_dataset"]} imagenes = '
          f'{conc["nombres_unicos_dataset"]} nombres unicos; crudas fuera del dataset: '
          f'{conc["crudas_fuera_del_dataset"]}')
    for k, lab in (('fx', 'centro X'), ('fy', 'centro Y'),
                   ('fw', 'ancho'), ('fh', 'alto')):
        print(f'  {lab:<10} mediana {r[k]["mediana"]:.3f}   '
              f'[p25 {r[k]["p25"]:.3f} · p75 {r[k]["p75"]:.3f}]')
    print(f'  JSON: {SALIDA}')


if __name__ == '__main__':
    main()
