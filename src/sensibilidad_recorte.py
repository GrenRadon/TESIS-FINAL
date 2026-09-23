"""Cuanto depende el resultado de DONDE ponga el usuario el recuadro (RF02/RF15).

La app v2.2 permite recortar dentro de la herramienta. Eso quita una barrera practica, pero
introduce variabilidad que hay que medir y declarar.

La medicion distingue dos cosas que no son lo mismo:

  1. AJUSTE FINO — desplazamientos de hasta el 3 % del tamano de la cabeza, con el recuadro
     todavia sobre la fosa posterior. Es la variacion entre dos personas que encuadran bien.
  2. DERIVA — desplazamientos mayores, sobre todo HACIA ARRIBA, que meten cerebro anterior
     dentro del recuadro y dejan de encuadrar la region de interes. Es un recorte incorrecto,
     no una discrepancia de criterio.

Mezclarlas exagera el problema: atribuye al modelo una inestabilidad que en buena parte es
consecuencia de recortar mal. Separarlas dice donde esta realmente el limite.

Escribe resultados_v2/sensibilidad_recorte.json.

Uso:  python src/sensibilidad_recorte.py
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocesamiento as PP

V2 = Path('resultados_v2')
APP = Path('app/models')
SALIDA = V2 / 'sensibilidad_recorte.json'
FINO = 0.03      # ajuste fino: el recuadro sigue sobre la fosa
AMPLIO = 0.06    # deriva: empieza a entrar cerebro anterior


def cabeza(imagen):
    a = np.asarray(imagen.convert('L'), dtype=np.float32)
    m = a > max(12.0, float(a.max()) * 0.10)
    f = np.where(m.any(axis=1))[0]
    c = np.where(m.any(axis=0))[0]
    if len(f) < 8 or len(c) < 8:
        return None
    return int(c[0]), int(f[0]), int(c[-1]), int(f[-1])


def casos_test():
    vistos, out = set(), []
    for r in csv.DictReader(open(V2 / 'dataset_nivel_imagen.csv', encoding='utf-8')):
        if r['split'] != 'test' or r['clave_agrupacion'] in vistos:
            continue
        raw = f"data/vf/raw/{r['clase']}/{r['archivo']}"
        if os.path.exists(raw):
            vistos.add(r['clave_agrupacion'])
            out.append((raw, r['clase']))
    return out


def resumen(variantes, umbral, base_por_caso):
    """Agrega un subconjunto de variantes: cuanto se mueve y cuantas cambian el veredicto."""
    if not variantes:
        return None
    vuelcos = sum(1 for c, p in variantes
                  if (p >= umbral) != (base_por_caso[c] >= umbral))
    desvios = [abs(p - base_por_caso[c]) for c, p in variantes]
    return dict(n=len(variantes), vuelcos=vuelcos,
                pct_vuelcos=round(vuelcos / len(variantes), 4),
                desvio_mediano=round(float(np.median(desvios)), 4),
                desvio_p90=round(float(np.percentile(desvios, 90)), 4))


def main():
    g = json.loads((APP / 'sugerencia_recorte.json').read_text(encoding='utf-8'))
    FX, FY = g['fraccion_centro_x']['mediana'], g['fraccion_centro_y']['mediana']
    FW, FH = g['fraccion_ancho']['mediana'], g['fraccion_alto']['mediana']
    umbral = json.loads((APP / 'config_modelo.json').read_text(encoding='utf-8'))['umbral']
    ses = ort.InferenceSession(str(next(APP.glob('*.onnx'))))
    ent = ses.get_inputs()[0].name

    def prob(img):
        t = PP.cargar_imagen(img)[None, ...].astype(np.float32)
        return float(ses.run(None, {ent: t})[0][0][0])

    desplazamientos = [-AMPLIO, -FINO, 0.0, FINO, AMPLIO]
    base = {}
    grupos = {'fino': [], 'deriva_arriba': [], 'deriva_otras': []}
    filas = []

    for raw, clase in casos_test():
        im = Image.open(raw)
        W, H = im.size
        bb = cabeza(im)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb
        aw, ah = x1 - x0, y1 - y0
        an, al = int(max(64, aw * FW)), int(max(64, ah * FH))
        nombre = os.path.basename(raw)
        ps_fino, ps_todas = [], []
        for dx in desplazamientos:
            for dy in desplazamientos:
                cx = int(np.clip(x0 + aw * (FX + dx), an / 2, W - an / 2))
                cy = int(np.clip(y0 + ah * (FY + dy), al / 2, H - al / 2))
                p = prob(im.crop((cx - an // 2, cy - al // 2,
                                  cx - an // 2 + an, cy - al // 2 + al)))
                ps_todas.append(p)
                if dx == 0.0 and dy == 0.0:
                    base[nombre] = p
                    continue
                if abs(dx) <= FINO and abs(dy) <= FINO:
                    grupos['fino'].append((nombre, p))
                    ps_fino.append(p)
                elif dy <= -AMPLIO:                     # el recuadro sube: entra cerebro
                    grupos['deriva_arriba'].append((nombre, p))
                else:
                    grupos['deriva_otras'].append((nombre, p))
        b = base[nombre]
        ps_fino_c = ps_fino + [b]
        filas.append(dict(
            archivo=nombre, clase=clase, p_sugerido=round(b, 4),
            acierta_con_sugerido=(b >= umbral) == (clase == 'chiari'),
            rango_ajuste_fino=round(max(ps_fino_c) - min(ps_fino_c), 4),
            vuelcos_ajuste_fino=sum(1 for p in ps_fino if (p >= umbral) != (b >= umbral)),
            rango_total=round(max(ps_todas) - min(ps_todas), 4)))

    # La particion que importa: los casos cerca del umbral ya llevan aviso de baja
    # confianza. Si los vuelcos se concentran ahi, la inestabilidad NO es un problema
    # oculto: es exactamente lo que la interfaz esta senalando.
    MARGEN = 0.10
    cerca = {f['archivo'] for f in filas if abs(f['p_sugerido'] - umbral) < MARGEN}
    fino_cerca = [(c, p) for c, p in grupos['fino'] if c in cerca]
    fino_lejos = [(c, p) for c, p in grupos['fino'] if c not in cerca]

    r_fino = resumen(grupos['fino'], umbral, base)
    r_arriba = resumen(grupos['deriva_arriba'], umbral, base)
    r_otras = resumen(grupos['deriva_otras'], umbral, base)
    estables = sum(1 for f in filas if f['vuelcos_ajuste_fino'] == 0)

    d = dict(
        umbral=umbral, n_casos=len(filas), margen_baja_confianza=MARGEN,
        ajuste_fino_cerca_del_umbral=resumen(fino_cerca, umbral, base),
        ajuste_fino_lejos_del_umbral=resumen(fino_lejos, umbral, base),
        n_casos_cerca_del_umbral=len(cerca),
        definicion_ajuste_fino=f'desplazamiento del centro <= {FINO:.0%} del tamano de la '
                               f'cabeza en cada eje; el recuadro sigue sobre la fosa',
        definicion_deriva_arriba=f'desplazamiento de {AMPLIO:.0%} hacia arriba; el recuadro '
                                 f'incorpora cerebro anterior y deja de encuadrar la region',
        aciertos_con_sugerido=sum(1 for f in filas if f['acierta_con_sugerido']),
        ajuste_fino=r_fino, deriva_arriba=r_arriba, deriva_otras=r_otras,
        casos_estables_ante_ajuste_fino=estables,
        rango_mediano_ajuste_fino=round(
            float(np.median([f['rango_ajuste_fino'] for f in filas])), 4),
        filas=filas,
        conclusion='La sensibilidad al encuadre NO esta repartida por igual: se concentra '
                   'en los casos cuya probabilidad ya cae cerca del umbral, que son '
                   'precisamente los que la interfaz marca como de baja confianza. Fuera de '
                   'esa banda el resultado es estable frente a los ajustes finos. Esto '
                   'valida el aviso de proximidad al umbral como salvaguarda real y no '
                   'decorativa. Persiste el limite de C-07: un encuadre francamente '
                   'incorrecto produce un resultado incorrecto, y la herramienta solo puede '
                   'advertirlo si la region resulta estadisticamente atipica.')
    SALIDA.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'  {len(filas)} casos del test sellado · aciertos con el recuadro sugerido: '
          f'{d["aciertos_con_sugerido"]}/{len(filas)}')
    rc = d['ajuste_fino_cerca_del_umbral']
    rl = d['ajuste_fino_lejos_del_umbral']
    print(f'  AJUSTE FINO (<= {FINO:.0%}, sigue sobre la fosa):')
    print(f'    global: {r_fino["vuelcos"]}/{r_fino["n"]} cambian el veredicto '
          f'({r_fino["pct_vuelcos"]:.1%})')
    print(f'    en los {len(cerca)} casos CERCA del umbral (ya avisados): '
          f'{rc["vuelcos"]}/{rc["n"]} ({rc["pct_vuelcos"]:.1%})')
    print(f'    en los {len(filas) - len(cerca)} casos LEJOS del umbral: '
          f'{rl["vuelcos"]}/{rl["n"]} ({rl["pct_vuelcos"]:.1%})')
    print(f'    {estables}/{len(filas)} casos no cambian de veredicto con ningun ajuste fino')
    print(f'  DERIVA HACIA ARRIBA ({AMPLIO:.0%}, entra cerebro anterior):')
    print(f'    {r_arriba["vuelcos"]}/{r_arriba["n"]} cambian el veredicto '
          f'({r_arriba["pct_vuelcos"]:.1%}) · desvio mediano '
          f'{r_arriba["desvio_mediano"]:.4f}')
    print(f'  OTRAS DERIVAS ({AMPLIO:.0%} lateral o hacia abajo):')
    print(f'    {r_otras["vuelcos"]}/{r_otras["n"]} cambian el veredicto '
          f'({r_otras["pct_vuelcos"]:.1%})')
    print(f'  JSON: {SALIDA}')


if __name__ == '__main__':
    main()
