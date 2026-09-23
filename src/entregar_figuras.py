"""Reune las figuras del libro en una carpeta unica, renombradas con SU numero en el libro.

El catalogo usa los nombres de archivo del pipeline; el libro usa numeros que ya no coinciden
con los originales (se retiraron tres figuras y se anadieron seis). Copiar y renombrar aqui
evita que alguien tenga que hacer la correspondencia a mano cada vez, que es justo donde se
cuelan los errores.

Escribe resultados_v2/entrega_figuras/ con PNG a 300 dpi, el PDF vectorial cuando existe, y
un INVENTARIO.txt con la procedencia y la resolucion real de cada archivo.

Uso:  python src/entregar_figuras.py
"""
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image

V2 = Path('resultados_v2')
DESTINO = V2 / 'entrega_figuras'

# (numero en el libro, seccion, origen, descripcion breve)
MAPA = [
    (6, '§4.2.2', V2 / 'figures/muestra_dataset.png',
     'Muestra del conjunto final: 4 Chiari y 4 Normal, con edad y particion'),
    (8, '§4.2.4', V2 / 'figures/efecto_clahe.png',
     'Orden CLAHE/resize: los dos ordenes y su diferencia absoluta'),
    (9, '§4.2.4', V2 / 'figures/tsne_sensibilidad.png',
     't-SNE, 4 perplejidades x 2 semillas'),
    (10, '§4.3.4', V2 / 'figures/chequeo_flip.png',
     'El volteo horizontal sobre cortes sagitales invierte el eje antero-posterior'),
    (16, '§5.1', V2 / 'figures/auc_por_fold.png',
     'AUC por pliegue frente al AUC out-of-fold agrupado, 6 variantes'),
    (17, '§5.2.3', V2 / 'figures/roc_oof_umbral.png',
     'ROC out-of-fold y eleccion del umbral'),
    (18, '§5.2.5', V2 / 'figures/pr_y_calibracion.png',
     'Curva precision-exhaustividad y diagrama de calibracion'),
    (19, '§5.3', V2 / 'gradcam/gradcam_pediatrico_vs_chiari.png',
     'Grad-CAM out-of-fold, 4 grupos'),
    (20, '§5.5.2', V2 / 'capturas_app/pantallazos/Pantallazo_figura_A.png',
     'Captura de la app: resultado no compatible (normal_22a, 1,8 %)'),
    (21, '§5.5.2', V2 / 'capturas_app/pantallazos/Pantallazo_figura_B.png',
     'Captura de la app: resultado compatible (chiari_58b, 100 %)'),
]
# Las que se entregan como codigo, no como imagen
CODIGO = [
    (5, '§4.2', 'DIAGRAMAS_CODIGO.txt §1', 'CONSORT/STARD — Graphviz recomendado'),
    (7, '§4.2.4', 'DIAGRAMAS_CODIGO.txt §2', 'Pipeline de preprocesamiento — Mermaid'),
    (11, '§4.4.3', 'DIAGRAMAS_CODIGO.txt §3', 'Flujo de la app con validacion — Mermaid'),
    (12, '§4.4.4', 'DIAGRAMAS_CODIGO.txt §4.1', 'Casos de uso — Mermaid'),
    (13, '§4.4.5', 'DIAGRAMAS_CODIGO.txt §4.2', 'Secuencia de prediccion — Mermaid'),
    (14, '§4.4.6', 'DIAGRAMAS_CODIGO.txt §4.3', 'Componentes — Mermaid'),
    (15, '§4.4.7', 'DIAGRAMAS_CODIGO.txt §4.4', 'Actividad de preprocesar() — Mermaid'),
]
# Sin ubicacion asignada en el libro
SIN_UBICAR = [
    (V2 / 'figures/chequeo_alineacion.png', 'Apendice B (propuesta)',
     'Comprobacion de alineacion del manifiesto: normal_8 a normal_26'),
]


def info(p):
    if not p.exists():
        return None
    im = Image.open(p)
    dpi = im.info.get('dpi', (None, None))[0]
    return dict(px=f'{im.size[0]}x{im.size[1]}', dpi=round(dpi) if dpi else None,
                kb=p.stat().st_size // 1024,
                fecha=datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M'))


def main():
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    DESTINO.mkdir(parents=True)

    L = ['# Figuras del libro — entrega', '',
         f'Generado el {datetime.now():%Y-%m-%d %H:%M} con `python src/entregar_figuras.py`.', '',
         'Los archivos van renombrados con **su numero en el libro corregido**, no con el '
         'nombre del pipeline. El nombre de origen queda en la tabla para poder rastrearlos.', '',
         'Cada figura de matplotlib se entrega en **PNG a 300 dpi** y en **PDF vectorial**. '
         'Para el documento impreso conviene el PDF: el texto sigue siendo texto y no se '
         'degrada al escalar.', '',
         '## Figuras entregadas como imagen', '',
         '| Fig. | Sección | Archivo entregado | Origen | Resolución | dpi | Generada |',
         '|---|---|---|---|---|---|---|']
    faltan, copiados = [], 0
    for n, sec, origen, desc in MAPA:
        i = info(origen)
        if i is None:
            faltan.append(str(origen))
            L.append(f'| {n} | {sec} | — | `{origen.name}` | **FALTA** | | |')
            continue
        nuevo = DESTINO / f'figura_{n:02d}{origen.suffix}'
        shutil.copy2(origen, nuevo)
        copiados += 1
        extra = ''
        pdf = origen.with_suffix('.pdf')
        if pdf.exists():
            shutil.copy2(pdf, DESTINO / f'figura_{n:02d}.pdf')
            extra = ' + `.pdf`'
        aviso = '' if (i['dpi'] or 0) >= 300 else ' ⚠'
        L.append(f'| {n} | {sec} | `figura_{n:02d}{origen.suffix}`{extra} | '
                 f'`{origen.name}` | {i["px"]} | {i["dpi"] or "—"}{aviso} | {i["fecha"]} |')

    L += ['', '⚠ = por debajo de 300 dpi. Las capturas de pantalla no pueden subirse de '
              'resolución sin volver a tomarlas: su tamaño lo fija la pantalla.', '',
          '## Figuras entregadas como código', '',
          'No son imágenes: se renderizan desde `resultados_v2/DIAGRAMAS_CODIGO.txt`.', '',
          '| Fig. | Sección | Dónde | Qué es |', '|---|---|---|---|']
    for n, sec, donde, desc in CODIGO:
        L.append(f'| {n} | {sec} | {donde} | {desc} |')

    L += ['', '## Sin ubicación asignada', '', '| Archivo | Propuesta | Qué muestra |',
          '|---|---|---|']
    for p, prop, desc in SIN_UBICAR:
        i = info(p)
        if i:
            nuevo = DESTINO / f'sin_ubicar_{p.name}'
            shutil.copy2(p, nuevo)
            pdf = p.with_suffix('.pdf')
            if pdf.exists():
                shutil.copy2(pdf, DESTINO / f'sin_ubicar_{p.stem}.pdf')
            copiados += 1
        L.append(f'| `{p.name}` | {prop} | {desc} |')

    L += ['', '## Descripción de cada figura', '', '| Fig. | Qué muestra |', '|---|---|']
    for n, _s, _o, desc in MAPA:
        L.append(f'| {n} | {desc} |')

    (DESTINO / 'INVENTARIO.txt').write_text('\n'.join(L), encoding='utf-8')
    print(f'  carpeta: {DESTINO}')
    print(f'  archivos copiados: {copiados}')
    if faltan:
        print('  FALTAN: ' + ', '.join(faltan))
    bajos = [n for n, _s, o, _d in MAPA
             if (i := info(o)) and (i['dpi'] or 0) < 300]
    if bajos:
        print(f'  por debajo de 300 dpi: figuras {bajos}')


if __name__ == '__main__':
    main()
