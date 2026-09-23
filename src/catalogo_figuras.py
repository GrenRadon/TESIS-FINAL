"""Catalogo de figuras: que figura del libro sobrevive, cual se rehace y con que archivo.

Cruza la lista de figuras del libro con lo que hay realmente en resultados_v2/figures/ y
en DIAGRAMAS_CODIGO.txt, y marca lo que falta. Regenerable:  python src/catalogo_figuras.py
"""
from pathlib import Path

OUT = Path('resultados_v2')
FIG = OUT / 'figures'
GC = OUT.parent / 'resultados_v2' / 'gradcam'
SALIDA = OUT / 'CATALOGO_FIGURAS.txt'

# (n, titulo en el libro, seccion, veredicto, reemplazo, hallazgo, script, motivo)
# veredicto: MANTENER | REHACER | RETIRAR | CODIGO
LIBRO = [
    (1, 'Malformación de Arnold-Chiari', '§3.1.1', 'MANTENER', '', 'C-10',
     '—', 'Ilustración conceptual. Solo hay que verificar atribución y licencia de la fuente.'),
    (2, 'MRI de un cerebro normal (sagital)', '§3.1.2', 'MANTENER', '', 'C-10',
     '—', 'Igual que la anterior: revisar atribución.'),
    (3, 'Estructura general de una CNN', '§3.1.4', 'MANTENER', '', '—',
     '—', 'Marco conceptual, no depende de los datos.'),
    (4, 'Marco general en transfer learning', '§3.1.5', 'MANTENER', '', '—',
     '—', 'Marco conceptual, no depende de los datos.'),
    (5, 'Muestra representativa de las MRI (Nb 01)', '§4.2.2', 'REHACER',
     'figures/muestra_dataset.png', 'C-01 · M-01', 'src/figuras_libro.py',
     'La muestra salía del inventario con fuga. Ahora sale del conjunto final de 177/117, '
     'con la edad y el split de cada imagen.'),
    (6, 'Análisis estadístico de intensidades (Nb 01)', '§4.2.2', 'RETIRAR', '', 'C-01',
     '—', 'Comparaba intensidades entre clases sobre el conjunto contaminado. La conclusión '
     'que sostenía ya no se sostiene y el análisis no se rehízo: se retira en vez de '
     'reciclarse.'),
    (7, 'Efecto del preprocesamiento CLAHE', '§4.2.4', 'REHACER',
     'figures/efecto_clahe.png', 'C-06 · C-08', 'src/figuras_libro.py',
     'El libro aplicaba CLAHE DESPUÉS del redimensionado. La figura nueva muestra los dos '
     'órdenes y su diferencia (media 3,0/255, máx 32/255): no son intercambiables.'),
    (8, 'EDA de las imágenes procesadas con CLAHE (Nb 02)', '§4.2.4', 'RETIRAR', '', 'C-01',
     '—', 'Mismo problema que la Figura 6: descriptivo sobre el conjunto con fuga.'),
    (9, 'Visualización t-SNE: separabilidad de clases (Nb 02)', '§4.2.4', 'REHACER',
     'figures/tsne_sensibilidad.png', 'M-14', 'src/tsne.py',
     'La versión del libro mostraba UNA proyección sin declarar perplejidad ni semilla, y se '
     'leía como evidencia de separabilidad. La nueva barre 4 perplejidades × 2 semillas y '
     'muestra que la separación aparente depende de los parámetros.'),
    (10, 'Diagrama de casos de uso', '§4.4.4', 'CODIGO', 'DIAGRAMAS_CODIGO.txt §4.1',
     'M-17 · M-19', 'src/diagramas_codigo.py',
     'Aparece la validación de entrada como sub-caso obligatorio y desaparece la promesa de '
     '«privacidad por diseño» que M-19 obliga a retirar.'),
    (11, 'Diagrama de secuencia: flujo de predicción', '§4.4.5', 'CODIGO',
     'DIAGRAMAS_CODIGO.txt §4.2', 'C-06 · C-07 · M-08', 'src/diagramas_codigo.py',
     'El del libro describe umbral 0,5, normalización a [0,1] y resize antes de CLAHE. '
     'Ninguna de las tres cosas es cierta hoy.'),
    (12, 'Diagrama de componentes', '§4.4.6', 'CODIGO', 'DIAGRAMAS_CODIGO.txt §4.3',
     'C-01 · C-06 · M-01', 'src/diagramas_codigo.py',
     'Los notebooks 03/04 ya no son los componentes de entrenamiento, y hay que representar '
     'las tres salvaguardas que abortan el pipeline.'),
    (13, 'Diagrama de actividad: preprocesar imagen', '§4.4.7', 'CODIGO',
     'DIAGRAMAS_CODIGO.txt §4.4', 'C-06 · C-08', 'src/diagramas_codigo.py',
     'El libro afirma que el flujo «no tiene bifurcaciones»; ahora sí las tiene, porque la '
     'validación de entrada puede detenerlo.'),
    (14, 'Distribución de métricas K-Fold (Nb 03)', '§5.1', 'REHACER',
     'figures/auc_por_fold.png', 'C-01 · C-03 · M-05 · M-11', 'src/figuras_libro.py',
     'El K-Fold del libro tenía fuga y parada temprana. La figura nueva pone el AUC de cada '
     'pliegue junto al OOF agrupado con su IC 95 %, y marca en rojo la variante con parada '
     'temprana para que no se lea como la mejor.'),
    (15, 'Dashboard de evaluación externa DenseNet121 (Nb 04)', '§5.2', 'RETIRAR', '',
     'C-01 · M-02', '—',
     'El conjunto «externo» compartía casos con el de entrenamiento y no contenía ni un solo '
     'negativo válido. La especificidad 0,9333 que sostenía esta figura no es recuperable.'),
    (16, 'Grad-CAM sobre casos externos (Nb 04)', '§5.3', 'REHACER',
     'gradcam/gradcam_pediatrico_vs_chiari.png', 'M-10', 'src/gradcam_edad.py',
     'Los casos «externos» estaban contaminados. La figura nueva es out-of-fold, sobre el '
     'modelo desplegado, y su lectura es la contraria: el mapa NO separa las clases '
     '(p=0,911 y p=0,558 entre clases opuestas; la única diferencia detectable, p=0,018, '
     'está dentro de la misma clase por edad, y no sobrevive a Bonferroni). Los normales '
     'adultos acertados activan el tercio inferior MÁS que todos los grupos Chiari.'),
    (17, 'Captura de la app: caso normal', '§5.5.1', 'REHACER',
     'capturas_app/pantallazos/Pantallazo_figura_A.png', 'M-17 · M-18', '—',
     'LISTA. Caso normal_22a del test sellado, sin recortar: clasificación no compatible al '
     '1,8 %. La primera toma salió con un falso positivo porque el recuadro incluyó cerebro '
     'anterior; se repitió encuadrando la fosa posterior.'),
    (18, 'Captura de la app: caso Chiari I', '§5.5.2', 'REHACER',
     'capturas_app/pantallazos/Pantallazo_figura_B.png', 'M-17 · M-18', '—',
     'LISTA. Caso chiari_58b del test sellado, sin recortar, con el recuadro sugerido: '
     'clasificación compatible al 100 %. Muestra además el recorte dentro de la app (RF02).'),
]

# figuras sin equivalente en el libro: (archivo, titulo, seccion propuesta, hallazgo, script, que muestra)
NUEVAS = [
    ('DIAGRAMAS_CODIGO.txt §1', 'Diagrama de flujo CONSORT/STARD', '§4.2 (nueva)', 'C-02',
     'src/diagramas_codigo.py',
     'De 120 casos documentados a 177 imágenes / 117 casos, con cada exclusión contada. '
     'Se entrega **como código** (Graphviz para el libro, Mermaid para revisar rápido), no '
     'como imagen: el PNG en `figures/consort_stard.png` queda solo como vista previa.'),
    ('figures/distribucion_umbral.png', 'Distribución de la probabilidad por clase, con los '
     'dos puntos de corte', '§5.2 (nueva)', 'M-08',
     'src/figura_umbral.py',
     'Un solo panel, 97 casos. Deja ver las tres cosas que ninguna tabla transmite: los 9 '
     'positivos que se perderían al subir el corte a 0,50, los 17 falsos positivos que '
     'explican la especificidad de 0,5278, y que el solapamiento entre clases es real. '
     'Ancho exacto de 16,5 cm a 300 ppp, con PDF vectorial.'),
    ('figures/roc_oof_umbral.png', 'ROC out-of-fold y elección del umbral', '§5.1 (nueva)',
     'M-08', 'src/umbral_y_edad.py',
     'Dónde cae el umbral 0,1484 sobre la ROC de desarrollo. Es la figura que justifica que '
     'el punto de corte NO se eligió mirando el test.'),
    ('figures/pr_y_calibracion.png', 'Curva precisión-exhaustividad y calibración',
     '§5.2 (nueva)', 'M-09 · M-13', 'src/analisis_finales.py',
     'Con 61 positivos y 36 negativos la ROC es optimista; la curva PR y el diagrama de '
     'calibración muestran lo que la ROC esconde.'),
    ('figures/chequeo_flip.png', 'El volteo horizontal sobre cortes sagitales', '§4.3 (nueva)',
     'M-16', 'src/figuras_chequeo.py',
     'Prueba visual de que np.fliplr invierte el eje antero-posterior, no la lateralidad: el '
     'modelo veía anatomías imposibles en la mitad de las épocas.'),
    ('figures/chequeo_alineacion.png', 'Comprobación de alineación del manifiesto',
     'Apéndice B (nueva)', 'M-01', 'src/figuras_chequeo.py',
     'Cómo se resolvió si el archivo normal_N corresponde a la fila N o a la N−1, usando el '
     'aspecto de la imagen (lactante o adulto) en vez de la confianza en el Excel.'),
]

CODIGO = [
    ('§1', 'Flujo CONSORT/STARD', 'Mermaid + Graphviz', 'C-02', '§4.2'),
    ('§2', 'Pipeline de preprocesamiento unificado', 'Mermaid', 'C-06 · C-08', '§4.2.4'),
    ('§3', 'Flujo de la app con validación de entrada', 'Mermaid', 'C-07 · M-17', '§4.4'),
    ('§4.1', 'Casos de uso (reemplaza Fig. 10)', 'Mermaid', 'M-17 · M-19', '§4.4.4'),
    ('§4.2', 'Secuencia de predicción (reemplaza Fig. 11)', 'Mermaid', 'C-06 · M-08', '§4.4.5'),
    ('§4.3', 'Componentes (reemplaza Fig. 12)', 'Mermaid', 'C-01 · C-06', '§4.4.6'),
    ('§4.4', 'Actividad de preprocesar() (reemplaza Fig. 13)', 'Mermaid', 'C-06 · C-08',
     '§4.4.7'),
]

EMOJI = {'MANTENER': '✅ se mantiene', 'REHACER': '🔄 se rehace', 'RETIRAR': '❌ se retira',
         'CODIGO': '📐 pasa a código'}


def existe(rel):
    if not rel or rel.endswith('.txt') or '§' in rel:
        return ''
    p = OUT / rel
    return f'{p.stat().st_size // 1024} KB' if p.exists() else '**FALTA**'


def main():
    L = ['# Catálogo de figuras', '',
         'Qué figura del libro sobrevive a las correcciones, cuál hay que rehacer y con qué '
         'archivo se reemplaza. Regenerable con `python src/catalogo_figuras.py`.', '',
         'Los diagramas estructurales no se entregan como imagen sino como código para '
         'renderizar, en [DIAGRAMAS_CODIGO.txt](DIAGRAMAS_CODIGO.txt).', '',
         '---', '', '## 1. Las 18 figuras del libro', '',
         '| # | Figura | Sección | Estado | Reemplazo | Hallazgo | Script |',
         '|---|---|---|---|---|---|---|']
    for n, tit, sec, ver, rep, hal, scr, _ in LIBRO:
        tam = existe(rep)
        r = f'`{rep}`' + (f' · {tam}' if tam else '') if rep else '—'
        L.append(f'| {n} | {tit} | {sec} | {EMOJI[ver]} | {r} | {hal} | '
                 f'{"`" + scr + "`" if scr != "—" else "—"} |')

    cnt = {k: sum(1 for x in LIBRO if x[3] == k) for k in EMOJI}
    L += ['', f'**{cnt["MANTENER"]} se mantienen · {cnt["REHACER"]} se rehacen · '
              f'{cnt["RETIRAR"]} se retiran · {cnt["CODIGO"]} pasan a código.**', '',
          '### Por qué, figura por figura', '']
    for n, tit, _, ver, _, _, _, mot in LIBRO:
        if ver != 'MANTENER':
            L.append(f'- **Figura {n} — {tit}.** {mot}')
    L += ['', '---', '', '## 2. Figuras nuevas, sin equivalente en el libro', '',
          '| Archivo | Figura | Sección propuesta | Hallazgo | Script |',
          '|---|---|---|---|---|']
    for arch, tit, sec, hal, scr, _ in NUEVAS:
        tam = existe(arch)
        L.append(f'| `{arch}`{" · " + tam if tam else ""} | {tit} | {sec} | {hal} | `{scr}` |')
    L += ['', '### Qué aporta cada una', '']
    for _, tit, _, _, _, q in NUEVAS:
        L.append(f'- **{tit}.** {q}')

    L += ['', '---', '', '## 3. Diagramas entregados como código', '',
          'En `resultados_v2/DIAGRAMAS_CODIGO.txt`. Se pegan en <https://mermaid.live> '
          '(Graphviz se renderiza con el comando `dot`).', '',
          '| Sección del documento | Diagrama | Formato | Hallazgo | Va en |',
          '|---|---|---|---|---|']
    for s, d, f_, h, v in CODIGO:
        L.append(f'| {s} | {d} | {f_} | {h} | {v} |')

    L += ['', '---', '', '## 4. Tareas pendientes de documentación', '',
          '| Qué | Acción requerida |', '|---|---|',
          '| Figuras 17 y 18 — capturas de la app | Hay que levantar la app y capturar la '
          'pantalla con la interfaz actual (umbral 0,1484, aviso de cercanía al umbral, '
          'validación de entrada, población adulta). `streamlit run app/app.py` |',
          '| Figuras 1 y 2 — atribución | Son ilustraciones de terceros; hay que declarar '
          'autor y licencia de cada una (C-10) |',
          '| Lista de Figuras del libro | La numeración cambia al retirar tres figuras; actualizar '
          'la numeración y las referencias cruzadas del documento |', '',
          '---', '',
          '## 5. Nota sobre las figuras que se retiran', '',
          'Las Figuras 6, 8 y 15 no se sustituyen por otra versión: describían el conjunto '
          'contaminado y el análisis que sostenían no se rehízo. Retirarlas es más honesto que '
          'regenerarlas sobre los datos limpios, porque su función en el texto era respaldar '
          'afirmaciones (separabilidad por intensidad, validación externa) que ya no se hacen. '
          'La Figura 15 en particular sostenía la especificidad de 0,9333 sobre un conjunto '
          'sin un solo negativo válido.', '']
    SALIDA.write_text('\n'.join(L), encoding='utf-8')
    falta = [x for x in (existe(r[4]) for r in LIBRO) if x == '**FALTA**']
    falta += [x for x in (existe(r[0]) for r in NUEVAS) if x == '**FALTA**']
    print(f'  escrito: {SALIDA}')
    print(f'  libro: {cnt["MANTENER"]} mantener · {cnt["REHACER"]} rehacer · '
          f'{cnt["RETIRAR"]} retirar · {cnt["CODIGO"]} a código')
    print(f'  figuras nuevas: {len(NUEVAS)}   diagramas como codigo: {len(CODIGO)}')
    print(f'  archivos referenciados que faltan: {len(falta)}')


if __name__ == '__main__':
    main()
