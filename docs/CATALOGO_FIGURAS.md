# Catálogo de figuras

Qué figura del libro sobrevive a las correcciones, cuál hay que rehacer y con qué archivo se reemplaza. Regenerable con `python src/catalogo_figuras.py`.

Los diagramas estructurales no se entregan como imagen sino como código para renderizar, en [DIAGRAMAS_CODIGO.md](DIAGRAMAS_CODIGO.md).

---

## 1. Las 18 figuras del libro

| # | Figura | Sección | Estado | Reemplazo | Hallazgo | Script |
|---|---|---|---|---|---|---|
| 1 | Malformación de Arnold-Chiari | §3.1.1 | ✅ se mantiene | — | C-10 | — |
| 2 | MRI de un cerebro normal (sagital) | §3.1.2 | ✅ se mantiene | — | C-10 | — |
| 3 | Estructura general de una CNN | §3.1.4 | ✅ se mantiene | — | — | — |
| 4 | Marco general en transfer learning | §3.1.5 | ✅ se mantiene | — | — | — |
| 5 | Muestra representativa de las MRI (Nb 01) | §4.2.2 | 🔄 se rehace | `figures/muestra_dataset.png` · 1864 KB | C-01 · M-01 | `src/figuras_libro.py` |
| 6 | Análisis estadístico de intensidades (Nb 01) | §4.2.2 | ❌ se retira | — | C-01 | — |
| 7 | Efecto del preprocesamiento CLAHE | §4.2.4 | 🔄 se rehace | `figures/efecto_clahe.png` · 917 KB | C-06 · C-08 | `src/figuras_libro.py` |
| 8 | EDA de las imágenes procesadas con CLAHE (Nb 02) | §4.2.4 | ❌ se retira | — | C-01 | — |
| 9 | Visualización t-SNE: separabilidad de clases (Nb 02) | §4.2.4 | 🔄 se rehace | `figures/tsne_sensibilidad.png` · 611 KB | M-14 | `src/tsne.py` |
| 10 | Diagrama de casos de uso | §4.4.4 | 📐 pasa a código | `DIAGRAMAS_CODIGO.md §4.1` | M-17 · M-19 | `src/diagramas_codigo.py` |
| 11 | Diagrama de secuencia: flujo de predicción | §4.4.5 | 📐 pasa a código | `DIAGRAMAS_CODIGO.md §4.2` | C-06 · C-07 · M-08 | `src/diagramas_codigo.py` |
| 12 | Diagrama de componentes | §4.4.6 | 📐 pasa a código | `DIAGRAMAS_CODIGO.md §4.3` | C-01 · C-06 · M-01 | `src/diagramas_codigo.py` |
| 13 | Diagrama de actividad: preprocesar imagen | §4.4.7 | 📐 pasa a código | `DIAGRAMAS_CODIGO.md §4.4` | C-06 · C-08 | `src/diagramas_codigo.py` |
| 14 | Distribución de métricas K-Fold (Nb 03) | §5.1 | 🔄 se rehace | `figures/auc_por_fold.png` · 323 KB | C-01 · C-03 · M-05 · M-11 | `src/figuras_libro.py` |
| 15 | Dashboard de evaluación externa DenseNet121 (Nb 04) | §5.2 | ❌ se retira | — | C-01 · M-02 | — |
| 16 | Grad-CAM sobre casos externos (Nb 04) | §5.3 | 🔄 se rehace | `gradcam/gradcam_pediatrico_vs_chiari.png` · 8914 KB | M-10 | `src/gradcam_edad.py` |
| 17 | Captura de la app: caso normal | §5.5.1 | 🔄 se rehace | `capturas_app/pantallazos/Pantallazo_figura_A.png` · 241 KB | M-17 · M-18 | — |
| 18 | Captura de la app: caso Chiari I | §5.5.2 | 🔄 se rehace | `capturas_app/pantallazos/Pantallazo_figura_B.png` · 286 KB | M-17 · M-18 | — |

**4 se mantienen · 7 se rehacen · 3 se retiran · 4 pasan a código.**

### Por qué, figura por figura

- **Figura 5 — Muestra representativa de las MRI (Nb 01).** La muestra salía del inventario con fuga. Ahora sale del conjunto final de 177/117, con la edad y el split de cada imagen.
- **Figura 6 — Análisis estadístico de intensidades (Nb 01).** Comparaba intensidades entre clases sobre el conjunto contaminado. La conclusión que sostenía ya no se sostiene y el análisis no se rehízo: se retira en vez de reciclarse.
- **Figura 7 — Efecto del preprocesamiento CLAHE.** El libro aplicaba CLAHE DESPUÉS del redimensionado. La figura nueva muestra los dos órdenes y su diferencia (media 3,0/255, máx 32/255): no son intercambiables.
- **Figura 8 — EDA de las imágenes procesadas con CLAHE (Nb 02).** Mismo problema que la Figura 6: descriptivo sobre el conjunto con fuga.
- **Figura 9 — Visualización t-SNE: separabilidad de clases (Nb 02).** La versión del libro mostraba UNA proyección sin declarar perplejidad ni semilla, y se leía como evidencia de separabilidad. La nueva barre 4 perplejidades × 2 semillas y muestra que la separación aparente depende de los parámetros.
- **Figura 10 — Diagrama de casos de uso.** Aparece la validación de entrada como sub-caso obligatorio y desaparece la promesa de «privacidad por diseño» que M-19 obliga a retirar.
- **Figura 11 — Diagrama de secuencia: flujo de predicción.** El del libro describe umbral 0,5, normalización a [0,1] y resize antes de CLAHE. Ninguna de las tres cosas es cierta hoy.
- **Figura 12 — Diagrama de componentes.** Los notebooks 03/04 ya no son los componentes de entrenamiento, y hay que representar las tres salvaguardas que abortan el pipeline.
- **Figura 13 — Diagrama de actividad: preprocesar imagen.** El libro afirma que el flujo «no tiene bifurcaciones»; ahora sí las tiene, porque la validación de entrada puede detenerlo.
- **Figura 14 — Distribución de métricas K-Fold (Nb 03).** El K-Fold del libro tenía fuga y parada temprana. La figura nueva pone el AUC de cada pliegue junto al OOF agrupado con su IC 95 %, y marca en rojo la variante con parada temprana para que no se lea como la mejor.
- **Figura 15 — Dashboard de evaluación externa DenseNet121 (Nb 04).** El conjunto «externo» compartía casos con el de entrenamiento y no contenía ni un solo negativo válido. La especificidad 0,9333 que sostenía esta figura no es recuperable.
- **Figura 16 — Grad-CAM sobre casos externos (Nb 04).** Los casos «externos» estaban contaminados. La figura nueva es out-of-fold, sobre el modelo desplegado, y su lectura es la contraria: el mapa NO separa las clases (p=0,911 y p=0,558 entre clases opuestas; la única diferencia detectable, p=0,018, está dentro de la misma clase por edad, y no sobrevive a Bonferroni). Los normales adultos acertados activan el tercio inferior MÁS que todos los grupos Chiari.
- **Figura 17 — Captura de la app: caso normal.** LISTA. Caso normal_22a del test sellado, sin recortar: clasificación no compatible al 1,8 %. La primera toma salió con un falso positivo porque el recuadro incluyó cerebro anterior; se repitió encuadrando la fosa posterior.
- **Figura 18 — Captura de la app: caso Chiari I.** LISTA. Caso chiari_58b del test sellado, sin recortar, con el recuadro sugerido: clasificación compatible al 100 %. Muestra además el recorte dentro de la app (RF02).

---

## 2. Figuras nuevas, sin equivalente en el libro

| Archivo | Figura | Sección propuesta | Hallazgo | Script |
|---|---|---|---|---|
| `DIAGRAMAS_CODIGO.md §1` | Diagrama de flujo CONSORT/STARD | §4.2 (nueva) | C-02 | `src/diagramas_codigo.py` |
| `figures/distribucion_umbral.png` · 139 KB | Distribución de la probabilidad por clase, con los dos puntos de corte | §5.2 (nueva) | M-08 | `src/figura_umbral.py` |
| `figures/roc_oof_umbral.png` · 158 KB | ROC out-of-fold y elección del umbral | §5.1 (nueva) | M-08 | `src/umbral_y_edad.py` |
| `figures/pr_y_calibracion.png` · 192 KB | Curva precisión-exhaustividad y calibración | §5.2 (nueva) | M-09 · M-13 | `src/analisis_finales.py` |
| `figures/chequeo_flip.png` · 1389 KB | El volteo horizontal sobre cortes sagitales | §4.3 (nueva) | M-16 | `src/figuras_chequeo.py` |
| `figures/chequeo_alineacion.png` · 2850 KB | Comprobación de alineación del manifiesto | Apéndice B (nueva) | M-01 | `src/figuras_chequeo.py` |

### Qué aporta cada una

- **Diagrama de flujo CONSORT/STARD.** De 120 casos documentados a 177 imágenes / 117 casos, con cada exclusión contada. Se entrega **como código** (Graphviz para el libro, Mermaid para revisar rápido), no como imagen: el PNG en `figures/consort_stard.png` queda solo como vista previa.
- **Distribución de la probabilidad por clase, con los dos puntos de corte.** Un solo panel, 97 casos. Deja ver las tres cosas que ninguna tabla transmite: los 9 positivos que se perderían al subir el corte a 0,50, los 17 falsos positivos que explican la especificidad de 0,5278, y que el solapamiento entre clases es real. Ancho exacto de 16,5 cm a 300 ppp, con PDF vectorial.
- **ROC out-of-fold y elección del umbral.** Dónde cae el umbral 0,1484 sobre la ROC de desarrollo. Es la figura que justifica que el punto de corte NO se eligió mirando el test.
- **Curva precisión-exhaustividad y calibración.** Con 61 positivos y 36 negativos la ROC es optimista; la curva PR y el diagrama de calibración muestran lo que la ROC esconde.
- **El volteo horizontal sobre cortes sagitales.** Prueba visual de que np.fliplr invierte el eje antero-posterior, no la lateralidad: el modelo veía anatomías imposibles en la mitad de las épocas.
- **Comprobación de alineación del manifiesto.** Cómo se resolvió si el archivo normal_N corresponde a la fila N o a la N−1, usando el aspecto de la imagen (lactante o adulto) en vez de la confianza en el Excel.

---

## 3. Diagramas entregados como código

En `resultados_v2/DIAGRAMAS_CODIGO.md`. Se pegan en <https://mermaid.live> (Graphviz en <https://dreampuf.github.io/GraphvizOnline>).

| Sección del documento | Diagrama | Formato | Hallazgo | Va en |
|---|---|---|---|---|
| §1 | Flujo CONSORT/STARD | Mermaid + Graphviz | C-02 | §4.2 |
| §2 | Pipeline de preprocesamiento unificado | Mermaid | C-06 · C-08 | §4.2.4 |
| §3 | Flujo de la app con validación de entrada | Mermaid | C-07 · M-17 | §4.4 |
| §4.1 | Casos de uso (reemplaza Fig. 10) | Mermaid | M-17 · M-19 | §4.4.4 |
| §4.2 | Secuencia de predicción (reemplaza Fig. 11) | Mermaid | C-06 · M-08 | §4.4.5 |
| §4.3 | Componentes (reemplaza Fig. 12) | Mermaid | C-01 · C-06 | §4.4.6 |
| §4.4 | Actividad de preprocesar() (reemplaza Fig. 13) | Mermaid | C-06 · C-08 | §4.4.7 |

---

## 4. Tareas pendientes de documentación

| Qué | Acción requerida |
|---|---|
| Figuras 17 y 18 — capturas de la app | Hay que levantar la app y capturar la pantalla con la interfaz actual (umbral 0,1484, aviso de cercanía al umbral, validación de entrada, población adulta). `streamlit run app/app.py` |
| Figuras 1 y 2 — atribución | Son ilustraciones de terceros; hay que declarar autor y licencia de cada una (C-10) |
| Lista de Figuras del libro | La numeración cambia al retirar tres figuras; actualizar la numeración y las referencias cruzadas del documento |

---

## 5. Nota sobre las figuras que se retiran

Las Figuras 6, 8 y 15 no se sustituyen por otra versión: describían el conjunto contaminado y el análisis que sostenían no se rehízo. Retirarlas es más honesto que regenerarlas sobre los datos limpios, porque su función en el texto era respaldar afirmaciones (separabilidad por intensidad, validación externa) que ya no se hacen. La Figura 15 en particular sostenía la especificidad de 0,9333 sobre un conjunto sin un solo negativo válido.
