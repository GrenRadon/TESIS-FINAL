# Guía de reproducibilidad del proyecto

**2026-09-06** · Workspace `Correcciones/proyecto_v2/`

Este anexo describe cómo reproducir, desde los datos originales, todas las cifras que aparecen
en el capítulo de resultados. Cada script es ejecutable de forma independiente y escribe sus
salidas en `resultados_*/`.

---

## Entorno

| | |
|---|---|
| Python | 3.10.0 |
| TensorFlow / Keras | 2.21.0 / 3.12.1 |
| scikit-learn | 1.7.2 |
| GPU | **ninguna** — todo se ejecuta en CPU. Verificado: `tf.config.list_physical_devices('GPU')` devuelve lista vacía |
| Procesador | Intel64 Family 6 Model 165 (i5-10300H), 8 núcleos lógicos |
| Latencia de inferencia | p50 34 ms · p95 43 ms (`src/latencia.py`) |
| Semilla global | **42** en todos los scripts |
| Pipeline de preprocesamiento | v2.0.0 (`src/preprocesamiento.py`) |

Dependencias: `tensorflow`, `scikit-learn`, `matplotlib`, `scipy`, `onnxruntime`, `tf2onnx`,
`opencv-python-headless`, `pillow`, `numpy`, `streamlit`.

**Todos los comandos se ejecutan desde `Correcciones/proyecto_v2/`.**

---

## Orden de ejecución

### 1 · Datos — de las imágenes al conjunto particionado

```bash
python src/auditoria.py        # inventario, hashes, detección de duplicados y fuga
python src/particion.py        # manifiesto por caso, partición y VERIFICACIÓN obligatoria
python src/procedencia.py      # manifiesto reproducible con autoría, licencia y fecha
python src/consort.py          # diagrama CONSORT/STARD (C-02)
```

`particion.py` **aborta** si la intersección de casos entre desarrollo y test no es vacía, o si
alguna imagen aparece en ambos lados por hash. Es la salvaguarda de C-01.

El conjunto de prueba está congelado en `resultados_v2/test_sellado.json` y **no se re-sortea**:
si un caso se excluye y estaba en el test, sale y no se sustituye.

### 2 · Entrenamiento

```bash
python src/entrenamiento_v4.py       # 30 entrenamientos: 3 arquitecturas + 3 ablaciones
python src/ablacion_dense.py         # 10 entrenamientos: Dense(16) y Dense(64)   — M-12
python src/ablacion_desbalance.py    # 10 entrenamientos: focal loss y batch balanceado — M-04
python src/reentrenar_sin_flip.py    # 5 entrenamientos: configuración desplegada, GUARDANDO modelos
```

`reentrenar_sin_flip.py` existe porque `entrenamiento_v4.py` solo persiste los modelos de las
variantes `base_*`. Sin los pesos de la configuración desplegada no se puede calcular su Grad-CAM.
Al terminar compara sus predicciones con las de la corrida v4 y reporta la diferencia.

Validación cruzada **k=5 agrupada por caso** (`StratifiedGroupKFold`), con `assert` de no
solapamiento en cada pliegue. Épocas **fijadas a priori, sin EarlyStopping**, para que el
pliegue de validación no intervenga en el entrenamiento (M-05).

Tiempo aproximado en CPU: 3,5 h + 50 min + 55 min + 28 min.

### 3 · Evaluación

```bash
python src/evaluacion_v4.py      # M-11, ablaciones, test sellado
python src/evaluacion_m12.py     # M-12
python src/analisis_finales.py   # PR, calibración, PPV/NPV, desagregado, control de tamaño
python src/confusor_edad.py      # M-10/M-14: estratificación por edad
python src/gradcam_edad.py       # M-10: Grad-CAM out-of-fold cuantificado
python src/tsne.py               # M-14: t-SNE con parámetros declarados
```

Intervalos de confianza: **Wilson** para sensibilidad y especificidad; **bootstrap de 2000
remuestreos agrupados por caso** para el AUC. Se remuestrean casos, nunca imágenes.

### 4 · Modelo de producción

```bash
python src/entrenamiento_final_v4.py --sin-flip
python app/convert_to_onnx.py --modelo models_final_v4/chiari_DenseNet121_final_v4_sin_flip.keras \
    --arquitectura DenseNet121 --umbral 0.1484 \
    --salida models_final_v4/sin_flip/chiari_DenseNet121_final_v4_sin_flip.onnx
python src/validacion_entrada.py   # referencia de detección fuera de distribución
python src/curva_operacion.py      # puntos de operación que muestra la app (M-08/M-22)
python src/geometria_recorte.py    # geometría del recuadro sugerido en la app (RF15)
python src/sensibilidad_recorte.py # cuánto cambia el resultado con el encuadre (RF16)
python src/latencia.py             # medición de latencia y su variabilidad
python src/pruebas_app.py          # 16 pruebas funcionales
```

`convert_to_onnx.py` **aborta** si la diferencia entre Keras y ONNX supera 1e-4, y escribe el
`config_modelo.json` **junto al `.onnx`**, para que un modelo no pueda quedar desincronizado
de su umbral.

### 5 · Documentos

```bash
python src/figuras_chequeo.py       # chequeo_flip y chequeo_alineacion
python src/figuras_libro.py         # reemplazos de las Figuras 5, 7 y 14 del libro
python src/figura_umbral.py        # distribución por clase con los dos cortes (M-08)
python src/diagramas_codigo.py      # diagramas Mermaid/Graphviz (no PNG)
python src/catalogo_figuras.py      # catálogo de las 18 figuras del libro
python src/entregar_figuras.py      # carpeta única con las figuras renumeradas
python src/generar_tablas.py       # muestra las tablas calculadas desde CSV y JSON
```

Las tablas de resultados se generan mediante `src/generar_tablas.py`
a partir de los archivos CSV y JSON del proyecto, para mantener
la coherencia entre los valores calculados y los reportados.

---

## Verificaciones automáticas

Cada una detiene el proceso si falla:

| Verificación | Dónde | Qué garantiza |
|---|---|---|
| Intersección de casos dev∩test vacía | `particion.py` | C-01 |
| Intersección de hashes sha256 = 0 | `particion.py` | Detecta duplicados invisibles al nombre |
| Grupos disjuntos por pliegue | `entrenamiento_v4.py` | C-03 |
| Todas las imágenes tienen predicción OOF | `entrenamiento_v4.py` | Ningún caso sin evaluar |
| Keras ≡ ONNX (< 1e-4) | `convert_to_onnx.py` | C-06 |
| Un solo `.onnx` en `app/models/` | `app/app.py` | Modelo y umbral sincronizados |
| Grad-CAM reproduce las OOF (< 1e-3) | `gradcam_edad.py` | El mapa describe al modelo desplegado, no a uno reconstruido mal |

---

## Fuentes únicas

Para evitar que el mismo dato exista en dos sitios y diverja:

| Dato | Fuente única | Lo consumen |
|---|---|---|
| Preprocesamiento | `src/preprocesamiento.py` | entrenamiento, exportación ONNX, app |
| Salida de figuras (300 dpi + PDF) | `src/figuras.py` | todos los scripts que dibujan |
| Composición del dataset | `resultados_v2/dataset_nivel_imagen.csv` | todas las tablas |
| Umbral del modelo | `app/models/config_modelo.json` | app y análisis |
| Conjunto de prueba | `resultados_v2/test_sellado.json` | particion.py |
| Estado de cada figura | `LIBRO`/`NUEVAS` en `src/catalogo_figuras.py` | CATALOGO_FIGURAS.md |

---

## Qué NO es reproducible, y por qué

- **Las imágenes no se redistribuyen.** Proceden de Radiopaedia bajo CC BY-NC-SA 3.0. `resultados_v2/procedencia_imagenes.csv` contiene, para las 177,
  el autor, el rID, la URL del caso, la fecha de acceso y el **sha256**, de modo que un tercero
  puede recuperar las mismas imágenes y verificar que son idénticas.
- **La licencia está verificada**: CC BY-NC-SA 3.0, única para todo el sitio, con las
  modificaciones de los Términos de Uso de Radiopaedia. No se redistribuyen las imágenes, pero
  el ShareAlike condicionaría cualquier publicación futura del dataset derivado.
- **En la misma máquina el entrenamiento SÍ es determinista.** Verificado el 2026-09-06:
  reentrenar los 5 pliegues de la configuración desplegada reprodujo las 148 predicciones
  out-of-fold con diferencia **0,000000** y los cinco AUC por pliegue idénticos.
  **Entre máquinas distintas no está comprobado** y cabría esperar diferencias en los últimos
  decimales.
