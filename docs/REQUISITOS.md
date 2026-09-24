# Requisitos de la herramienta

**Herramienta de apoyo para la identificación de imágenes compatibles con malformación de Chiari I**

Versión 2.4 · 2026-09-06 · Reemplaza la Tabla 10 del libro (§4.4.1)

El nombre evita deliberadamente «diagnóstico», «detección temprana» y «detección de la
malformación». El objeto sobre el que opera la herramienta es **una imagen**, no un paciente ni
una enfermedad (C-09). Cada requisito lleva el hallazgo del que nace, cuando nace de uno.

---

## Requisitos funcionales

| ID | Requisito | Criterio de aceptación | Origen | Estado |
|---|---|---|---|---|
| **RF01** | Cargar una imagen en formato JPG, JPEG o PNG desde el dispositivo del usuario | La imagen se muestra en pantalla tras cargarse; un archivo ilegible produce un mensaje de error explícito y no detiene la aplicación | — | Implementado |
| **RF02** | Permitir recortar la imagen a la región de fosa posterior dentro de la propia herramienta | El usuario delimita una región rectangular con vista previa en vivo; el análisis se ejecuta sobre la región recortada, no sobre la imagen completa | C-07 | Implementado v2.0 |
| **RF16** | Declarar la sensibilidad del resultado al encuadre | La interfaz declara lo medido: con el recuadro sobre la fosa, un ajuste de hasta el 3 % cambia el veredicto en el 7.6% de los casos con resultado claro y en el 81.2% de los que ya caen cerca del punto de corte | C-07, M-08 | Implementado v2.3 |
| **RF15** | Proponer un recuadro inicial y evitar que los controles se interfieran | La posición y el tamaño iniciales se derivan de la geometría medida sobre los 175 recortes manuales (error mediano del centro: 8,4 % del lado de la imagen). Los cuatro controles tienen rango fijo 0–100 %, de modo que mover uno no reajusta a los demás | Comité | Implementado v2.2 |
| **RF03** | Validar la entrada antes de analizarla | Se comprueban lado mínimo (64 px), proporción máxima (3:1), desviación típica mínima (8) y número mínimo de niveles de gris (16). Si falla, se rechaza indicando el motivo concreto y no se emite predicción | M-18 | Implementado |
| **RF04** | Advertir cuando la entrada se aparta de la distribución de entrenamiento | Distancia de Mahalanobis sobre seis descriptores; aviso de «atípica» en el percentil 95 y de «fuera de distribución» en el 99,5 | M-18 | Implementado |
| **RF05** | Aplicar el mismo preprocesamiento del entrenamiento | La herramienta importa `src/preprocesamiento.py`; no reimplementa ninguna etapa. La versión del pipeline se muestra en pantalla | C-06, C-08 | Implementado |
| **RF06** | Ejecutar la inferencia con el modelo en producción | Un único `.onnx` en `app/models/`; si hay cero o más de uno, la aplicación se detiene con error visible en vez de elegir por orden alfabético | C-06 | Implementado |
| **RF07** | Presentar la probabilidad estimada y la decisión respecto al umbral | Se muestran la probabilidad, la barra proporcional, el umbral vigente y la clasificación como compatible / no compatible | M-08 | Implementado |
| **RF08** | Advertir cuando el resultado queda cerca del umbral | Si la distancia al umbral es menor que 0,10, se indica baja confianza | M-08 | Implementado |
| **RF09** | Declarar la población objetivo y el alcance no clínico de forma permanente | Aviso siempre visible, más una sección de documentación accesible sin abandonar la pantalla de análisis | C-09, M-25 | Implementado |
| **RF10** | Ofrecer un manual de uso y una guía de interpretación del resultado | Secciones navegables por pestañas, con el procedimiento de recorte y la lectura correcta de la probabilidad | Comité | Implementado v2.0 |
| **RF11** | Identificar el modelo activo y su procedencia | Se muestran nombre del artefacto, arquitectura, umbral y versión del pipeline | M-01 | Implementado |
| **RF12** | Informar la calibración del modelo junto al resultado | La guía de interpretación declara la sobreconfianza medida: en el tramo alto el modelo predice 0,954 y la proporción real es 0,884 | C-05 | Implementado v2.0 |
| **RF13** | Comunicar el rendimiento en frecuencias naturales y por escenario de uso | El usuario elige la prevalencia del grupo examinado y ve el resultado expresado en personas, no en valores predictivos. El escenario de cribado poblacional se marca como no previsto | M-03 | Implementado v2.0 |
| **RF14** | Exponer el punto de operación como una decisión, no como un límite del modelo | La sección técnica muestra el mismo modelo en siete puntos de corte, con sensibilidad, especificidad, precisión, F1 y VPP por prevalencia, desde `app/models/curva_operacion.json` | M-08, M-22 | Implementado v2.0 |

---

## Requisitos no funcionales

| ID | Requisito | Criterio de aceptación | Origen | Estado |
|---|---|---|---|---|
| **RNF01** | Las imágenes se procesan en memoria y no se escriben en disco ni se transmiten a terceros | El código no contiene escrituras a disco de la imagen ni llamadas de red. **La afirmación cubre el código de la aplicación, no el entorno de despliegue** | M-19 | Implementado con límites declarados |
| **RNF02** | Latencia de inferencia adecuada para uso interactivo | p50 ≤ 100 ms y p95 ≤ 200 ms en CPU. Medido sobre 5 repeticiones de 100 inferencias: **p50 34 ms, p95 43 ms** (rango entre repeticiones 38–48 ms en el p95). Solo inferencia del grafo ONNX | RNF original | Cumplido |
| **RNF03** | Acceso público sin registro ni autenticación | La herramienta se usa sin credenciales | RNF original | Cumplido |
| **RNF04** | Independencia de TensorFlow en producción | El entorno de ejecución solo requiere `onnxruntime`, `pillow`, `opencv`, `numpy` y `streamlit` | — | Cumplido |
| **RNF05** | Trazabilidad entre modelo y umbral | El `config_modelo.json` vive junto al `.onnx`; la exportación aborta si Keras y ONNX difieren en más de 1e-4 | C-06 | Cumplido |
| **RNF06** | Reproducibilidad del entorno | Todas las dependencias se instalan desde `requirements.txt` con versiones fijadas. **No se admiten componentes de terceros no reproducibles**: el recorte se implementa con controles nativos de Streamlit | — | Cumplido v2.0 |
| **RNF07** | Presentación profesional y sobria | Sin emojis ni iconografía decorativa. Paleta de dos colores funcionales, tipografía del sistema, jerarquía visual por espaciado y no por adorno | Comité | Implementado v2.0 |
| **RNF08** | Accesibilidad de la información crítica | Las advertencias de alcance, población y uso no clínico son alcanzables en un clic desde la pantalla de análisis y no dependen de que el usuario haga scroll | Comité | Implementado v2.0 |
| **RNF09** | Comportamiento predecible ante el error | Ningún fallo de entrada produce una traza de Python en pantalla; todos los caminos de error terminan en un mensaje en lenguaje natural | — | Implementado v2.0 |

---

## Lo que la herramienta NO hace, y consta como requisito ausente

Se listan explícitamente porque su ausencia es una limitación declarada, no un olvido.

| Ausencia | Por qué | Hallazgo |
|---|---|---|
| No verifica que la imagen sea una resonancia magnética | Requiere un clasificador de modalidad, no desarrollado | C-07 |
| No verifica que el plano sea sagital medio | Requiere un clasificador de plano, no desarrollado | C-07 |
| No localiza automáticamente la fosa posterior | El recorte lo delimita el usuario (RF02); la sugerencia de RF15 es una regularidad geométrica, no una detección. **Medido**: con el recuadro sobre la región, el resultado es estable ante ajustes finos (7.6% de cambios de veredicto en los casos con resultado claro), pero un encuadre que se va de la fosa produce un resultado incorrecto que la herramienta solo detecta si la región resulta estadísticamente atípica | C-07 |
| No mide el descenso amigdalar en milímetros | El modelo es un clasificador binario, no un sistema de medición | C-09 |
| No emite diagnóstico ni indicación terapéutica | Un hallazgo compatible no equivale a enfermedad sintomática | C-09 |
| No está validada en población pediátrica | Restricción por prudencia: con 8 casos pediátricos negativos el análisis solo detectaría efectos muy grandes | M-10 |
| No ha sido validada de forma externa ni prospectiva | Requiere datos de otra institución | M-02 |
| No ha sido sometida a evaluación de ciberseguridad ni de impacto en protección de datos | Fuera del alcance del proyecto de grado | M-19 |

---

## Trazabilidad con el capítulo 5

| Requisito | Cómo se verifica | Evidencia |
|---|---|---|
| RF03, RF04 | 16 pruebas funcionales automatizadas | `src/pruebas_app.py`, `resultados_v2/pruebas_app.json` |
| RF05 | Igualdad del preprocesamiento en las tres rutas | `src/preprocesamiento.py`, error ≤ 7,15e-07 |
| RF06, RNF05 | Equivalencia Keras ≡ ONNX | `app/convert_to_onnx.py`, diferencia < 1e-4 |
| RF07, RF08 | Umbral elegido solo con predicción out-of-fold | `src/umbral_y_edad.py`, `figures/roc_oof_umbral.png` |
| RF12 | Curva de calibración por tramos | `resultados_v4/analisis_finales.json` |
| RF13 | Valores predictivos derivados de sensibilidad y especificidad | `resultados_v4/analisis_finales.json`, sección `ppv_npv` |
| RF14 | Barrido de puntos de corte sobre la predicción out-of-fold | `src/curva_operacion.py`, `app/models/curva_operacion.json` |
| RF15 | Localización de los recortes manuales por correlación de plantilla sobre 175 pares | `src/geometria_recorte.py`, `app/models/sugerencia_recorte.json` |
| RF16 | Rejilla 5×5 de desplazamientos sobre los 20 casos del test sellado | `src/sensibilidad_recorte.py`, `resultados_v2/sensibilidad_recorte.json` |
| RNF01 | Auditoría del código: cero coincidencias de escritura y de red | `app/PRIVACIDAD_Y_FLUJO_DE_DATOS.md` |
| RNF02 | 5 repeticiones de 100 inferencias, con su variabilidad | `src/latencia.py`, `app/models/latencia.json` |
