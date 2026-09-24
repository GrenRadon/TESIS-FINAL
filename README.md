# Apoyo a la identificación de imágenes compatibles con malformación de Chiari I

Clasificador de resonancias magnéticas sagitales T1 recortadas a la fosa posterior, con una
aplicación web para usarlo. Proyecto de grado, Universidad Industrial de Santander.

> **Uso no clínico.** La herramienta clasifica una **imagen** como compatible o no compatible
> con hallazgos de MC-I. No emite diagnóstico, no mide el descenso amigdalar y no ha sido
> validada clínicamente ni de forma prospectiva. Población objetivo: pacientes adultos de
> 18 años o más.

---

## Qué hace, en una tabla

| | |
|---|---|
| Arquitectura | DenseNet121 con transferencia de aprendizaje, cabeza propia |
| Entrada | Resonancia sagital T1; la aplicación permite recortar la fosa posterior |
| Salida | Probabilidad, y clasificación respecto al umbral **0,1484** |
| Validación | `StratifiedGroupKFold` k=5 **agrupada por caso**, 97 casos de desarrollo |
| AUC out-of-fold | **0,8456** [0,761 – 0,921] |
| Sensibilidad · Especificidad | 0,9508 [0,865 – 0,983] · 0,5278 [0,370 – 0,680] |
| Conjunto de prueba sellado | 20 casos, abierto **una sola vez** |

El umbral no es 0,5. Se eligió maximizando la especificidad con sensibilidad ≥ 0,95, usando
**solo** las predicciones out-of-fold del conjunto de desarrollo.

---

## Ejecutar en local

```bash
python -m venv .venv
.venv\Scripts\activate          # en Linux o macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app/app.py
```

Se abre en <http://localhost:8501>. No hace falta TensorFlow: el modelo se sirve en ONNX.

Para **reproducir el pipeline** (entrenar, evaluar, regenerar figuras) hacen falta las
dependencias de desarrollo y las imágenes de origen:

```bash
pip install -r requirements-dev.txt
```

---

## Estructura

```
app/
  app.py                     la aplicación
  convert_to_onnx.py         exporta el modelo y verifica Keras ≡ ONNX
  models/                    el modelo desplegado y sus artefactos de configuración
src/                         el pipeline completo, 43 scripts
resultados_v2/               artefactos del conjunto de datos y los análisis
resultados_v4/               artefactos de la corrida definitiva
resultados_v2/figures/       las figuras, en PNG a 300 ppp y PDF vectorial
docs/                        documentación (ver abajo)
data/                        imágenes de origen — NO versionadas, ver más abajo
```

### Documentación

| Documento | Qué contiene |
|---|---|
| `docs/DESPLIEGUE.md` | Cómo desplegar en Streamlit Community Cloud |
| `docs/REPRODUCIBILIDAD.md` | Orden de ejecución, entorno, verificaciones automáticas |
| `docs/REQUISITOS.md` | 16 requisitos funcionales, 9 no funcionales y 8 ausencias declaradas |
| `docs/PRIVACIDAD_Y_FLUJO_DE_DATOS.md` | Recorrido de la imagen, y qué NO cubre esa afirmación |
| `docs/DIAGRAMAS_CODIGO.md` | Los diagramas del trabajo como código Mermaid y Graphviz |
| `docs/CATALOGO_FIGURAS.md` | Cada figura, su origen y el script que la genera |

---

## Las imágenes no están en el repositorio

Las **177 imágenes** de 117 casos proceden de **Radiopaedia.org** bajo licencia
**CC BY-NC-SA 3.0**, con las modificaciones de sus Términos de Uso. Este trabajo **no las
redistribuye**, y `.gitignore` deja `data/` fuera del repositorio.

Para recuperarlas: `resultados_v2/procedencia_imagenes.csv` contiene, para cada una, el
**autor**, el **rID**, la **URL del caso**, la **fecha de acceso** y el **sha256**, de modo
que un tercero puede descargarlas y verificar que son idénticas a las utilizadas aquí.

La aplicación **no necesita** las imágenes: el modelo ya está entrenado y exportado.

---

## Qué NO hace la herramienta

Consta explícitamente, porque su ausencia es una decisión declarada y no un olvido. La lista
completa con su justificación está en `docs/REQUISITOS.md`.

- No verifica que la imagen sea una resonancia magnética, ni que el plano sea sagital medio.
- No localiza automáticamente la fosa posterior: **el recorte lo delimita el usuario**, y el
  encuadre influye en el resultado. Está medido: ver `resultados_v2/sensibilidad_recorte.json`.
- No mide el descenso amigdalar en milímetros.
- No emite diagnóstico ni indicación terapéutica.
- No está validada en población pediátrica.
- No ha sido validada de forma externa ni prospectiva.
- No ha sido sometida a evaluación de ciberseguridad ni de impacto en protección de datos.
  **No se reivindica privacidad por diseño.**

---

## Publicar en GitHub

El repositorio está pensado para ser **privado**. Si aún no existe:

1. Crea el repositorio vacío en <https://github.com/new>, **sin** README ni `.gitignore`
   (este proyecto ya trae el suyo).
2. Desde esta carpeta:

```bash
git init
git branch -M main
git add .
git commit -m "Version inicial del proyecto"
git remote add origin https://github.com/<tu-usuario>/chiari-mci-apoyo.git
git push -u origin main
```

Antes del primer `push`, comprueba que las imágenes quedan fuera:

```bash
git status --short | grep "^A.*data/" || echo "correcto: data/ no se va a subir"
```

---

## Licencia

**El código** de este repositorio se publica bajo licencia MIT (ver `LICENSE`).

**Las imágenes de origen NO están incluidas** y conservan su licencia CC BY-NC-SA 3.0 de
Radiopaedia.org, con atribución a cada colaborador. Consulta la política oficial en
<https://doi.org/10.53347/rID-89930>.

**El modelo entrenado** (`app/models/*.onnx`) es un trabajo derivado de esas imágenes. La
cláusula *ShareAlike* de la licencia de origen condicionaría cualquier redistribución
comercial o con licencia distinta.
