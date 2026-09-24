# Despliegue en Streamlit Community Cloud

La aplicación está preparada para desplegarse sin cambios. Lo único que hay que saber de
antemano es que **el modelo va dentro del repositorio** (`app/models/*.onnx`, 27 MB), de modo
que no hace falta ningún almacenamiento externo ni credenciales.

---

## Antes de empezar

| Requisito | Cómo se cumple aquí |
|---|---|
| Repositorio en GitHub | Puede ser **privado**: Streamlit Cloud lo lee si le concedes acceso |
| Fichero de entrada | `app/app.py` |
| Dependencias | `requirements.txt` en la raíz |
| Versión de Python | `runtime.txt` fija `python-3.11` |
| Configuración | `.streamlit/config.toml` en la raíz |
| Modelo | `app/models/chiari_DenseNet121_final_v4_sin_flip.onnx`, versionado |

**No subas `data/`.** Está en `.gitignore` y debe seguir estándolo: son imágenes de
Radiopaedia bajo CC BY-NC-SA 3.0 y este trabajo declara que no las redistribuye. La
aplicación no las necesita para funcionar.

---

## Pasos

### 1 · Sube el repositorio a GitHub

Ver `README.md`, sección «Publicar en GitHub».

### 2 · Conecta Streamlit Cloud

1. Entra en <https://share.streamlit.io> e inicia sesión con la cuenta de GitHub.
2. **New app** → **Deploy a public app from GitHub** (también admite privados; te pedirá
   autorizar el acceso a los repositorios).
3. Rellena:

| Campo | Valor |
|---|---|
| Repository | `<tu-usuario>/chiari-mci-apoyo` |
| Branch | `main` |
| Main file path | `app/app.py` |
| App URL | el subdominio que prefieras |

4. En **Advanced settings**, comprueba que la versión de Python sea **3.11**.
5. **Deploy**.

El primer despliegue tarda unos minutos: instala las dependencias y descarga el modelo del
repositorio.

### 3 · Comprueba que arrancó bien

Abre la aplicación y verifica, en este orden:

1. El título se lee completo y la cinta de uso no clínico está visible.
2. La pestaña **Ficha técnica** muestra el nombre del artefacto y **umbral 0,1484**.
3. Carga una resonancia sagital T1 y comprueba que el recuadro sugerido aparece.

Si el umbral que aparece es **0,5**, el despliegue no encontró `config_modelo.json`: la
aplicación lo avisa en rojo. Comprueba que el fichero esté junto al `.onnx` en el
repositorio.

---

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| «No se encontró un modelo .onnx en `models/`» | El `.onnx` no llegó al repositorio | Comprueba que `.gitignore` no lo excluya. `*.keras` y `*.h5` sí se excluyen, `*.onnx` no |
| «Hay N modelos .onnx en `models/`» | Quedó más de un modelo | Debe haber **exactamente uno**, junto a su `config_modelo.json`. La aplicación se detiene a propósito en vez de elegir uno al azar |
| `ImportError: libGL.so.1` | `opencv-python` en vez de la versión *headless* | `requirements.txt` ya usa `opencv-python-headless`. No lo cambies |
| El despliegue instala TensorFlow y se queda sin memoria | Se usó `requirements-dev.txt` | El despliegue usa **`requirements.txt`**, que no incluye TensorFlow. `requirements-dev.txt` es solo para reentrenar en local |
| La app tarda en responder la primera vez | Arranque en frío | Normal. Tras el primer arranque la sesión cachea el modelo con `@st.cache_resource` |

---

## Sobre recursos

La aplicación cabe con holgura en el plan gratuito:

- **Memoria**: el modelo ocupa 27 MB y `onnxruntime` añade poco. Sin TensorFlow, el consumo
  se mantiene muy por debajo del límite de 1 GB.
- **Latencia**: medida en CPU local, p50 34 ms y p95 43 ms por inferencia
  (`app/models/latencia.json`). Streamlit Cloud es más lento, pero el orden de magnitud se
  mantiene.
- **Límite de subida**: `.streamlit/config.toml` lo fija en **10 MB**, suficiente para una
  resonancia y una barrera razonable frente a subidas accidentales.

---

## Un aviso sobre privacidad que conviene no olvidar

El documento `PRIVACIDAD_Y_FLUJO_DE_DATOS.md` afirma que **el código** de la aplicación no
escribe la imagen en disco ni la transmite a terceros. Esa afirmación **no cubre el entorno de
despliegue**: al alojar la aplicación en Streamlit Cloud, las imágenes que suban los usuarios
pasan por la infraestructura de un tercero, sobre la que este trabajo no ha hecho ninguna
verificación.

Si la aplicación va a recibir imágenes de pacientes reales, eso deja de ser una nota al pie.
Antes de anunciarla públicamente conviene decidir y documentar la política de retención, la
ubicación de procesamiento y el aviso de privacidad, que hoy figuran como pendientes de
entorno.
