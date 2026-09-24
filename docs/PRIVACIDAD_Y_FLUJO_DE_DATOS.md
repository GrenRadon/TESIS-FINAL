# Flujo de datos, privacidad y seguridad del servicio — M-19 / M-25

**Última revisión: 2026-09-06** · Modelo desplegado: `chiari_DenseNet121_final_v4_sin_flip.onnx`

El hallazgo M-19 objeta que *"no escribir explícitamente a disco no demuestra ausencia de logs,
cachés, telemetría, copias de plataforma o persistencia temporal"*. Este documento describe lo
que **sí se puede afirmar del código**, lo que **depende del entorno de despliegue** y lo que
**no se ha evaluado**. No reivindica privacidad por diseño.

---

## 1. Recorrido de una imagen dentro de la aplicación

| Paso | Qué ocurre | Persistencia |
|---|---|---|
| 1. Subida | `st.file_uploader` recibe el archivo | Streamlit lo mantiene **en memoria** como `UploadedFile`; en configuraciones por defecto no lo escribe en disco |
| 2. Apertura | `PIL.Image.open(archivo)` | En memoria |
| 3. Validación | `validacion_entrada.validar()` sobre el array | En memoria |
| 4. Preprocesamiento | `preprocesamiento.cargar_imagen()` → tensor float32 | En memoria |
| 5. Inferencia | `onnxruntime` sobre el tensor | En memoria |
| 6. Presentación | Se muestran imagen y probabilidad | En la sesión del navegador |

**El código de la aplicación no ejecuta ninguna operación de escritura de la imagen ni de la
predicción a disco, base de datos o servicio externo.** Verificable: no hay llamadas a `open(...,'w')`,
`savefig`, `to_csv`, `requests`, ni cliente de telemetría en `app/app.py`.

**Lo que esa afirmación NO cubre**, y por eso no basta:

- El buffer temporal que el servidor web pueda crear al recibir el `multipart/form-data`.
- La caché de Streamlit (`st.cache_resource` guarda **el modelo**, no las imágenes, pero el
  mecanismo de caché en disco existe y podría activarse con otra configuración).
- Los registros del servidor HTTP subyacente (accesos, tamaños, direcciones IP).
- Copias, instantáneas o registros de la plataforma de alojamiento.
- Telemetría de Streamlit, activada por defecto salvo que se desactive explícitamente.

---

## 2. Configuración necesaria antes de un despliegue público

Ninguna de estas medidas está aplicada por el hecho de ejecutar la aplicación: hay que
configurarlas en el entorno.

| Medida | Cómo | Estado |
|---|---|---|
| Desactivar telemetría de Streamlit | `[browser] gatherUsageStats = false` en `.streamlit/config.toml` | ✅ Aplicado |
| Limitar tamaño de subida | `maxUploadSize = 10` | ✅ Aplicado |
| Forzar HTTPS/TLS | Depende del proveedor; obligatorio si se procesan imágenes de personas | **Pendiente** |
| Política de retención de logs | Definir y documentar qué registra el proveedor y por cuánto tiempo | **Pendiente** |
| Ubicación de procesamiento | Declarar en qué país se procesan los datos | **Pendiente** |
| Aviso de privacidad visible | Texto accesible antes de subir la primera imagen | **Pendiente** |

---

## 3. Lo que NO se ha evaluado

Declarado de forma explícita, en lugar de omitido:

- **No se ha realizado evaluación formal de ciberseguridad** del servicio: ni análisis de
  vulnerabilidades, ni pruebas de penetración, ni revisión de dependencias.
- **No se ha realizado evaluación de impacto en protección de datos.**
- **No hay autenticación ni control de acceso**: cualquiera con la URL puede usar el servicio.
- **No hay registro de auditoría** de quién sube qué, lo cual es coherente con no almacenar,
  pero impide investigar un incidente.
- **No se ha verificado el cumplimiento** de la normativa colombiana de protección de datos
  (Ley 1581 de 2012) ni de otras jurisdicciones desde las que pudiera accederse.

---

## 4. Cambio de naturaleza del riesgo (M-25)

La investigación retrospectiva analiza imágenes **públicas y ya publicadas** en Radiopaedia.
Un servicio público abierto procesa **imágenes de salud que aporta el usuario**, cuya
procedencia y titularidad se desconocen. No son la misma actividad y no comparten el mismo
marco de riesgo.

Consecuencia práctica: la clasificación de "investigación sin riesgo" aplicable al estudio
retrospectivo **no se extiende automáticamente** al servicio público, y el despliegue debería
someterse a revisión institucional antes de abrirse.

---

## 5. Medidas técnicas efectivamente implementadas

Verificables en el código, a fecha de esta revisión:

- La imagen no se escribe a disco desde el código de la aplicación.
- Límite de subida de 10 MB.
- Se aceptan únicamente JPG, JPEG y PNG.
- Rechazo de imágenes que no superan el control de calidad (resolución, contraste, proporción).
- Aviso explícito cuando la entrada queda fuera de la distribución de entrenamiento.
- Advertencia de uso no clínico y de población objetivo adulta, visible antes de subir nada.
- La aplicación se detiene con error si hay más de un modelo en `models/`, para impedir que se
  aplique un umbral que no corresponde al modelo cargado.

---

## 6. Recomendación

**No abrir el servicio al público** hasta cubrir los puntos pendientes de la sección 2 y
obtener el concepto institucional de la sección 4. Mientras tanto, el uso debería restringirse
a un entorno controlado con fines académicos y de demostración.
