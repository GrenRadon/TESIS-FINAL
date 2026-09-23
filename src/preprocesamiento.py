"""FUENTE UNICA DE PREPROCESAMIENTO — C-06 / C-08.

Toda ruta (entrenamiento, exportacion ONNX y app) debe importar de aqui.
No duplicar estas operaciones en ningun otro archivo.

ORDEN CANONICO, en este orden exacto:
    1. abrir imagen y convertir a escala de grises (modo 'L')
    2. CLAHE  (clipLimit=2.0, tileGridSize=(8,8))  <- SOBRE LA RESOLUCION ORIGINAL
    3. redimensionar a 224x224 con LANCZOS
    4. apilar a 3 canales
    5. salida float32 en rango [0, 255]   <- NO se divide por 255 aqui

El paso 5 es deliberado: la normalizacion especifica de cada arquitectura
(preprocess_input) va DENTRO del modelo, como capas. Asi existe un unico lugar donde
puede desviarse y viaja con el .keras y con el .onnx exportado.

Antes (incoherente entre fuentes):
    nb_03    : gris -> CLAHE -> resize -> /255
    app.py   : gris -> resize -> CLAHE -> /255     <- orden invertido
    ninguno  : preprocess_input por arquitectura   <- pesos ImageNet mal usados
"""
VERSION = '2.0.0'

IMG_SIZE = (224, 224)
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# Constantes de normalizacion por arquitectura, equivalentes a
# keras.applications.<arq>.preprocess_input aplicado sobre entrada en [0,255].
PREPROCESS = {
    # densenet.preprocess_input -> modo 'torch': x/255, luego (x-mean)/std de ImageNet
    'DenseNet121': dict(modo='torch', escala=1 / 255.0, offset=0.0,
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # resnet_v2.preprocess_input -> modo 'tf': x/127.5 - 1
    'ResNet50V2': dict(modo='tf', escala=1 / 127.5, offset=-1.0, mean=None, std=None),
    # efficientnet.preprocess_input es identidad: el modelo ya lleva Rescaling+Normalization
    'EfficientNetB0': dict(modo='identidad', escala=1.0, offset=0.0, mean=None, std=None),
}


def cargar_imagen(origen):
    """Devuelve float32 [224,224,3] en rango [0,255]. `origen` = ruta o PIL.Image."""
    import numpy as np
    import cv2
    from PIL import Image

    im = origen if hasattr(origen, 'convert') else Image.open(origen)
    arr = np.array(im.convert('L'), dtype=np.uint8)          # 1. gris
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    arr = clahe.apply(arr)                                    # 2. CLAHE en resolucion original
    arr = np.array(Image.fromarray(arr).resize(IMG_SIZE, Image.LANCZOS),
                   dtype=np.float32)                          # 3. resize
    return np.stack([arr, arr, arr], axis=-1)                 # 4-5. 3 canales, [0,255]


def capas_preprocess(arquitectura):
    """Capas Keras que replican preprocess_input. Van dentro del modelo."""
    from tensorflow import keras
    from tensorflow.keras import layers

    cfg = PREPROCESS[arquitectura]
    if cfg['modo'] == 'identidad':
        return []
    caps = [layers.Rescaling(cfg['escala'], offset=cfg['offset'],
                             name=f'preproc_rescale_{arquitectura}')]
    if cfg['mean'] is not None:
        import numpy as np
        caps.append(layers.Normalization(
            axis=-1, mean=cfg['mean'],
            variance=[s ** 2 for s in cfg['std']],
            name=f'preproc_norm_{arquitectura}'))
    return caps


def descripcion(arquitectura):
    cfg = PREPROCESS[arquitectura]
    p = (f"v{VERSION} | gris -> CLAHE(clip={CLAHE_CLIP}, tile={CLAHE_TILE}) -> "
         f"resize{IMG_SIZE} LANCZOS -> 3ch -> [0,255] | dentro del modelo: ")
    if cfg['modo'] == 'identidad':
        return p + f"identidad ({arquitectura} normaliza internamente)"
    s = f"x*{cfg['escala']:.6g}"
    if cfg['offset']:
        s += f" {cfg['offset']:+g}"
    if cfg['mean'] is not None:
        s += f", luego (x-{cfg['mean']})/{cfg['std']}"
    return p + s
