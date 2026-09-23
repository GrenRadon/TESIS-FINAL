"""M-18 — validación de la entrada y detección fuera de distribución.

El hallazgo objeta que aceptar JPG/PNG no garantiza que la imagen sea una MRI sagital T1
recortada a fosa posterior, y que no hay control de calidad ni detección de entradas fuera
de distribución. Las pruebas de M-17 lo confirmaron: una imagen de 1x1 px obtiene 0,9996 y
ruido aleatorio obtiene 0,8348, ambos por encima del umbral.

Dos capas:

  1. CONTROL DE CALIDAD — reglas duras y auditables (resolución mínima, imagen no constante,
     contraste suficiente, proporción razonable). Rechazan la entrada.

  2. DETECCIÓN FUERA DE DISTRIBUCIÓN — distancia de Mahalanobis sobre estadísticos simples
     de imagen, con la referencia calculada SOBRE EL CONJUNTO DE DESARROLLO. No rechaza:
     advierte, porque un falso rechazo en una imagen legítima es peor que una advertencia.

Lo que esto NO hace, y debe declararse: no verifica que la imagen sea una MRI, ni que el
plano sea sagital medio, ni que la ROI sea la fosa posterior. Para eso haría falta un
clasificador de modalidad/plano entrenado, que queda como trabajo futuro (C-07).
"""
import json
from pathlib import Path

import numpy as np

REFERENCIA = Path(__file__).resolve().parent.parent / 'app' / 'models' / 'referencia_ood.json'

# Reglas duras. Los valores se justifican en el propio mensaje de error.
MIN_LADO = 64           # por debajo, el redimensionado a 224 inventa información
MAX_ASPECTO = 3.0       # un recorte de fosa posterior no es una tira
MIN_STD = 8.0           # sobre 0-255; por debajo la imagen es casi constante
MIN_NIVELES = 16        # niveles de gris distintos


def rasgos(arr):
    """Estadísticos simples e interpretables de una imagen en escala de grises 0-255."""
    a = np.asarray(arr, dtype=np.float64)
    h = np.histogram(a, bins=32, range=(0, 255))[0].astype(float)
    p = h / max(h.sum(), 1)
    entropia = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    gy, gx = np.gradient(a)
    bordes = float(np.mean(np.hypot(gx, gy)))
    return np.array([a.mean(), a.std(), entropia, bordes,
                     float((a > 40).mean()), float(np.median(a))])


NOMBRES = ['media', 'desv. típica', 'entropía', 'densidad de bordes',
           'fracción de tejido', 'mediana']


def control_calidad(arr):
    """Reglas duras. Devuelve lista de problemas; vacía = pasa."""
    a = np.asarray(arr, dtype=np.float64)
    problemas = []
    h, w = a.shape[:2]
    if min(h, w) < MIN_LADO:
        problemas.append(f'resolución insuficiente ({w}×{h} px; mínimo {MIN_LADO} px por lado)')
    asp = max(w / h, h / w)
    if asp > MAX_ASPECTO:
        problemas.append(f'proporción anómala ({asp:.1f}:1; máximo admitido {MAX_ASPECTO:.0f}:1)')
    if a.std() < MIN_STD:
        problemas.append(f'contraste insuficiente (desviación típica {a.std():.1f}; mínimo {MIN_STD})')
    if len(np.unique(a.astype(np.uint8))) < MIN_NIVELES:
        problemas.append(f'imagen casi constante ({len(np.unique(a.astype(np.uint8)))} niveles de gris)')
    return problemas


def cargar_referencia():
    if not REFERENCIA.exists():
        return None
    d = json.loads(REFERENCIA.read_text(encoding='utf-8'))
    return (np.array(d['media']), np.array(d['inv_cov']),
            d['umbral_aviso'], d['umbral_fuerte'], d['n_referencia'])


def distancia_ood(arr, ref=None):
    """Mahalanobis sobre los rasgos. Mayor = más lejos de lo visto en entrenamiento."""
    ref = ref or cargar_referencia()
    if ref is None:
        return None, None
    mu, inv, av, fu, _ = ref
    x = rasgos(arr) - mu
    d = float(np.sqrt(max(x @ inv @ x, 0)))
    nivel = 'fuera de distribución' if d >= fu else ('atípica' if d >= av else 'dentro del rango')
    return d, nivel


def validar(arr):
    """Devuelve (aceptable, problemas, distancia, nivel).

    El control de calidad va PRIMERO: si la imagen no lo pasa, ni siquiera se calcula la
    distancia (sobre una imagen degenerada los estadísticos no están definidos).
    """
    problemas = control_calidad(arr)
    if problemas:
        return False, problemas, None, None
    d, nivel = distancia_ood(arr)
    return True, [], d, nivel


# ---------------------------------------------------------------- construcción
def construir_referencia():
    """Calcula media y covarianza de los rasgos sobre el conjunto de DESARROLLO."""
    import csv
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from PIL import Image

    base = Path(__file__).resolve().parent.parent
    filas = [r for r in csv.DictReader(open(base / 'resultados_v2' / 'dataset_nivel_imagen.csv',
                                            encoding='utf-8'))
             if r['split'] == 'desarrollo']
    X = np.array([rasgos(np.array(Image.open(base / r['ruta']).convert('L'))) for r in filas])
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
    inv = np.linalg.inv(cov)
    d = np.array([np.sqrt(max((x - mu) @ inv @ (x - mu), 0)) for x in X])
    av, fu = float(np.percentile(d, 95)), float(np.percentile(d, 99.5))
    REFERENCIA.parent.mkdir(parents=True, exist_ok=True)
    REFERENCIA.write_text(json.dumps(
        dict(media=mu.tolist(), inv_cov=inv.tolist(), umbral_aviso=round(av, 4),
             umbral_fuerte=round(fu, 4), n_referencia=len(X), rasgos=NOMBRES,
             nota='Referencia calculada sobre el conjunto de desarrollo. Los umbrales son los '
                  'percentiles 95 y 99,5 de la distancia de Mahalanobis dentro de ese conjunto.'),
        indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'  referencia construida sobre {len(X)} imágenes de desarrollo')
    print(f'  distancia: mediana {np.median(d):.2f} · p95 {av:.2f} · p99,5 {fu:.2f} · max {d.max():.2f}')
    return mu, inv, av, fu, len(X)


if __name__ == '__main__':
    construir_referencia()
