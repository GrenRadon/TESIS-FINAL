"""Exporta el modelo final a ONNX y verifica la equivalencia tensor a tensor (C-06).

Usa la misma fuente de preprocesamiento que el entrenamiento y que app.py.
Escribe models/config_modelo.json con el umbral y la version del pipeline, para que la
app no pueda quedar desincronizada del modelo.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / 'src'))
import preprocesamiento as PP  # noqa: E402

DIR_MODELOS = Path(__file__).resolve().parent / 'models'
TOLERANCIA = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modelo', required=True, help='ruta al .keras final')
    ap.add_argument('--arquitectura', required=True, choices=list(PP.PREPROCESS))
    ap.add_argument('--umbral', type=float, required=True, help='umbral elegido en validacion')
    ap.add_argument('--salida', default=None)
    ap.add_argument('--muestras', default='resultados_v2/dataset_nivel_imagen.csv',
                    help='CSV con columna ruta, para la prueba de equivalencia')
    args = ap.parse_args()

    import tensorflow as tf
    import tf2onnx
    import onnxruntime as ort

    # El config acompana SIEMPRE al .onnx en su misma carpeta, para que un modelo no pueda
    # quedar desincronizado de su umbral. Exportar a otra carpeta no toca la app.
    salida = (Path(args.salida) if args.salida
              else DIR_MODELOS / f'chiari_{args.arquitectura}_final.onnx')
    salida.parent.mkdir(parents=True, exist_ok=True)
    dir_config = salida.parent

    print(f'Cargando {args.modelo}')
    modelo = tf.keras.models.load_model(args.modelo)

    print('Convirtiendo a ONNX...')
    sig = [tf.TensorSpec([1, 224, 224, 3], tf.float32, name='entrada_0_255')]
    onnx_model, _ = tf2onnx.convert.from_keras(modelo, input_signature=sig, opset=13)
    salida.write_bytes(onnx_model.SerializeToString())
    mb = salida.stat().st_size / 1e6
    print(f'Guardado: {salida}  ({mb:.1f} MB)')

    # ---------- prueba de equivalencia Keras vs ONNX ----------
    import csv
    rutas = [r['ruta'] for r in csv.DictReader(open(RAIZ / args.muestras, encoding='utf-8'))][:20]
    ses = ort.InferenceSession(str(salida))
    nombre_entrada = ses.get_inputs()[0].name
    difs = []
    for r in rutas:
        x = PP.cargar_imagen(RAIZ / r)[None, ...].astype(np.float32)
        p_keras = float(modelo.predict(x, verbose=0)[0][0])
        p_onnx = float(ses.run(None, {nombre_entrada: x})[0][0][0])
        difs.append(abs(p_keras - p_onnx))
    peor = max(difs)
    print(f'\nEquivalencia Keras vs ONNX en {len(rutas)} imagenes: '
          f'max|dif| = {peor:.3e}  ->  {"OK" if peor < TOLERANCIA else "FALLA"}')
    if peor >= TOLERANCIA:
        raise SystemExit('La exportacion ONNX no es equivalente al modelo Keras.')

    cfg = dict(arquitectura=args.arquitectura, umbral=round(float(args.umbral), 4),
               pipeline=PP.VERSION, modelo_origen=str(args.modelo),
               onnx=salida.name, tamano_mb=round(mb, 1),
               max_dif_keras_onnx=float(f'{peor:.3e}'), fecha=str(date.today()),
               poblacion_objetivo='adultos (>=18 anios)',
               entrada='float32 [1,224,224,3] en rango [0,255]; preprocess_input va dentro del modelo')
    (dir_config / 'config_modelo.json').write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Config: {dir_config / "config_modelo.json"}')


if __name__ == '__main__':
    main()
