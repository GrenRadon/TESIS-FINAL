"""C-06: prueba de equivalencia tensor a tensor entre las TRES fuentes + latencia (RNF02).

  1. entrenamiento : FoldGen -> PP.cargar_imagen(ruta)
  2. exportacion   : convert_to_onnx -> PP.cargar_imagen(ruta)
  3. app           : app.py -> PP.cargar_imagen(PIL.Image abierta del uploader)

Se comparan los tensores de entrada y las probabilidades de salida.
"""
import csv, io, json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / 'src'))
import preprocesamiento as PP  # noqa: E402

ONNX = RAIZ / 'app' / 'models' / 'chiari_DenseNet121_final.onnx'
CFG = json.loads((RAIZ / 'app' / 'models' / 'config_modelo.json').read_text(encoding='utf-8'))
N = 20
TOL = 1e-5


def main():
    rutas = [r['ruta'] for r in csv.DictReader(
        open(RAIZ / 'resultados_v2' / 'dataset_nivel_imagen.csv', encoding='utf-8'))][:N]

    print('=' * 76)
    print(f'  EQUIVALENCIA DE LAS 3 FUENTES — pipeline v{PP.VERSION}')
    print('=' * 76)
    print(f'  Orden canonico: {PP.descripcion(CFG["arquitectura"])}\n')

    ses = ort.InferenceSession(str(ONNX))
    nombre = ses.get_inputs()[0].name
    print(f'  entrada ONNX: {nombre} {ses.get_inputs()[0].shape} {ses.get_inputs()[0].type}')

    d_tensor, d_prob = [], []
    for r in rutas:
        p = RAIZ / r
        # fuente 1 y 2: desde ruta en disco
        t_entrena = PP.cargar_imagen(p)
        # fuente 3: como lo hace la app, desde un buffer subido
        buf = io.BytesIO(p.read_bytes())
        t_app = PP.cargar_imagen(Image.open(buf))
        d_tensor.append(float(np.abs(t_entrena - t_app).max()))
        p1 = float(ses.run(None, {nombre: t_entrena[None].astype(np.float32)})[0][0][0])
        p2 = float(ses.run(None, {nombre: t_app[None].astype(np.float32)})[0][0][0])
        d_prob.append(abs(p1 - p2))

    print(f'\n  tensores  entrenamiento vs app : max|dif| = {max(d_tensor):.3e}')
    print(f'  salidas   entrenamiento vs app : max|dif| = {max(d_prob):.3e}')
    print(f'  Keras vs ONNX (de la exportacion): max|dif| = {CFG["max_dif_keras_onnx"]:.3e}')
    ok = max(d_tensor) < TOL and max(d_prob) < TOL and CFG['max_dif_keras_onnx'] < TOL
    print(f'\n  RESULTADO: {"LAS 3 FUENTES COINCIDEN" if ok else "DIFIEREN — revisar"}')

    # ---------- latencia RNF02 ----------
    print('\n' + '=' * 76)
    print('  LATENCIA DE INFERENCIA (RNF02) — CPU, ONNX Runtime')
    print('=' * 76)
    img = PP.cargar_imagen(RAIZ / rutas[0])[None].astype(np.float32)
    for _ in range(5):
        ses.run(None, {nombre: img})

    solo_inf, extremo = [], []
    for r in rutas:
        p = RAIZ / r
        t0 = time.perf_counter()
        x = PP.cargar_imagen(p)[None].astype(np.float32)
        t1 = time.perf_counter()
        ses.run(None, {nombre: x})
        t2 = time.perf_counter()
        solo_inf.append((t2 - t1) * 1000)
        extremo.append((t2 - t0) * 1000)

    def q(v, p):
        return float(np.percentile(v, p))

    print(f'  {"":<26}{"p50":>9}{"p95":>9}{"max":>9}')
    print(f'  {"solo inferencia ONNX":<26}{q(solo_inf,50):>8.0f}ms{q(solo_inf,95):>8.0f}ms{max(solo_inf):>8.0f}ms')
    print(f'  {"preproceso + inferencia":<26}{q(extremo,50):>8.0f}ms{q(extremo,95):>8.0f}ms{max(extremo):>8.0f}ms')
    print(f'\n  tamano ONNX: {ONNX.stat().st_size / 1e6:.1f} MB')
    cumple = q(extremo, 95) < 1000
    print(f'  RNF02 (< 1 s de extremo a extremo): {"CUMPLE" if cumple else "NO CUMPLE"} '
          f'(p95 = {q(extremo,95):.0f} ms)')

    json.dump(dict(pipeline=PP.VERSION, n_muestras=N,
                   max_dif_tensor_entrenamiento_app=float(f'{max(d_tensor):.3e}'),
                   max_dif_prob_entrenamiento_app=float(f'{max(d_prob):.3e}'),
                   max_dif_keras_onnx=CFG['max_dif_keras_onnx'], tres_fuentes_coinciden=bool(ok),
                   onnx_mb=round(ONNX.stat().st_size / 1e6, 1),
                   latencia_inferencia_p50_ms=round(q(solo_inf, 50), 1),
                   latencia_inferencia_p95_ms=round(q(solo_inf, 95), 1),
                   latencia_extremo_p50_ms=round(q(extremo, 50), 1),
                   latencia_extremo_p95_ms=round(q(extremo, 95), 1),
                   rnf02_cumple=bool(cumple)),
              open(RAIZ / 'resultados_final' / 'equivalencia_y_latencia.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'\nJSON: resultados_final/equivalencia_y_latencia.json')


if __name__ == '__main__':
    main()
