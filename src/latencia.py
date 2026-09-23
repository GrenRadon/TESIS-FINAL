"""Medicion de latencia estable para RNF02, con su variabilidad declarada.

Motivo: la bateria de pruebas mide la latencia en cada corrida y el valor fluctua unos pocos
milisegundos entre ejecuciones, lo que produjo cifras distintas en documentos distintos
(39/55, 45/56, 39/42). No es un error de transcripcion: es ruido de medicion que nadie habia
caracterizado.

Este script hace varias repeticiones completas, reporta la mediana de cada percentil entre
repeticiones y su rango, para que el libro cite UNA cifra con su incertidumbre.

Escribe app/models/latencia.json.

Uso:  python src/latencia.py
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocesamiento as PP

APP = Path('app/models')
SALIDA = APP / 'latencia.json'
REPETICIONES = 5
N_POR_REPETICION = 100
CALENTAMIENTO = 20


def main():
    onnx = next(APP.glob('*.onnx'))
    ses = ort.InferenceSession(str(onnx))
    ent = ses.get_inputs()[0].name
    rng = np.random.default_rng(42)
    lote = [rng.integers(0, 256, (224, 224, 3)).astype(np.float32)
            for _ in range(N_POR_REPETICION)]

    for x in lote[:CALENTAMIENTO]:
        ses.run(None, {ent: x[None, ...]})

    reps = []
    for _ in range(REPETICIONES):
        t = []
        for x in lote:
            t0 = time.perf_counter()
            ses.run(None, {ent: x[None, ...]})
            t.append((time.perf_counter() - t0) * 1000)
        reps.append(dict(p50=float(np.percentile(t, 50)), p95=float(np.percentile(t, 95)),
                         p99=float(np.percentile(t, 99)), max=float(max(t))))

    def agg(k):
        v = [r[k] for r in reps]
        return dict(mediana=round(float(np.median(v)), 1),
                    minimo=round(float(min(v)), 1), maximo=round(float(max(v)), 1))

    d = dict(
        modelo=onnx.name, procesador=platform.processor(),
        sistema=f'{platform.system()} {platform.release()}',
        gpu=False, onnxruntime=ort.__version__,
        repeticiones=REPETICIONES, inferencias_por_repeticion=N_POR_REPETICION,
        calentamiento=CALENTAMIENTO,
        entrada='tensor sintetico 224x224x3 en [0,255], mismo formato que produce el pipeline',
        p50_ms=agg('p50'), p95_ms=agg('p95'), p99_ms=agg('p99'), max_ms=agg('max'),
        nota='Solo INFERENCIA del grafo ONNX. No incluye la lectura del archivo ni el '
             'preprocesamiento, que se miden aparte en pruebas_app.py (imagen de 4000x4000 '
             'extremo a extremo). La cifra a citar es la mediana entre repeticiones; el rango '
             'da la variabilidad entre ejecuciones en la misma maquina.')
    SALIDA.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'  {onnx.name} · CPU · {REPETICIONES} repeticiones de {N_POR_REPETICION}')
    for k in ('p50_ms', 'p95_ms', 'p99_ms', 'max_ms'):
        a = d[k]
        print(f'  {k:<8} mediana {a["mediana"]:>6.1f} ms   '
              f'[entre repeticiones: {a["minimo"]:.1f} – {a["maximo"]:.1f}]')
    print(f'  JSON: {SALIDA}')


if __name__ == '__main__':
    main()
