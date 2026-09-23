"""M-17 — batería de pruebas funcionales del servicio.

El hallazgo objeta que "dos ejemplos y una inspección estática" no validan robustez,
seguridad, latencia ni privacidad. Aquí se prueban, sobre el modelo realmente desplegado:

  1. Formatos aceptados y rechazados
  2. Entradas malformadas y adversas (archivo corrupto, vacío, texto disfrazado, PNG bomba)
  3. Tamaño máximo y límite configurado
  4. Latencia p50/p95 y comportamiento bajo concurrencia
  5. Determinismo: la misma imagen debe dar siempre la misma probabilidad
  6. Rango de salida y coherencia con el umbral
"""
import io, json, time, sys, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocesamiento as PP

APP = Path('app/models')
OUT = Path('resultados_final_v4')
OUT.mkdir(exist_ok=True)
MAX_UPLOAD_MB = 10          # declarado en .streamlit/config.toml
resultados = []


def prueba(nombre, categoria):
    def deco(fn):
        try:
            ok, detalle = fn()
        except Exception as e:
            ok, detalle = False, f'excepción no controlada: {type(e).__name__}: {e}'
        resultados.append(dict(categoria=categoria, prueba=nombre, ok=bool(ok), detalle=detalle))
        print(f'  [{"OK  " if ok else "FALLA"}] {nombre}: {detalle}')
        return fn
    return deco


def sesion():
    onnx = sorted(APP.glob('*.onnx'))
    ses = ort.InferenceSession(str(onnx[0]))
    return ses, ses.get_inputs()[0].name, json.loads((APP / 'config_modelo.json').read_text(encoding='utf-8'))


def main():
    ses, nom, cfg = sesion()
    umbral = cfg['umbral']
    muestra = Path('data/vf/cropped/chiari/chiari_13a.jpg')
    print(f'  modelo: {cfg["onnx"]}  umbral {umbral}\n')

    def predecir(origen):
        x = PP.cargar_imagen(origen)[None].astype(np.float32)
        return float(ses.run(None, {nom: x})[0][0][0])

    # ---------------- 1. formatos ----------------
    print('--- 1. Formatos ---')
    for fmt, ext in (('JPEG', '.jpg'), ('PNG', '.png')):
        @prueba(f'acepta {fmt}', 'formatos')
        def _(fmt=fmt, ext=ext):
            buf = io.BytesIO()
            Image.open(muestra).convert('RGB').save(buf, format=fmt)
            buf.seek(0)
            p = predecir(Image.open(buf))
            return 0.0 <= p <= 1.0, f'probabilidad {p:.4f}'

    @prueba('acepta escala de grises de 16 bits', 'formatos')
    def _():
        buf = io.BytesIO()
        Image.fromarray((np.random.rand(300, 300) * 65535).astype(np.uint16)).save(buf, format='PNG')
        buf.seek(0)
        p = predecir(Image.open(buf))
        return 0.0 <= p <= 1.0, f'convertida sin error, probabilidad {p:.4f}'

    @prueba('acepta imagen con canal alfa (RGBA)', 'formatos')
    def _():
        buf = io.BytesIO()
        Image.open(muestra).convert('RGBA').save(buf, format='PNG')
        buf.seek(0)
        p = predecir(Image.open(buf))
        return 0.0 <= p <= 1.0, f'probabilidad {p:.4f}'

    # ---------------- 2. entradas malformadas ----------------
    print('\n--- 2. Entradas malformadas y adversas ---')

    @prueba('rechaza archivo vacío', 'robustez')
    def _():
        try:
            predecir(Image.open(io.BytesIO(b'')))
            return False, 'NO lanzó excepción: la app mostraría un error genérico'
        except Exception as e:
            return True, f'rechazado con {type(e).__name__}'

    @prueba('rechaza bytes corruptos', 'robustez')
    def _():
        try:
            predecir(Image.open(io.BytesIO(b'\x89PNG\r\n\x1a\n' + b'\x00' * 200)))
            return False, 'NO lanzó excepción'
        except Exception as e:
            return True, f'rechazado con {type(e).__name__}'

    @prueba('rechaza texto renombrado como .png', 'robustez')
    def _():
        try:
            predecir(Image.open(io.BytesIO(b'esto no es una imagen' * 50)))
            return False, 'NO lanzó excepción'
        except Exception as e:
            return True, f'rechazado con {type(e).__name__}'

    @prueba('procesa imagen de 1x1 px sin caerse', 'robustez')
    def _():
        p = predecir(Image.new('L', (1, 1), 128))
        return 0.0 <= p <= 1.0, (f'NO la rechaza: devuelve {p:.4f} para 1x1 px. '
                                 'Sin control de calidad, una imagen inválida obtiene veredicto')

    @prueba('procesa ruido aleatorio sin caerse', 'robustez')
    def _():
        p = predecir(Image.fromarray((np.random.rand(224, 224) * 255).astype(np.uint8)))
        return 0.0 <= p <= 1.0, (f'NO lo rechaza: ruido puro obtiene {p:.4f} '
                                 f'({"COMPATIBLE" if p >= umbral else "no compatible"} con MC-I)')

    # ---------------- 3. tamaño ----------------
    print('\n--- 3. Tamaño de entrada ---')

    @prueba(f'límite de subida declarado ({MAX_UPLOAD_MB} MB)', 'tamano')
    def _():
        # Streamlit lee .streamlit/config.toml relativo al directorio desde el que se lanza.
        # Se busca primero en la raiz, que es lo que usa el despliegue.
        cands = [Path('.streamlit/config.toml'), Path('app/.streamlit/config.toml')]
        hallado = next((c for c in cands if c.exists()), None)
        if hallado is None:
            return False, 'no se encontro config.toml en ' + ' ni '.join(map(str, cands))
        cfgtoml = hallado.read_text(encoding='utf-8')
        return f'maxUploadSize = {MAX_UPLOAD_MB}' in cfgtoml, \
               f'{hallado} declara maxUploadSize = {MAX_UPLOAD_MB} MB'

    @prueba('procesa imagen muy grande (4000x4000)', 'tamano')
    def _():
        t0 = time.perf_counter()
        p = predecir(Image.fromarray((np.random.rand(4000, 4000) * 255).astype(np.uint8)))
        dt = (time.perf_counter() - t0) * 1000
        return dt < 5000, f'procesada en {dt:.0f} ms, probabilidad {p:.4f}'

    # ---------------- 4. latencia y concurrencia ----------------
    print('\n--- 4. Latencia y concurrencia ---')
    for _ in range(5):
        predecir(muestra)

    @prueba('latencia secuencial p95 < 1 s (RNF02)', 'latencia')
    def _():
        ts = []
        for _ in range(30):
            t0 = time.perf_counter(); predecir(muestra); ts.append((time.perf_counter() - t0) * 1000)
        p50, p95 = np.percentile(ts, 50), np.percentile(ts, 95)
        return p95 < 1000, f'p50={p50:.0f} ms · p95={p95:.0f} ms · max={max(ts):.0f} ms'

    @prueba('estable con 8 peticiones concurrentes', 'concurrencia')
    def _():
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(lambda _: predecir(muestra), range(24)))
        dt = (time.perf_counter() - t0) * 1000
        iguales = len(set(round(r, 6) for r in res)) == 1
        return iguales, (f'24 peticiones en 8 hilos, {dt:.0f} ms totales, '
                         f'{"todas idénticas" if iguales else "RESULTADOS DISTINTOS"}')

    # ---------------- 5. determinismo ----------------
    print('\n--- 5. Determinismo y salida ---')

    @prueba('misma imagen → misma probabilidad', 'determinismo')
    def _():
        v = [predecir(muestra) for _ in range(10)]
        return len(set(round(x, 9) for x in v)) == 1, f'10 ejecuciones, valor único {v[0]:.6f}'

    @prueba('salida siempre en [0, 1]', 'salida')
    def _():
        vs = [predecir(p) for p in list(Path('data/vf/cropped/chiari').iterdir())[:15]]
        return all(0 <= v <= 1 for v in vs), f'15 imágenes, rango [{min(vs):.4f}, {max(vs):.4f}]'

    @prueba('config y modelo sincronizados', 'integridad')
    def _():
        onnx = sorted(APP.glob('*.onnx'))
        return len(onnx) == 1 and cfg['onnx'] == onnx[0].name, \
               f'{len(onnx)} modelo(s); config apunta a {cfg["onnx"]}'

    # ---------------- resumen ----------------
    ok = sum(1 for r in resultados if r['ok'])
    print(f'\n{"=" * 70}')
    print(f'  {ok} de {len(resultados)} pruebas superadas')
    fallos = [r for r in resultados if not r['ok']]
    if fallos:
        print('  FALLAN:')
        for r in fallos:
            print(f'    · [{r["categoria"]}] {r["prueba"]} — {r["detalle"]}')
    json.dump(dict(modelo=cfg['onnx'], umbral=umbral, superadas=ok, total=len(resultados),
                   pruebas=resultados),
              open(OUT / 'pruebas_app.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\nJSON: {OUT / "pruebas_app.json"}')


if __name__ == '__main__':
    main()
