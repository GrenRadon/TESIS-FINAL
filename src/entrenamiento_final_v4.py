"""Modelo final de produccion sobre el dataset v4 (97 casos de desarrollo).

Configuracion: la MISMA de la variante base_DenseNet121, que es la que produjo las cifras
primarias reportadas (AUC 0,8511 · Sens 0,9508 · Espec 0,5278 al umbral 0,1903).
Se mantiene el volteo horizontal aunque la ablacion M-16 demostro que es anatomicamente
invalido y que quitarlo no cuesta nada: cambiar la configuracion aqui haria que el modelo
desplegado no correspondiera a las metricas publicadas. La retirada del volteo queda
documentada como mejora para una version siguiente.

Sin fold reservado: la validacion ya se hizo en el K-Fold. Al no haber validacion no se usa
EarlyStopping; las epocas son las mismas preespecificadas de la corrida v4.
"""
import os, sys, csv, json, time, platform, warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocesamiento as PP
from entrenamiento_v4 import FoldGen, construir, CFG, EPOCAS, SEED

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

V4 = Path('resultados_v4')
OUT = Path('resultados_final_v4'); OUT.mkdir(exist_ok=True)
DIR = Path('models_final_v4'); DIR.mkdir(exist_ok=True)
ESTADO = OUT / 'estado.json'
_t0 = time.time()


def estado(**kw):
    b = {}
    if ESTADO.exists():
        try:
            b = json.loads(ESTADO.read_text(encoding='utf-8'))
        except Exception:
            b = {}
    b.update(kw)
    b['actualizado'] = time.strftime('%H:%M:%S')
    b['transcurrido_min'] = round((time.time() - _t0) / 60, 1)
    ESTADO.write_text(json.dumps(b, indent=2, ensure_ascii=False), encoding='utf-8')


class Progreso(keras.callbacks.Callback):
    def __init__(self, fase, total):
        super().__init__()
        self.fase, self.total = fase, total

    def on_epoch_end(self, epoch, logs=None):
        estado(fase=self.fase, epoca=epoch + 1, epocas_total=self.total,
               loss=round(float((logs or {}).get('loss', 0)), 4),
               auc=round(float((logs or {}).get('auc', 0)), 4))


def umbral_de(archivo, smin=0.95):
    """Umbral propio de una variante: max especificidad sujeta a Sens >= smin, sobre su OOF."""
    from collections import defaultdict
    from sklearn.metrics import roc_curve
    a = defaultdict(lambda: {'p': [], 'y': None})
    for r in csv.DictReader(open(archivo, encoding='utf-8')):
        d = a[r['clave_agrupacion']]
        d['p'].append(float(r['prob']))
        d['y'] = int(r['label'])
    cl = sorted(a)
    y = np.array([a[c]['y'] for c in cl])
    p = np.array([float(np.mean(a[c]['p'])) for c in cl])
    fpr, tpr, thr = roc_curve(y, p)
    v = [(t, f) for t, s, f in zip(thr, tpr, fpr) if s >= smin]
    return round(float(min(v, key=lambda x: x[1])[0]), 4)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sin-flip', action='store_true',
                    help='entrena sin volteo horizontal (variante M-16)')
    args = ap.parse_args()

    r4 = json.load(open(V4 / 'resultados_v4.json', encoding='utf-8'))
    arq = r4['ganadora'].replace('base_', '')
    if args.sin_flip:
        variante, flip = 'sin_flip', False
        umbral = umbral_de(V4 / 'oof_sin_flip.csv')
    else:
        variante, flip = 'base_DenseNet121', True
        umbral = r4['umbral']
    ep1, ep2 = EPOCAS[arq]
    sufijo = '_sin_flip' if args.sin_flip else ''
    print(f'Variante     : {variante} (volteo horizontal: {"NO" if not flip else "si"})')

    rows = list(csv.DictReader(open('resultados_v2/dataset_nivel_imagen.csv', encoding='utf-8')))
    dev = [r for r in rows if r['split'] == 'desarrollo']
    X = np.array([r['ruta'] for r in dev]); y = np.array([int(r['label']) for r in dev])
    casos = len({r['clave_agrupacion'] for r in dev})

    print(f'Arquitectura : {arq}')
    print(f'Umbral       : {umbral}')
    print(f'Epocas       : fase1={ep1}  fase2={ep2}  (preespecificadas, sin EarlyStopping)')
    print(f'Entrenamiento: {len(X)} imagenes / {casos} casos '
          f'({int((y==1).sum())} chiari / {int((y==0).sum())} normal)', flush=True)
    estado(fase='iniciando', variante=variante, arquitectura=arq, umbral=umbral, epocas_f1=ep1, epocas_f2=ep2,
           n_imagenes=int(len(X)), n_casos=casos)

    tf.keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    modelo, base = construir(arq)
    gen = FoldGen(X, y, CFG['BATCH_SIZE'], augment=True, shuffle=True, flip=flip)
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}

    def compilar(lr):
        modelo.compile(optimizer=keras.optimizers.Adam(lr), loss='binary_crossentropy',
                       metrics=['accuracy', keras.metrics.AUC(name='auc')])

    compilar(CFG['LR_F1'])
    modelo.fit(gen, epochs=ep1, class_weight=class_weight, verbose=0,
               callbacks=[Progreso('fase 1 (cabeza)', ep1)])
    base.trainable = True
    for lay in base.layers[:-CFG['UNFREEZE_TOP']]:
        lay.trainable = False
    compilar(CFG['LR_F2'])
    modelo.fit(gen, epochs=ep2, class_weight=class_weight, verbose=0,
               callbacks=[Progreso('fase 2 (fine-tuning)', ep2)])

    ruta = DIR / f'chiari_{arq}_final_v4{sufijo}.keras'
    modelo.save(str(ruta))
    mb = ruta.stat().st_size / 1e6
    print(f'\nModelo guardado: {ruta}  ({mb:.1f} MB)', flush=True)

    x = PP.cargar_imagen(X[0])[None, ...].astype(np.float32)
    for _ in range(3):
        modelo.predict(x, verbose=0)
    ts = []
    for _ in range(20):
        t = time.time(); modelo.predict(x, verbose=0); ts.append((time.time() - t) * 1000)
    print(f'Latencia Keras: p50={np.percentile(ts,50):.0f} ms  p95={np.percentile(ts,95):.0f} ms')

    json.dump(dict(arquitectura=arq, umbral=umbral, epocas_f1=ep1, epocas_f2=ep2,
                   n_imagenes=int(len(X)), n_casos=casos, class_weight=class_weight,
                   modelo=str(ruta), tamano_mb=round(mb, 1),
                   latencia_keras_p50_ms=round(float(np.percentile(ts, 50)), 1),
                   latencia_keras_p95_ms=round(float(np.percentile(ts, 95)), 1),
                   pipeline=PP.VERSION, semilla=SEED,
                   variante=variante, volteo_horizontal=flip,
                   entorno=dict(tensorflow=tf.__version__, keras=keras.__version__,
                                plataforma=platform.platform(),
                                gpus=len(tf.config.list_physical_devices('GPU')))),
              open(OUT / f'modelo_final_v4{sufijo}.json', 'w'), indent=2, ensure_ascii=False)
    estado(fase='completado')
    print('Listo.', flush=True)


if __name__ == '__main__':
    main()
