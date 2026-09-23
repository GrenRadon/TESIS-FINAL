"""Punto 5: modelo final de produccion, entrenado sobre TODO el desarrollo (81 casos).

Sin fold reservado: la validacion ya se hizo en el K-Fold. Al no haber conjunto de
validacion no puede usarse EarlyStopping, asi que el numero de epocas se fija de antemano
con la MEDIANA de epocas efectivas de los folds de la arquitectura ganadora (v3).
"""
import os, sys, csv, json, time, warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocesamiento as PP
from entrenamiento_v3 import FoldGen, construir_modelo, CFG, SEED

V3 = Path('resultados_v3')
OUT = Path('resultados_final'); OUT.mkdir(exist_ok=True)
DIR = Path('models_final'); DIR.mkdir(exist_ok=True)
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


def main():
    r3 = json.load(open(V3 / 'resultados_v3.json', encoding='utf-8'))
    arq, umbral = r3['ganadora'], r3['umbral_final']
    folds = json.load(open(V3 / 'folds_v3.json', encoding='utf-8'))['folds']
    mios = [f for f in folds if f['arq'] == arq]
    ep1 = int(np.median([f['epocas_f1'] for f in mios]))
    ep2 = int(np.median([f['epocas_f2'] for f in mios]))

    rows = list(csv.DictReader(open('resultados_v2/dataset_nivel_imagen.csv', encoding='utf-8')))
    dev = [r for r in rows if r['split'] == 'desarrollo']
    X = np.array([r['ruta'] for r in dev]); y = np.array([int(r['label']) for r in dev])

    print(f'Arquitectura : {arq}')
    print(f'Umbral       : {umbral}')
    print(f'Epocas       : fase1={ep1}  fase2={ep2}  (mediana de los 5 folds)')
    print(f'Entrenamiento: {len(X)} imagenes, {len(set(r["clave_agrupacion"] for r in dev))} casos')
    estado(fase='iniciando', arquitectura=arq, umbral=umbral, epocas_f1=ep1, epocas_f2=ep2)

    tf.keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    modelo, base = construir_modelo(arq)
    gen = FoldGen(X, y, CFG['BATCH_SIZE'], augment=True, shuffle=True)
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

    ruta = DIR / f'chiari_{arq}_final.keras'
    modelo.save(str(ruta))
    mb = ruta.stat().st_size / 1e6
    print(f'\nModelo guardado: {ruta}  ({mb:.1f} MB)')

    # latencia del modelo Keras (referencia; la de ONNX se mide en la exportacion)
    x = PP.cargar_imagen(X[0])[None, ...].astype(np.float32)
    for _ in range(3):
        modelo.predict(x, verbose=0)
    ts = []
    for _ in range(20):
        t = time.time(); modelo.predict(x, verbose=0); ts.append((time.time() - t) * 1000)
    print(f'Latencia Keras: p50={np.percentile(ts, 50):.0f} ms  p95={np.percentile(ts, 95):.0f} ms')

    json.dump(dict(arquitectura=arq, umbral=umbral, epocas_f1=ep1, epocas_f2=ep2,
                   n_imagenes=len(X), class_weight=class_weight,
                   modelo=str(ruta), tamano_mb=round(mb, 1),
                   latencia_keras_p50_ms=round(float(np.percentile(ts, 50)), 1),
                   latencia_keras_p95_ms=round(float(np.percentile(ts, 95)), 1),
                   pipeline=PP.VERSION),
              open(OUT / 'modelo_final.json', 'w'), indent=2, ensure_ascii=False)
    estado(fase='completado')
    print('Listo.')


if __name__ == '__main__':
    main()
