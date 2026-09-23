"""M-04 — comparación de estrategias frente al desbalance de clases.

El hallazgo objeta que se descartaron positivos por submuestreo cuando existían alternativas,
y pide comparar estrategias en vez de justificar la elegida solo con argumentos.

Referencia ya entrenada: `sin_flip` (class_weight balanceado), que es la configuración
DESPLEGADA. Aquí se añaden las dos alternativas que menciona el hallazgo, con todo lo demás
idéntico: misma partición, mismas épocas, misma semilla, sin volteo.

  focal_loss      : focal loss binaria (alpha=0,25 · gamma=2), sin class_weight
  batch_balanceado: muestreo que fuerza ~50/50 por lote, sin class_weight

Comparación DESCRIPTIVA, como M-12: no se selecciona la mejor. Elegir sobre el mismo OOF con
el que se estima el rendimiento incurriria en M-05.
"""
import os, sys, csv, json, time, warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocesamiento as PP
from entrenamiento_v4 import FoldGen, construir, CFG, EPOCAS, SEED, cargar_imagen, augmentar

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

OUT = Path('resultados_v4')
ESTADO = OUT / 'estado_m04.json'
ARQ = 'DenseNet121'
ALPHA, GAMMA = 0.25, 2.0
VARIANTES = ['focal_loss', 'batch_balanceado']
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
    b['total_folds'] = len(VARIANTES) * CFG['K_FOLDS']
    ESTADO.write_text(json.dumps(b, indent=2, ensure_ascii=False), encoding='utf-8')


def focal_loss(alpha=ALPHA, gamma=GAMMA):
    def fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        p = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), p, 1 - p)
        a = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return tf.reduce_mean(-a * tf.pow(1 - pt, gamma) * tf.math.log(pt))
    fn.__name__ = 'focal_loss'
    return fn


class GenBalanceado(FoldGen):
    """Cada lote se compone con ~50 % de cada clase, remuestreando la minoritaria."""

    def __init__(self, paths, labels, bs, **kw):
        super().__init__(paths, labels, bs, **kw)
        self.pos = np.where(self.labels == 1)[0]
        self.neg = np.where(self.labels == 0)[0]
        self.rng = np.random.default_rng(SEED)
        self._componer()

    def _componer(self):
        n = len(self.labels)
        mitad = n // 2
        self.orden = np.empty(n, dtype=int)
        p = self.rng.choice(self.pos, size=mitad, replace=len(self.pos) < mitad)
        q = self.rng.choice(self.neg, size=n - mitad, replace=len(self.neg) < n - mitad)
        self.orden[0::2] = np.resize(p, len(self.orden[0::2]))
        self.orden[1::2] = np.resize(q, len(self.orden[1::2]))

    def __getitem__(self, i):
        bi = self.orden[i * self.bs:(i + 1) * self.bs]
        X = np.array([augmentar(cargar_imagen(self.paths[j], self.clahe), self.flip)
                      if self.augment else cargar_imagen(self.paths[j], self.clahe)
                      for j in bi], dtype=np.float32)
        return X, self.labels[bi].astype(np.float32)

    def on_epoch_end(self):
        self._componer()


def entrenar_fold(variante, X_tr, y_tr, X_val, y_val, fold):
    tf.keras.backend.clear_session()
    keras.utils.set_random_seed(SEED + fold)
    modelo, base = construir(ARQ)
    ep1, ep2 = EPOCAS[ARQ]

    if variante == 'batch_balanceado':
        gen_tr = GenBalanceado(X_tr, y_tr, CFG['BATCH_SIZE'], augment=True, shuffle=True, flip=False)
        perdida, cw = 'binary_crossentropy', None
    elif variante == 'focal_loss':
        gen_tr = FoldGen(X_tr, y_tr, CFG['BATCH_SIZE'], augment=True, shuffle=True, flip=False)
        perdida, cw = focal_loss(), None
    else:
        raise ValueError(variante)
    gen_val = FoldGen(X_val, y_val, CFG['BATCH_SIZE'], augment=False, shuffle=False, flip=False)

    def compilar(lr):
        modelo.compile(optimizer=keras.optimizers.Adam(lr), loss=perdida,
                       metrics=['accuracy', keras.metrics.AUC(name='auc')])

    cbs = [ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-8, verbose=0)]
    t0 = time.time()
    compilar(CFG['LR_F1'])
    modelo.fit(gen_tr, epochs=ep1, callbacks=cbs, class_weight=cw, verbose=0)
    base.trainable = True
    for lay in base.layers[:-CFG['UNFREEZE_TOP']]:
        lay.trainable = False
    compilar(CFG['LR_F2'])
    modelo.fit(gen_tr, epochs=ep2, callbacks=cbs, class_weight=cw, verbose=0)

    yp = modelo.predict(gen_val, verbose=0).flatten()
    auc = roc_auc_score(y_val, yp) if len(set(y_val.tolist())) > 1 else float('nan')
    dt = time.time() - t0
    print(f'    [{variante}] fold {fold}: n_val={len(X_val)} AUC_val={auc:.4f} ({dt/60:.1f} min)',
          flush=True)
    return yp, dict(variante=variante, fold=fold, n_train=len(X_tr), n_val=len(X_val),
                    auc_val=None if np.isnan(auc) else round(float(auc), 4),
                    minutos=round(dt / 60, 1))


def main():
    rows = list(csv.DictReader(open('resultados_v2/dataset_nivel_imagen.csv', encoding='utf-8')))
    dev = [r for r in rows if r['split'] == 'desarrollo']
    Xd = np.array([r['ruta'] for r in dev]); yd = np.array([int(r['label']) for r in dev])
    gd = np.array([r['clave_agrupacion'] for r in dev])
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=yd)

    print('=' * 74)
    print('  M-04 — estrategias frente al desbalance de clases')
    print(f'  referencia YA entrenada: sin_flip (class_weight {cw[0]:.2f}/{cw[1]:.2f})')
    print(f'  se añaden: focal loss (alpha={ALPHA}, gamma={GAMMA}) y muestreo balanceado por lote')
    print(f'  {len(Xd)} img / {len(set(gd))} casos · comparación DESCRIPTIVA, no se elige la mejor')
    print('=' * 74, flush=True)
    estado(fase='iniciando', variantes=VARIANTES, folds_hechos=0, folds=[])

    sgkf = StratifiedGroupKFold(n_splits=CFG['K_FOLDS'], shuffle=True, random_state=SEED)
    particion = list(sgkf.split(Xd, yd, groups=gd))
    todos, hechos = [], 0

    for variante in VARIANTES:
        print(f'\n{"=" * 24} {variante} {"=" * 24}', flush=True)
        estado(fase=f'entrenando {variante}', variante_actual=variante)
        oof = np.full(len(Xd), np.nan); fold_de = np.full(len(Xd), -1)
        for fold, (tr, va) in enumerate(particion, start=1):
            assert not (set(gd[tr]) & set(gd[va])), f'FUGA en {variante} fold {fold}'
            yp, info = entrenar_fold(variante, Xd[tr], yd[tr], Xd[va], yd[va], fold)
            oof[va] = yp; fold_de[va] = fold
            todos.append(info); hechos += 1
            estado(folds_hechos=hechos, folds=todos)
        assert not np.isnan(oof).any()
        with open(OUT / f'oof_{variante}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['archivo', 'ruta', 'clase', 'label', 'clave_agrupacion', 'rID', 'fold', 'prob'])
            for r, f, p in zip(dev, fold_de, oof):
                w.writerow([r['archivo'], r['ruta'], r['clase'], r['label'],
                            r['clave_agrupacion'], r['rID'], int(f), round(float(p), 6)])

    json.dump(dict(referencia='sin_flip (class_weight)', variantes=VARIANTES,
                   alpha=ALPHA, gamma=GAMMA, class_weight=[float(cw[0]), float(cw[1])],
                   folds=todos, semilla=SEED,
                   nota='Comparacion descriptiva. No se selecciona la mejor estrategia: hacerlo '
                        'sobre el mismo OOF con el que se estima el rendimiento incurriria en M-05.'),
              open(OUT / 'ablacion_desbalance.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    estado(fase='completado', folds_hechos=hechos)
    print('\nAblación M-04 completada.', flush=True)


if __name__ == '__main__':
    main()
