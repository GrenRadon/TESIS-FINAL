"""v3 — K-Fold agrupado para las 3 arquitecturas con el pipeline unico (C-06/C-08 + M-11).

Diferencias respecto a v2:
  - preprocesamiento importado de src/preprocesamiento.py (fuente unica)
  - preprocess_input especifico de cada arquitectura, DENTRO del modelo
  - se entrenan DenseNet121, EfficientNetB0 y ResNet50V2 bajo el mismo protocolo
Todo lo demas identico: misma particion limpia, mismos hiperparametros, OOF + ensamble.
"""
import os, sys, csv, json, time, warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121, EfficientNetB0, ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocesamiento as PP

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

OUT = Path('resultados_v3'); OUT.mkdir(exist_ok=True)
DIR_MODELS = Path('models_v3'); DIR_MODELS.mkdir(exist_ok=True)
ESTADO = OUT / 'estado.json'

ARQS = ['DenseNet121', 'EfficientNetB0', 'ResNet50V2']
BASES = {'DenseNet121': DenseNet121, 'EfficientNetB0': EfficientNetB0, 'ResNet50V2': ResNet50V2}
CFG = {'IMG_SHAPE': (224, 224, 3), 'BATCH_SIZE': 8, 'K_FOLDS': 5,
       'EPOCHS_F1': 40, 'LR_F1': 1e-3, 'EPOCHS_F2': 30, 'LR_F2': 1e-6,
       'UNFREEZE_TOP': 10, 'PATIENCE': 15, 'SEED': SEED}

_t0 = time.time()


def estado(**kw):
    """Escribe el avance para que el panel lo lea."""
    base = {}
    if ESTADO.exists():
        try:
            base = json.loads(ESTADO.read_text(encoding='utf-8'))
        except Exception:
            base = {}
    base.update(kw)
    base['actualizado'] = time.strftime('%H:%M:%S')
    base['transcurrido_min'] = round((time.time() - _t0) / 60, 1)
    base['total_folds'] = len(ARQS) * CFG['K_FOLDS']
    ESTADO.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding='utf-8')


def augmentar(img):
    """Equivalente a nb_03 pero sobre rango [0,255] en vez de [0,1]."""
    if np.random.rand() > 0.5:
        img = np.fliplr(img)
    if np.random.rand() > 0.5:
        img = np.clip(img * np.random.uniform(0.90, 1.10), 0, 255)
    if np.random.rand() > 0.6:
        img = np.clip(img + np.random.normal(0, 0.008 * 255, img.shape).astype(np.float32), 0, 255)
    return img


class FoldGen(keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size, augment=False, shuffle=True):
        super().__init__()
        self.paths = np.array(paths); self.labels = np.array(labels)
        self.bs = batch_size; self.augment = augment; self.shuffle = shuffle
        self.idx = np.arange(len(self.paths))
        if shuffle:
            np.random.shuffle(self.idx)

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.bs))

    def __getitem__(self, i):
        bi = self.idx[i * self.bs:(i + 1) * self.bs]
        X = np.array([augmentar(PP.cargar_imagen(self.paths[j])) if self.augment
                      else PP.cargar_imagen(self.paths[j]) for j in bi], dtype=np.float32)
        return X, self.labels[bi].astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)


def construir_modelo(arq):
    base = BASES[arq](include_top=False, weights='imagenet', input_shape=CFG['IMG_SHAPE'])
    base.trainable = False
    inp = keras.Input(shape=CFG['IMG_SHAPE'], name='entrada_0_255')
    x = inp
    for cap in PP.capas_preprocess(arq):      # preprocess_input dentro del modelo
        x = cap(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name=arq), base


def entrenar_fold(arq, X_tr, y_tr, X_val, y_val, fold_n):
    tf.keras.backend.clear_session()
    keras.utils.set_random_seed(SEED + fold_n)
    modelo, base = construir_modelo(arq)
    gen_tr = FoldGen(X_tr, y_tr, CFG['BATCH_SIZE'], augment=True, shuffle=True)
    gen_val = FoldGen(X_val, y_val, CFG['BATCH_SIZE'], augment=False, shuffle=False)
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=np.array(y_tr))
    class_weight = {0: float(cw[0]), 1: float(cw[1])}
    cbs = [EarlyStopping(monitor='val_auc', patience=CFG['PATIENCE'], mode='max',
                         restore_best_weights=True, verbose=0),
           ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-8, verbose=0)]

    def compilar(lr):
        modelo.compile(optimizer=keras.optimizers.Adam(lr), loss='binary_crossentropy',
                       metrics=['accuracy', keras.metrics.AUC(name='auc')])

    t0 = time.time()
    compilar(CFG['LR_F1'])
    h1 = modelo.fit(gen_tr, validation_data=gen_val, epochs=CFG['EPOCHS_F1'],
                    callbacks=cbs, class_weight=class_weight, verbose=0)
    base.trainable = True
    for lay in base.layers[:-CFG['UNFREEZE_TOP']]:
        lay.trainable = False
    compilar(CFG['LR_F2'])
    h2 = modelo.fit(gen_tr, validation_data=gen_val, epochs=CFG['EPOCHS_F2'],
                    callbacks=cbs, class_weight=class_weight, verbose=0)
    y_prob = modelo.predict(gen_val, verbose=0).flatten()
    dt = time.time() - t0
    auc = roc_auc_score(y_val, y_prob) if len(set(y_val.tolist())) > 1 else float('nan')
    print(f'    [{arq}] fold {fold_n}: n_tr={len(X_tr)} n_val={len(X_val)} '
          f'ep={len(h1.history["loss"])}+{len(h2.history["loss"])} '
          f'AUC_val={auc:.4f}  ({dt / 60:.1f} min)', flush=True)
    return modelo, y_prob, dict(arq=arq, fold=fold_n, n_train=len(X_tr), n_val=len(X_val),
                                epocas_f1=len(h1.history['loss']), epocas_f2=len(h2.history['loss']),
                                class_weight=class_weight,
                                auc_val=None if np.isnan(auc) else round(float(auc), 4),
                                minutos=round(dt / 60, 1))


def main():
    rows = list(csv.DictReader(open('resultados_v2/dataset_nivel_imagen.csv', encoding='utf-8')))
    dev = [r for r in rows if r['split'] == 'desarrollo']
    tst = [r for r in rows if r['split'] == 'test']
    Xd = np.array([r['ruta'] for r in dev]); yd = np.array([int(r['label']) for r in dev])
    gd = np.array([r['clave_agrupacion'] for r in dev])
    Xt = np.array([r['ruta'] for r in tst]); yt = np.array([int(r['label']) for r in tst])

    print('=' * 74)
    print(f'  PIPELINE v{PP.VERSION}')
    for a in ARQS:
        print(f'    {a:<16}: {PP.descripcion(a)[-70:]}')
    print(f'  DESARROLLO: {len(Xd)} img / {len(set(gd))} casos   TEST: {len(Xt)} img')
    print('=' * 74, flush=True)
    estado(fase='iniciando', arquitecturas=ARQS, folds_hechos=0, folds=[],
           dev_img=len(Xd), dev_casos=len(set(gd)), test_img=len(Xt))

    sgkf = StratifiedGroupKFold(n_splits=CFG['K_FOLDS'], shuffle=True, random_state=SEED)
    particion = list(sgkf.split(Xd, yd, groups=gd))
    todos, hechos = [], 0

    for arq in ARQS:
        print(f'\n{"=" * 30} {arq} {"=" * 30}', flush=True)
        estado(fase=f'entrenando {arq}', arq_actual=arq)
        oof = np.full(len(Xd), np.nan); fold_de = np.full(len(Xd), -1)
        test_probs = []
        for fold, (tr_idx, val_idx) in enumerate(particion, start=1):
            assert not (set(gd[tr_idx]) & set(gd[val_idx])), f'FUGA en {arq} fold {fold}'
            modelo, y_prob, info = entrenar_fold(arq, Xd[tr_idx], yd[tr_idx],
                                                 Xd[val_idx], yd[val_idx], fold)
            oof[val_idx] = y_prob; fold_de[val_idx] = fold
            gen_t = FoldGen(Xt, yt, CFG['BATCH_SIZE'], augment=False, shuffle=False)
            test_probs.append(modelo.predict(gen_t, verbose=0).flatten())
            modelo.save(str(DIR_MODELS / f'{arq}_v3_fold{fold}.keras'))
            todos.append(info); hechos += 1
            estado(folds_hechos=hechos, folds=todos, arq_actual=arq)
        assert not np.isnan(oof).any()
        ens = np.mean(np.vstack(test_probs), axis=0)
        with open(OUT / f'oof_{arq}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['archivo', 'ruta', 'clase', 'label', 'clave_agrupacion', 'rID', 'fold', 'prob'])
            for r, f, p in zip(dev, fold_de, oof):
                w.writerow([r['archivo'], r['ruta'], r['clase'], r['label'],
                            r['clave_agrupacion'], r['rID'], int(f), round(float(p), 6)])
        with open(OUT / f'test_{arq}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['archivo', 'ruta', 'clase', 'label', 'clave_agrupacion', 'rID', 'prob_ensamble'])
            for i, r in enumerate(tst):
                w.writerow([r['archivo'], r['ruta'], r['clase'], r['label'],
                            r['clave_agrupacion'], r['rID'], round(float(ens[i]), 6)])

    json.dump({'cfg': CFG, 'pipeline': PP.VERSION, 'folds': todos,
               'descripcion': {a: PP.descripcion(a) for a in ARQS}},
              open(OUT / 'folds_v3.json', 'w'), indent=2, ensure_ascii=False)
    estado(fase='completado', folds_hechos=hechos, folds=todos)
    print('\nEntrenamiento v3 completado.', flush=True)


if __name__ == '__main__':
    main()
