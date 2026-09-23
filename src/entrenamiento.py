"""Paso 4-5: K-Fold agrupado por caso sobre desarrollo + evaluacion OOF y test reservado.

Replica nb_03: mismo cargar_imagen (CLAHE -> resize -> /255), misma DenseNet121,
mismos hiperparametros (CFG). Unicos cambios: particion agrupada por clave_agrupacion,
class_weight en vez de submuestreo 30/30, y prediccion out-of-fold en vez de 'mejor fold'.
"""
import os, sys, csv, json, time, warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

OUT = Path('resultados_v2'); OUT.mkdir(exist_ok=True)
DIR_MODELS = Path('models_v2'); DIR_MODELS.mkdir(exist_ok=True)

CFG = {  # identico a nb_03
    'IMG_SHAPE': (224, 224, 3), 'BATCH_SIZE': 8, 'K_FOLDS': 5,
    'EPOCHS_F1': 40, 'LR_F1': 1e-3, 'EPOCHS_F2': 30, 'LR_F2': 1e-6,
    'UNFREEZE_TOP': 10, 'PATIENCE': 15, 'SEED': SEED,
}


def cargar_dataset():
    rows = list(csv.DictReader(open(OUT / 'dataset_nivel_imagen.csv', encoding='utf-8')))
    dev = [r for r in rows if r['split'] == 'desarrollo']
    tst = [r for r in rows if r['split'] == 'test']
    return dev, tst


def cargar_imagen(ruta):  # identico a nb_03
    arr = np.array(Image.open(ruta).convert('L'), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)
    arr_f = np.array(Image.fromarray(arr).resize(CFG['IMG_SHAPE'][:2], Image.LANCZOS),
                     dtype=np.float32) / 255.0
    return np.stack([arr_f, arr_f, arr_f], axis=-1)


def augmentar(img):  # identico a nb_03
    if np.random.rand() > 0.5:
        img = np.fliplr(img)
    if np.random.rand() > 0.5:
        img = np.clip(img * np.random.uniform(0.90, 1.10), 0, 1)
    if np.random.rand() > 0.6:
        img = np.clip(img + np.random.normal(0, 0.008, img.shape).astype(np.float32), 0, 1)
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
        X = np.array([augmentar(cargar_imagen(self.paths[j])) if self.augment
                      else cargar_imagen(self.paths[j]) for j in bi], dtype=np.float32)
        return X, self.labels[bi].astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)


def construir_modelo():  # identico a nb_03 (DenseNet121)
    base = DenseNet121(include_top=False, weights='imagenet', input_shape=CFG['IMG_SHAPE'])
    base.trainable = False
    inp = keras.Input(shape=CFG['IMG_SHAPE'])
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name='DenseNet121'), base


def entrenar_fold(X_tr, y_tr, X_val, y_val, fold_n):
    tf.keras.backend.clear_session()
    keras.utils.set_random_seed(SEED + fold_n)
    modelo, base = construir_modelo()
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
    for layer in base.layers[:-CFG['UNFREEZE_TOP']]:
        layer.trainable = False
    compilar(CFG['LR_F2'])
    h2 = modelo.fit(gen_tr, validation_data=gen_val, epochs=CFG['EPOCHS_F2'],
                    callbacks=cbs, class_weight=class_weight, verbose=0)

    y_prob = modelo.predict(gen_val, verbose=0).flatten()
    dt = time.time() - t0
    auc = roc_auc_score(y_val, y_prob) if len(set(y_val.tolist())) > 1 else float('nan')
    print(f'    fold {fold_n}: n_tr={len(X_tr)} n_val={len(X_val)} '
          f'cw={class_weight[0]:.2f}/{class_weight[1]:.2f} '
          f'ep={len(h1.history["loss"])}+{len(h2.history["loss"])} '
          f'AUC_val={auc:.4f}  ({dt / 60:.1f} min)', flush=True)
    return modelo, y_prob, dict(fold=fold_n, n_train=len(X_tr), n_val=len(X_val),
                                epocas_f1=len(h1.history['loss']), epocas_f2=len(h2.history['loss']),
                                class_weight=class_weight,
                                auc_val=None if np.isnan(auc) else round(float(auc), 4),
                                minutos=round(dt / 60, 1))


def main():
    dev, tst = cargar_dataset()
    Xd = np.array([r['ruta'] for r in dev]); yd = np.array([int(r['label']) for r in dev])
    gd = np.array([r['clave_agrupacion'] for r in dev])
    Xt = np.array([r['ruta'] for r in tst]); yt = np.array([int(r['label']) for r in tst])

    print('=' * 70)
    print(f'  DESARROLLO: {len(Xd)} img ({int((yd == 1).sum())} chiari / {int((yd == 0).sum())} normal) '
          f'| {len(set(gd))} casos')
    print(f'  TEST       : {len(Xt)} img ({int((yt == 1).sum())} chiari / {int((yt == 0).sum())} normal)')
    print('=' * 70, flush=True)

    sgkf = StratifiedGroupKFold(n_splits=CFG['K_FOLDS'], shuffle=True, random_state=SEED)
    oof = np.full(len(Xd), np.nan)
    fold_de = np.full(len(Xd), -1)
    test_probs, folds_info = [], []

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(Xd, yd, groups=gd), start=1):
        assert not (set(gd[tr_idx]) & set(gd[val_idx])), f'FUGA en fold {fold}'
        modelo, y_prob, info = entrenar_fold(Xd[tr_idx], yd[tr_idx], Xd[val_idx], yd[val_idx], fold)
        oof[val_idx] = y_prob
        fold_de[val_idx] = fold
        gen_t = FoldGen(Xt, yt, CFG['BATCH_SIZE'], augment=False, shuffle=False)
        test_probs.append(modelo.predict(gen_t, verbose=0).flatten())
        modelo.save(str(DIR_MODELS / f'densenet121_v2_fold{fold}.keras'))
        folds_info.append(info)

    assert not np.isnan(oof).any(), 'quedaron imagenes sin prediccion OOF'
    test_ens = np.mean(np.vstack(test_probs), axis=0)

    with open(OUT / 'predicciones_oof.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['archivo', 'ruta', 'clase', 'label', 'clave_agrupacion', 'rID', 'fold', 'prob'])
        for r, f, p in zip(dev, fold_de, oof):
            w.writerow([r['archivo'], r['ruta'], r['clase'], r['label'],
                        r['clave_agrupacion'], r['rID'], int(f), round(float(p), 6)])

    with open(OUT / 'predicciones_test.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['archivo', 'ruta', 'clase', 'label', 'clave_agrupacion', 'rID', 'prob_ensamble']
                   + [f'prob_fold{i}' for i in range(1, 6)])
        for i, r in enumerate(tst):
            w.writerow([r['archivo'], r['ruta'], r['clase'], r['label'], r['clave_agrupacion'],
                        r['rID'], round(float(test_ens[i]), 6)]
                       + [round(float(tp[i]), 6) for tp in test_probs])

    json.dump({'cfg': CFG, 'folds': folds_info}, open(OUT / 'folds_v2.json', 'w'), indent=2)
    print('\nEntrenamiento completado. Predicciones guardadas.', flush=True)


if __name__ == '__main__':
    main()
