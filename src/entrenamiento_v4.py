"""v4 — corrida definitiva. Opcion A (epocas fijas) + controles y ablaciones.

VARIANTES (cada una es un K-Fold agrupado completo sobre el mismo desarrollo):
  base_<arq>   : epocas fijas, SIN EarlyStopping           -> M-05 parcial, M-06, M-11
  control_es   : idem pero CON EarlyStopping               -> cuantifica el sesgo del ES
  sin_flip     : sin volteo horizontal                     -> M-16
  sin_clahe    : sin CLAHE en el preprocesamiento          -> M-15

Por que epocas fijas: con EarlyStopping, la epoca se elige maximizando val_auc sobre el
mismo fold cuya prediccion luego se reporta como OOF. Eso sesga el OOF al alza (M-05).
Fijandolas de antemano, el fold de validacion no interviene en el entrenamiento.
"""
import os, sys, csv, json, time, platform, warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
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

OUT = Path('resultados_v4'); OUT.mkdir(exist_ok=True)
MODELOS = Path('models_v4'); MODELOS.mkdir(exist_ok=True)
ESTADO = OUT / 'estado.json'

BASES = {'DenseNet121': DenseNet121, 'EfficientNetB0': EfficientNetB0, 'ResNet50V2': ResNet50V2}
CFG = {'IMG_SHAPE': (224, 224, 3), 'BATCH_SIZE': 8, 'K_FOLDS': 5,
       'LR_F1': 1e-3, 'LR_F2': 1e-6, 'UNFREEZE_TOP': 10, 'SEED': SEED}
# Epocas fijadas a priori con la mediana observada en la corrida v3 (una corrida previa
# sobre datos equivalentes). Se declara asi en Metodologia.
EPOCAS = {'DenseNet121': (40, 15), 'EfficientNetB0': (30, 15), 'ResNet50V2': (24, 15)}

VARIANTES = [
    ('base_DenseNet121',    dict(arq='DenseNet121',    es=False, flip=True,  clahe=True)),
    ('base_EfficientNetB0', dict(arq='EfficientNetB0', es=False, flip=True,  clahe=True)),
    ('base_ResNet50V2',     dict(arq='ResNet50V2',     es=False, flip=True,  clahe=True)),
    ('control_es',          dict(arq='DenseNet121',    es=True,  flip=True,  clahe=True)),
    ('sin_flip',            dict(arq='DenseNet121',    es=False, flip=False, clahe=True)),
    ('sin_clahe',           dict(arq='DenseNet121',    es=False, flip=True,  clahe=False)),
]

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


def cargar_imagen(ruta, clahe=True):
    """Igual que PP.cargar_imagen, con CLAHE conmutable para la ablacion M-15."""
    if clahe:
        return PP.cargar_imagen(ruta)
    arr = np.array(Image.open(ruta).convert('L'), dtype=np.uint8)
    arr = np.array(Image.fromarray(arr).resize(PP.IMG_SIZE, Image.LANCZOS), dtype=np.float32)
    return np.stack([arr, arr, arr], axis=-1)


def augmentar(img, flip=True):
    # OJO: en cortes sagitales fliplr invierte el eje antero-posterior (M-16).
    if flip and np.random.rand() > 0.5:
        img = np.fliplr(img)
    if np.random.rand() > 0.5:
        img = np.clip(img * np.random.uniform(0.90, 1.10), 0, 255)
    if np.random.rand() > 0.6:
        img = np.clip(img + np.random.normal(0, 0.008 * 255, img.shape).astype(np.float32), 0, 255)
    return img


class FoldGen(keras.utils.Sequence):
    def __init__(self, paths, labels, bs, augment=False, shuffle=True, flip=True, clahe=True):
        super().__init__()
        self.paths = np.array(paths); self.labels = np.array(labels)
        self.bs = bs; self.augment = augment; self.shuffle = shuffle
        self.flip = flip; self.clahe = clahe
        self.idx = np.arange(len(self.paths))
        if shuffle:
            np.random.shuffle(self.idx)

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.bs))

    def __getitem__(self, i):
        bi = self.idx[i * self.bs:(i + 1) * self.bs]
        X = np.array([augmentar(cargar_imagen(self.paths[j], self.clahe), self.flip)
                      if self.augment else cargar_imagen(self.paths[j], self.clahe)
                      for j in bi], dtype=np.float32)
        return X, self.labels[bi].astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)


def construir(arq):
    base = BASES[arq](include_top=False, weights='imagenet', input_shape=CFG['IMG_SHAPE'])
    base.trainable = False
    inp = keras.Input(shape=CFG['IMG_SHAPE'], name='entrada_0_255')
    x = inp
    for cap in PP.capas_preprocess(arq):
        x = cap(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.5)(x)
    return Model(inp, layers.Dense(1, activation='sigmoid')(x), name=arq), base


def entrenar_fold(cfgv, X_tr, y_tr, X_val, y_val, fold, nombre):
    tf.keras.backend.clear_session()
    keras.utils.set_random_seed(SEED + fold)
    arq = cfgv['arq']
    modelo, base = construir(arq)
    ep1, ep2 = EPOCAS[arq]
    gen_tr = FoldGen(X_tr, y_tr, CFG['BATCH_SIZE'], augment=True, shuffle=True,
                     flip=cfgv['flip'], clahe=cfgv['clahe'])
    gen_val = FoldGen(X_val, y_val, CFG['BATCH_SIZE'], augment=False, shuffle=False,
                      flip=cfgv['flip'], clahe=cfgv['clahe'])
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=np.array(y_tr))
    class_weight = {0: float(cw[0]), 1: float(cw[1])}

    cbs = [ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-8, verbose=0)]
    if cfgv['es']:
        cbs = [EarlyStopping(monitor='val_auc', patience=15, mode='max',
                             restore_best_weights=True, verbose=0),
               ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-8, verbose=0)]

    def compilar(lr):
        modelo.compile(optimizer=keras.optimizers.Adam(lr), loss='binary_crossentropy',
                       metrics=['accuracy', keras.metrics.AUC(name='auc')])

    t0 = time.time()
    compilar(CFG['LR_F1'])
    h1 = modelo.fit(gen_tr, validation_data=gen_val, epochs=ep1, callbacks=cbs,
                    class_weight=class_weight, verbose=0)
    base.trainable = True
    for lay in base.layers[:-CFG['UNFREEZE_TOP']]:
        lay.trainable = False
    compilar(CFG['LR_F2'])
    h2 = modelo.fit(gen_tr, validation_data=gen_val, epochs=ep2, callbacks=cbs,
                    class_weight=class_weight, verbose=0)

    y_prob = modelo.predict(gen_val, verbose=0).flatten()
    auc = roc_auc_score(y_val, y_prob) if len(set(y_val.tolist())) > 1 else float('nan')
    dt = time.time() - t0
    print(f'    [{nombre}] fold {fold}: n_tr={len(X_tr)} n_val={len(X_val)} '
          f'ep={len(h1.history["loss"])}+{len(h2.history["loss"])} '
          f'AUC_val={auc:.4f} ({dt/60:.1f} min)', flush=True)
    # M-06: curvas por fold
    curvas = {k: [float(x) for x in v] for k, v in
              {**{f'f1_{k}': v for k, v in h1.history.items()},
               **{f'f2_{k}': v for k, v in h2.history.items()}}.items()}
    info = dict(variante=nombre, arq=arq, fold=fold, n_train=len(X_tr), n_val=len(X_val),
                epocas_f1=len(h1.history['loss']), epocas_f2=len(h2.history['loss']),
                class_weight=class_weight,
                auc_val=None if np.isnan(auc) else round(float(auc), 4),
                minutos=round(dt / 60, 1), curvas=curvas)
    if cfgv['es']:
        va = h1.history.get('val_auc', [])
        info['mejor_epoca_f1'] = int(np.argmax(va)) + 1 if va else None
    return modelo, y_prob, info


def main():
    rows = list(csv.DictReader(open('resultados_v2/dataset_nivel_imagen.csv', encoding='utf-8')))
    dev = [r for r in rows if r['split'] == 'desarrollo']
    tst = [r for r in rows if r['split'] == 'test']
    Xd = np.array([r['ruta'] for r in dev]); yd = np.array([int(r['label']) for r in dev])
    gd = np.array([r['clave_agrupacion'] for r in dev])
    Xt = np.array([r['ruta'] for r in tst]); yt = np.array([int(r['label']) for r in tst])

    entorno = dict(python=platform.python_version(), tensorflow=tf.__version__,
                   keras=keras.__version__, plataforma=platform.platform(),
                   procesador=platform.processor(), gpus=len(tf.config.list_physical_devices('GPU')),
                   pipeline=PP.VERSION, semilla=SEED)
    print('=' * 78)
    print(f'  DESARROLLO {len(Xd)} img / {len(set(gd))} casos '
          f'({int((yd==1).sum())} chiari / {int((yd==0).sum())} normal)   TEST {len(Xt)} img')
    print(f'  {entorno["tensorflow"]} / keras {entorno["keras"]} / GPUs={entorno["gpus"]}')
    print('=' * 78, flush=True)
    estado(fase='iniciando', variantes=[v for v, _ in VARIANTES], folds_hechos=0,
           folds=[], entorno=entorno)

    sgkf = StratifiedGroupKFold(n_splits=CFG['K_FOLDS'], shuffle=True, random_state=SEED)
    particion = list(sgkf.split(Xd, yd, groups=gd))
    todos, hechos = [], 0

    for nombre, cfgv in VARIANTES:
        print(f'\n{"="*26} {nombre} {"="*26}', flush=True)
        estado(fase=f'entrenando {nombre}', variante_actual=nombre)
        oof = np.full(len(Xd), np.nan); fold_de = np.full(len(Xd), -1)
        tprobs = []
        for fold, (tr, va) in enumerate(particion, start=1):
            assert not (set(gd[tr]) & set(gd[va])), f'FUGA en {nombre} fold {fold}'
            modelo, yp, info = entrenar_fold(cfgv, Xd[tr], yd[tr], Xd[va], yd[va], fold, nombre)
            oof[va] = yp; fold_de[va] = fold
            gen_t = FoldGen(Xt, yt, CFG['BATCH_SIZE'], augment=False, shuffle=False,
                            flip=cfgv['flip'], clahe=cfgv['clahe'])
            tprobs.append(modelo.predict(gen_t, verbose=0).flatten())
            if nombre.startswith('base_'):
                modelo.save(str(MODELOS / f'{nombre}_fold{fold}.keras'))
            todos.append(info); hechos += 1
            estado(folds_hechos=hechos, folds=[{k: v for k, v in f.items() if k != 'curvas'}
                                               for f in todos])
        assert not np.isnan(oof).any()
        ens = np.mean(np.vstack(tprobs), axis=0)
        with open(OUT / f'oof_{nombre}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['archivo', 'ruta', 'clase', 'label', 'clave_agrupacion', 'rID', 'fold', 'prob'])
            for r, f, p in zip(dev, fold_de, oof):
                w.writerow([r['archivo'], r['ruta'], r['clase'], r['label'],
                            r['clave_agrupacion'], r['rID'], int(f), round(float(p), 6)])
        with open(OUT / f'test_{nombre}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['archivo', 'ruta', 'clase', 'label', 'clave_agrupacion', 'rID', 'prob_ensamble'])
            for i, r in enumerate(tst):
                w.writerow([r['archivo'], r['ruta'], r['clase'], r['label'],
                            r['clave_agrupacion'], r['rID'], round(float(ens[i]), 6)])

    json.dump(dict(cfg=CFG, epocas=EPOCAS, entorno=entorno,
                   variantes={n: c for n, c in VARIANTES}, folds=todos),
              open(OUT / 'folds_v4.json', 'w'), indent=2, ensure_ascii=False)
    estado(fase='completado', folds_hechos=hechos)
    print('\nCorrida v4 completada.', flush=True)


if __name__ == '__main__':
    main()
