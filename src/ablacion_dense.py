"""M-12 — ablacion DESCRIPTIVA de la capa densa: Dense(16) y Dense(64).

Dense(32) es la configuracion preespecificada y ya esta entrenada en la corrida v4
(variante base_DenseNet121). Aqui se entrenan las otras dos EXCLUSIVAMENTE como
analisis de sensibilidad.

IMPORTANTE — no se selecciona la mejor.
Elegir la mejor de las tres sobre el mismo conjunto de desarrollo con el que luego se
estima el rendimiento seria incurrir en M-05 (seleccionar y estimar sobre los mismos datos).
El modelo sigue siendo Dense(32). Lo que se reporta es si la eleccion importaba o no.
"""
import os, sys, csv, json, time, warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocesamiento as PP
from entrenamiento_v4 import FoldGen, CFG, EPOCAS, SEED

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

OUT = Path('resultados_v4')
ESTADO = OUT / 'estado_m12.json'
UNIDADES = [16, 64]          # Dense(32) ya esta en base_DenseNet121
PREESPECIFICADO = 32
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
    b['total_folds'] = len(UNIDADES) * CFG['K_FOLDS']
    ESTADO.write_text(json.dumps(b, indent=2, ensure_ascii=False), encoding='utf-8')


def construir(unidades):
    base = DenseNet121(include_top=False, weights='imagenet', input_shape=CFG['IMG_SHAPE'])
    base.trainable = False
    inp = keras.Input(shape=CFG['IMG_SHAPE'], name='entrada_0_255')
    x = inp
    for cap in PP.capas_preprocess('DenseNet121'):
        x = cap(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(unidades, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.5)(x)
    return Model(inp, layers.Dense(1, activation='sigmoid')(x)), base


def entrenar_fold(unidades, X_tr, y_tr, X_val, y_val, fold):
    tf.keras.backend.clear_session()
    keras.utils.set_random_seed(SEED + fold)
    modelo, base = construir(unidades)
    ep1, ep2 = EPOCAS['DenseNet121']
    gen_tr = FoldGen(X_tr, y_tr, CFG['BATCH_SIZE'], augment=True, shuffle=True)
    gen_val = FoldGen(X_val, y_val, CFG['BATCH_SIZE'], augment=False, shuffle=False)
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=np.array(y_tr))
    class_weight = {0: float(cw[0]), 1: float(cw[1])}
    cbs = [ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-8, verbose=0)]

    def compilar(lr):
        modelo.compile(optimizer=keras.optimizers.Adam(lr), loss='binary_crossentropy',
                       metrics=['accuracy', keras.metrics.AUC(name='auc')])

    t0 = time.time()
    compilar(CFG['LR_F1'])
    modelo.fit(gen_tr, epochs=ep1, callbacks=cbs, class_weight=class_weight, verbose=0)
    base.trainable = True
    for lay in base.layers[:-CFG['UNFREEZE_TOP']]:
        lay.trainable = False
    compilar(CFG['LR_F2'])
    modelo.fit(gen_tr, epochs=ep2, callbacks=cbs, class_weight=class_weight, verbose=0)

    yp = modelo.predict(gen_val, verbose=0).flatten()
    auc = roc_auc_score(y_val, yp) if len(set(y_val.tolist())) > 1 else float('nan')
    dt = time.time() - t0
    print(f'    [Dense({unidades})] fold {fold}: n_val={len(X_val)} '
          f'AUC_val={auc:.4f} ({dt/60:.1f} min)', flush=True)
    return yp, dict(unidades=unidades, fold=fold, n_train=len(X_tr), n_val=len(X_val),
                    auc_val=None if np.isnan(auc) else round(float(auc), 4),
                    minutos=round(dt / 60, 1))


def main():
    rows = list(csv.DictReader(open('resultados_v2/dataset_nivel_imagen.csv', encoding='utf-8')))
    dev = [r for r in rows if r['split'] == 'desarrollo']
    Xd = np.array([r['ruta'] for r in dev]); yd = np.array([int(r['label']) for r in dev])
    gd = np.array([r['clave_agrupacion'] for r in dev])

    print('=' * 74)
    print(f'  M-12 — ablacion descriptiva de la capa densa')
    print(f'  preespecificado: Dense({PREESPECIFICADO}) (ya entrenado en base_DenseNet121)')
    print(f'  ablacion: {UNIDADES}   ·   {len(Xd)} img / {len(set(gd))} casos')
    print('  NO se selecciona la mejor: el modelo sigue siendo Dense(32)')
    print('=' * 74, flush=True)
    estado(fase='iniciando', unidades=UNIDADES, folds_hechos=0, folds=[])

    sgkf = StratifiedGroupKFold(n_splits=CFG['K_FOLDS'], shuffle=True, random_state=SEED)
    particion = list(sgkf.split(Xd, yd, groups=gd))
    todos, hechos = [], 0

    for u in UNIDADES:
        print(f'\n{"="*24} Dense({u}) {"="*24}', flush=True)
        estado(fase=f'entrenando Dense({u})', unidades_actual=u)
        oof = np.full(len(Xd), np.nan); fold_de = np.full(len(Xd), -1)
        for fold, (tr, va) in enumerate(particion, start=1):
            assert not (set(gd[tr]) & set(gd[va])), f'FUGA en Dense({u}) fold {fold}'
            yp, info = entrenar_fold(u, Xd[tr], yd[tr], Xd[va], yd[va], fold)
            oof[va] = yp; fold_de[va] = fold
            todos.append(info); hechos += 1
            estado(folds_hechos=hechos, folds=todos)
        assert not np.isnan(oof).any()
        with open(OUT / f'oof_dense{u}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['archivo', 'ruta', 'clase', 'label', 'clave_agrupacion', 'rID', 'fold', 'prob'])
            for r, f, p in zip(dev, fold_de, oof):
                w.writerow([r['archivo'], r['ruta'], r['clase'], r['label'],
                            r['clave_agrupacion'], r['rID'], int(f), round(float(p), 6)])

    json.dump(dict(preespecificado=PREESPECIFICADO, ablacion=UNIDADES, folds=todos,
                   nota='Analisis de sensibilidad. No se selecciona la mejor configuracion: '
                        'hacerlo sobre el mismo desarrollo con el que se estima el rendimiento '
                        'incurriria en M-05.'),
              open(OUT / 'ablacion_dense.json', 'w'), indent=2, ensure_ascii=False)
    estado(fase='completado', folds_hechos=hechos)
    print('\nAblacion M-12 completada.', flush=True)


if __name__ == '__main__':
    main()
