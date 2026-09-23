"""Reentrena los 5 pliegues de la variante DESPLEGADA (sin volteo) GUARDANDO los modelos.

En la corrida v4 solo se persistieron los modelos de las variantes base_*, por lo que el
analisis de Grad-CAM tuvo que hacerse sobre la configuracion base. Esto reentrena sin_flip
con la MISMA particion, semilla y epocas, para poder recalcular Grad-CAM sobre el modelo
realmente desplegado.

Verificacion incluida: las predicciones OOF deben reproducir las de la corrida v4.
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
from entrenamiento_v4 import FoldGen, construir, CFG, EPOCAS, SEED

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

V4 = Path('resultados_v4')
MODELOS = Path('models_v4')
ESTADO = V4 / 'estado_sinflip.json'
ARQ = 'DenseNet121'
_t0 = time.time()


def estado(**kw):
    b = json.loads(ESTADO.read_text(encoding='utf-8')) if ESTADO.exists() else {}
    b.update(kw)
    b['actualizado'] = time.strftime('%H:%M:%S')
    b['transcurrido_min'] = round((time.time() - _t0) / 60, 1)
    b['total_folds'] = CFG['K_FOLDS']
    ESTADO.write_text(json.dumps(b, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    rows = list(csv.DictReader(open('resultados_v2/dataset_nivel_imagen.csv', encoding='utf-8')))
    dev = [r for r in rows if r['split'] == 'desarrollo']
    Xd = np.array([r['ruta'] for r in dev]); yd = np.array([int(r['label']) for r in dev])
    gd = np.array([r['clave_agrupacion'] for r in dev])
    ep1, ep2 = EPOCAS[ARQ]

    print(f'  Reentrenando sin_flip: {len(Xd)} img / {len(set(gd))} casos, '
          f'epocas {ep1}+{ep2}, semilla {SEED}', flush=True)
    estado(fase='iniciando', folds_hechos=0)

    sgkf = StratifiedGroupKFold(n_splits=CFG['K_FOLDS'], shuffle=True, random_state=SEED)
    oof = np.full(len(Xd), np.nan)
    for fold, (tr, va) in enumerate(sgkf.split(Xd, yd, groups=gd), start=1):
        assert not (set(gd[tr]) & set(gd[va])), f'FUGA en fold {fold}'
        tf.keras.backend.clear_session()
        keras.utils.set_random_seed(SEED + fold)
        modelo, base = construir(ARQ)
        gen_tr = FoldGen(Xd[tr], yd[tr], CFG['BATCH_SIZE'], augment=True, shuffle=True, flip=False)
        gen_va = FoldGen(Xd[va], yd[va], CFG['BATCH_SIZE'], augment=False, shuffle=False, flip=False)
        cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=yd[tr])
        cwd = {0: float(cw[0]), 1: float(cw[1])}
        cbs = [ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-8, verbose=0)]

        def compilar(lr):
            modelo.compile(optimizer=keras.optimizers.Adam(lr), loss='binary_crossentropy',
                           metrics=['accuracy', keras.metrics.AUC(name='auc')])

        t0 = time.time()
        compilar(CFG['LR_F1'])
        modelo.fit(gen_tr, epochs=ep1, callbacks=cbs, class_weight=cwd, verbose=0)
        base.trainable = True
        for lay in base.layers[:-CFG['UNFREEZE_TOP']]:
            lay.trainable = False
        compilar(CFG['LR_F2'])
        modelo.fit(gen_tr, epochs=ep2, callbacks=cbs, class_weight=cwd, verbose=0)

        yp = modelo.predict(gen_va, verbose=0).flatten()
        oof[va] = yp
        modelo.save(str(MODELOS / f'sin_flip_fold{fold}.keras'))    # <- lo que faltaba
        auc = roc_auc_score(yd[va], yp)
        print(f'    fold {fold}: AUC_val={auc:.4f} ({(time.time()-t0)/60:.1f} min) — modelo guardado',
              flush=True)
        estado(folds_hechos=fold)

    # --- verificacion: ¿reproduce la corrida v4? ---
    # Emparejar por RUTA, no por nombre: 3 archivos comparten nombre entre carpetas
    # (chiari_25b.jpg, chiari_59a.png, chiari_70a.jpg) y un dict por nombre los colapsa.
    prev = {r['ruta']: float(r['prob'])
            for r in csv.DictReader(open(V4 / 'oof_sin_flip.csv', encoding='utf-8'))}
    difs = [abs(prev[r['ruta']] - p) for r, p in zip(dev, oof) if r['ruta'] in prev]
    print(f'\n  Reproducibilidad frente a la corrida v4:')
    print(f'    max|dif| = {max(difs):.4f}   ·   media = {np.mean(difs):.4f}')
    auc_new = roc_auc_score(yd, oof)
    print(f'    AUC por imagen: nueva {auc_new:.4f}')
    estado(fase='completado', max_dif=round(float(max(difs)), 5),
           media_dif=round(float(np.mean(difs)), 5))
    with open(V4 / 'oof_sin_flip_reentrenado.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['archivo', 'ruta', 'clase', 'label', 'clave_agrupacion', 'rID', 'prob'])
        for r, p in zip(dev, oof):
            w.writerow([r['archivo'], r['ruta'], r['clase'], r['label'],
                        r['clave_agrupacion'], r['rID'], round(float(p), 6)])
    print('  Listo.')


if __name__ == '__main__':
    main()
