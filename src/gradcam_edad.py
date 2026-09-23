"""M-10/M-14: Grad-CAM comparando FP pediatricos vs aciertos Chiari.

Cada imagen se explica con el modelo del fold que NO la vio en entrenamiento (el mismo
que produjo su prediccion out-of-fold), no con un modelo que ya la memorizo.

Ademas de la inspeccion visual, se cuantifica donde cae la activacion:
  - centro de masa vertical del mapa (0 = arriba, 1 = abajo)
  - fraccion de la masa de activacion en el tercio inferior (proxy de fosa posterior)
"""
import csv, json, os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from figuras import guardar
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from scipy.stats import mannwhitneyu

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocesamiento as PP

OUT = Path('resultados_v2')
V4 = Path('resultados_v4')
FIG = OUT / 'gradcam'
FIG.mkdir(parents=True, exist_ok=True)
MODELOS = Path('models_v4')
LAST_CONV = 'conv5_block16_concat'
IMG = (224, 224)
CORTE_PEDIATRICO = 18.0


def cargar_imagen(ruta):
    """Fuente unica (C-06). NO normalizar aqui: el modelo desplegado lleva preprocess_input
    dentro del grafo y espera la entrada en [0,255]. Dividir por 255 aqui hacia que el
    Grad-CAM se calculara sobre una entrada fuera de escala y las probabilidades no
    coincidieran con las out-of-fold."""
    return PP.cargar_imagen(ruta)


class Explicador:
    """Reconstruye base->feature map y la cabeza para poder derivar el mapa de activacion."""

    def __init__(self, ruta_modelo):
        self.modelo = keras.models.load_model(str(ruta_modelo))
        base = None
        for lay in self.modelo.layers:
            if isinstance(lay, keras.Model) and lay.name.lower().startswith('densenet'):
                base = lay
                break
        if base is None:
            raise RuntimeError('no se encontro la base DenseNet dentro del modelo')
        self.base = base
        self.feat = Model(base.input, [base.get_layer(LAST_CONV).output, base.output])
        i = self.modelo.layers.index(base)
        # Entre la entrada y la base van las capas de preprocess_input (C-06). Si se alimenta
        # la base directamente se salta ese bloque y el mapa NO describe al modelo desplegado.
        self.previas = [l for l in self.modelo.layers[:i]
                        if not isinstance(l, keras.layers.InputLayer)]
        self.cabeza = self.modelo.layers[i + 1:]
        print(f'    capas previas a la base: '
              f'{[l.name for l in self.previas] or "ninguna"}')

    def _preparar(self, x):
        for lay in self.previas:
            x = lay(x, training=False)
        return x

    def _predecir(self, base_out):
        x = base_out
        for lay in self.cabeza:
            x = lay(x, training=False)
        return x

    def gradcam(self, img):
        x = tf.convert_to_tensor(img[None, ...], dtype=tf.float32)
        x = self._preparar(x)
        with tf.GradientTape() as tape:
            conv, base_out = self.feat(x, training=False)
            tape.watch(conv)
            pred = self._predecir(base_out)[:, 0]
        grads = tape.gradient(pred, conv)
        pesos = tf.reduce_mean(grads, axis=(0, 1, 2))
        hm = tf.squeeze(conv[0] @ pesos[..., None]).numpy()
        hm = np.maximum(hm, 0)
        if hm.max() > 0:
            hm = hm / hm.max()
        return cv2.resize(hm, IMG), float(pred[0])


def metricas_mapa(hm):
    s = hm.sum()
    if s <= 0:
        return dict(centro_masa_y=float('nan'), frac_tercio_inferior=float('nan'))
    filas = hm.sum(axis=1)
    ys = np.arange(hm.shape[0])
    cm = float((filas * ys).sum() / s / (hm.shape[0] - 1))
    corte = int(hm.shape[0] * 2 / 3)
    return dict(centro_masa_y=round(cm, 4),
                frac_tercio_inferior=round(float(hm[corte:, :].sum() / s), 4))


def main():
    oof = list(csv.DictReader(open(V4 / 'oof_sin_flip.csv', encoding='utf-8')))
    man = {r['clave_agrupacion']: r for r in
           csv.DictReader(open(OUT / 'manifiesto_final_casos.csv', encoding='utf-8'))}

    def edad(c):
        s = (man.get(c, {}).get('edad') or '').strip()
        try:
            return float(s)
        except ValueError:
            return None

    # probabilidad media por caso para decidir acierto/error (igual que la metrica primaria)
    agg = defaultdict(lambda: {'p': [], 'y': None})
    for r in oof:
        a = agg[r['clave_agrupacion']]
        a['p'].append(float(r['prob']))
        a['y'] = int(r['label'])
    pred_caso = {c: float(np.mean(v['p'])) for c, v in agg.items()}
    UMBRAL = json.load(open(Path('app/models/config_modelo.json'),
                            encoding='utf-8'))['umbral']
    print(f'  umbral de operacion usado para definir acierto/error: {UMBRAL}')

    grupos = defaultdict(list)
    for r in oof:
        c = r['clave_agrupacion']
        e = edad(c)
        if e is None:
            continue
        ped = e < CORTE_PEDIATRICO
        y = int(r['label'])
        acierto = (pred_caso[c] >= UMBRAL) == (y == 1)
        if y == 0 and ped and not acierto:
            grupos['FP_pediatrico'].append(r)
        elif y == 1 and ped and acierto:
            grupos['TP_chiari_pediatrico'].append(r)
        elif y == 1 and not ped and acierto:
            grupos['TP_chiari_adulto'].append(r)
        elif y == 0 and not ped and acierto:
            grupos['TN_normal_adulto'].append(r)

    print('Imagenes por grupo:', {k: len(v) for k, v in grupos.items()})

    cache, filas = {}, []
    for g, rows in grupos.items():
        for r in rows:
            f = int(r['fold'])
            if f not in cache:
                print(f'  cargando modelo fold {f}...', flush=True)
                cache[f] = Explicador(MODELOS / f'sin_flip_fold{f}.keras')
            img = cargar_imagen(r['ruta'])
            hm, prob = cache[f].gradcam(img)
            m = metricas_mapa(hm)
            filas.append(dict(grupo=g, archivo=r['archivo'], clave=r['clave_agrupacion'],
                              fold=f, prob=round(prob, 4), edad=edad(r['clave_agrupacion']),
                              ruta=r['ruta'], **m))

    # SALVAGUARDA: la probabilidad que devuelve el explicador tiene que coincidir con la
    # out-of-fold. Si no coincide, la entrada no esta en la escala que espera el modelo y
    # los mapas no describen al modelo desplegado.
    ref = {r['ruta']: float(r['prob']) for r in oof}
    difs = [abs(f['prob'] - ref[f['ruta']]) for f in filas if f['ruta'] in ref]
    peor = max(difs) if difs else 0.0
    print(f'\n  verificacion Grad-CAM vs OOF: max|dif| = {peor:.6f} sobre {len(difs)} imagenes')
    assert peor < 1e-3, (f'Las probabilidades del Grad-CAM no reproducen las out-of-fold '
                         f'(max {peor:.4f}). Revisa la escala de la entrada.')

    with open(OUT / 'gradcam_metricas.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    print('\n=== LOCALIZACION DE LA ACTIVACION POR GRUPO ===')
    print(f'  {"grupo":<24}{"n":>4}{"centro masa Y":>16}{"frac tercio inf":>18}')
    resumen = {}
    for g in ('FP_pediatrico', 'TP_chiari_pediatrico', 'TP_chiari_adulto', 'TN_normal_adulto'):
        sub = [f for f in filas if f['grupo'] == g]
        if not sub:
            continue
        cm = np.array([f['centro_masa_y'] for f in sub], dtype=float)
        fi = np.array([f['frac_tercio_inferior'] for f in sub], dtype=float)
        deg = int(np.isnan(fi).sum())          # mapas degenerados: Grad-CAM todo cero
        cm, fi = cm[~np.isnan(cm)], fi[~np.isnan(fi)]
        resumen[g] = dict(n=len(sub), n_validos=int(len(fi)), degenerados=deg,
                          centro_masa_y=round(float(cm.mean()), 4) if len(cm) else None,
                          frac_tercio_inferior=round(float(fi.mean()), 4) if len(fi) else None)
        print(f'  {g:<24}{len(sub):>4}{cm.mean():>10.3f} ± {cm.std():.3f}'
              f'{fi.mean():>12.3f} ± {fi.std():.3f}   (degenerados: {deg})')

    print('\n=== COMPARACIONES (Mann-Whitney sobre frac. tercio inferior) ===')
    comp = {}
    pares = [('FP_pediatrico', 'TP_chiari_pediatrico'), ('FP_pediatrico', 'TP_chiari_adulto'),
             ('TP_chiari_pediatrico', 'TP_chiari_adulto')]
    for a, b in pares:
        va = [f['frac_tercio_inferior'] for f in filas
              if f['grupo'] == a and not np.isnan(f['frac_tercio_inferior'])]
        vb = [f['frac_tercio_inferior'] for f in filas
              if f['grupo'] == b and not np.isnan(f['frac_tercio_inferior'])]
        if not va or not vb:
            continue
        u, pv = mannwhitneyu(va, vb, alternative='two-sided')
        comp[f'{a}_vs_{b}'] = round(float(pv), 4)
        print(f'  {a:<24} vs {b:<24} p={pv:.4f}')

    json.dump(dict(corrida='v4 · configuracion DESPLEGADA (sin volteo), 97 casos',
                   resumen=resumen, comparaciones=comp,
                   corte_pediatrico=CORTE_PEDIATRICO, ultima_conv=LAST_CONV),
              open(OUT / 'gradcam_resumen.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    # --- figura comparativa ---
    # Las CUATRO filas: la de negativos adultos acertados es la que sostiene el argumento
    # del apartado (activan el tercio inferior MAS que los Chiari), asi que tiene que verse.
    orden = ['FP_pediatrico', 'TP_chiari_pediatrico', 'TP_chiari_adulto', 'TN_normal_adulto']
    ETIQUETAS = {'FP_pediatrico': 'Falso positivo\npediátrico',
                 'TP_chiari_pediatrico': 'Acierto Chiari\npediátrico',
                 'TP_chiari_adulto': 'Acierto Chiari\nadulto',
                 'TN_normal_adulto': 'Acierto Normal\nadulto'}
    ncol = max(len([f for f in filas if f['grupo'] == g]) for g in orden)
    ncol = min(ncol, 6)
    fig, axes = plt.subplots(len(orden), ncol, figsize=(2.4 * ncol, 2.7 * len(orden)))
    fig.suptitle('Grad-CAM out-of-fold sobre la configuración desplegada — '
                 '¿mira el modelo la fosa posterior?\n'
                 'Los aciertos Normal adultos (última fila) son los que MÁS activan el '
                 'tercio inferior: si esa activación indicara descenso amigdalar, el orden '
                 'estaría invertido.', fontsize=9)
    for fila, g in enumerate(orden):
        sub = [f for f in filas if f['grupo'] == g][:ncol]
        for col in range(ncol):
            ax = axes[fila][col] if ncol > 1 else axes[fila]
            ax.axis('off')
            if col >= len(sub):
                continue
            f = sub[col]
            img = cargar_imagen(f['ruta'])
            hm, _ = cache[f['fold']].gradcam(img)
            hc = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET),
                              cv2.COLOR_BGR2RGB)
            ax.imshow(cv2.addWeighted(np.uint8(np.clip(img, 0, 255)), 0.6, hc, 0.4, 0))
            ax.set_title(f"{f['archivo'][:16]}\np={f['prob']:.2f} · {f['edad']:.0f}a · "
                         f"inf={f['frac_tercio_inferior']:.2f}", fontsize=6.5)
        ax0 = axes[fila][0] if ncol > 1 else axes[fila]
        ax0.set_ylabel(ETIQUETAS.get(g, g), fontsize=7.5)
        ax0.axis('on'); ax0.set_xticks([]); ax0.set_yticks([])
    plt.tight_layout()
    guardar(fig, FIG / 'gradcam_pediatrico_vs_chiari.png')
    print(f'\nfigura: {FIG / "gradcam_pediatrico_vs_chiari.png"}')
    print(f'CSV   : {OUT / "gradcam_metricas.csv"}')


if __name__ == '__main__':
    main()
