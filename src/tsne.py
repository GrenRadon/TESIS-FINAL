"""M-14 — t-SNE del EDA, rehecho declarando parametros y semilla.

El hallazgo objeta que la tesis usaba la ausencia de separacion en t-SNE como argumento
sobre la viabilidad de la CNN. t-SNE es una tecnica de VISUALIZACION: no preserva
distancias globales, es muy sensible a la perplejidad y a la semilla, y no permite
inferencia de ningun tipo.

Este script lo demuestra empiricamente: dibuja el mismo conjunto con varias perplejidades
y varias semillas para que se vea que la estructura aparente cambia con los parametros.
"""
import csv, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from figuras import guardar
from sklearn.manifold import TSNE, trustworthiness
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import procrustes

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocesamiento as PP

V2 = Path('resultados_v2')
FIG = V2 / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
PERPLEJIDADES = [5, 15, 30, 50]
K_VECINOS = 10
SEMILLAS = [0, 42]
N_PCA = 50
COLOR = {1: '#D64545', 0: '#1D9E75'}


def cargar():
    """Vectores de pixel de las imagenes ya preprocesadas (mismo pipeline del modelo)."""
    rows = [r for r in csv.DictReader(open(V2 / 'dataset_nivel_imagen.csv', encoding='utf-8'))]
    X, y = [], []
    for r in rows:
        img = PP.cargar_imagen(r['ruta'])[:, :, 0]
        X.append(np.asarray(img, dtype=np.float32).reshape(-1)[::16])   # submuestreo espacial
        y.append(int(r['label']))
    return np.array(X), np.array(y), rows


def cohesion_clase(emb, y, k=None):
    """Fraccion de los k vecinos mas cercanos EN LA PROYECCION que son de la misma clase.

    Es la lectura que un ojo humano hace de un t-SNE ('las clases se separan'), puesta en
    numero. El valor de referencia es la proporcion esperada al azar, que depende del
    desbalance: sum_c p_c^2.
    """
    k = k or K_VECINOS
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    idx = nn.kneighbors(emb, return_distance=False)[:, 1:]
    return float((y[idx] == y[:, None]).mean())


def azar_clase(y):
    p = np.array([(y == c).mean() for c in np.unique(y)])
    return float((p ** 2).sum())


def discrepancia(a, b):
    """Disimilitud de Procrustes entre dos proyecciones del MISMO conjunto de puntos.

    0 = identicas salvo rotacion, escala y traslacion; 1 = sin relacion. Es la forma de
    poner numero a 'la estructura cambia' sin depender de la inspeccion visual.
    """
    return float(procrustes(a, b)[2])


def main():
    X, y, rows = cargar()
    print(f'  {len(X)} imágenes · vector de {X.shape[1]} dimensiones (pixeles submuestreados)')
    Xp = PCA(n_components=min(N_PCA, len(X) - 1), random_state=0).fit_transform(X)
    print(f'  PCA previo a {Xp.shape[1]} componentes (práctica estándar antes de t-SNE)')

    embs = {}
    fig, axes = plt.subplots(len(SEMILLAS), len(PERPLEJIDADES),
                             figsize=(4 * len(PERPLEJIDADES), 4 * len(SEMILLAS)))
    res = []
    for i, sem in enumerate(SEMILLAS):
        for j, per in enumerate(PERPLEJIDADES):
            emb = TSNE(n_components=2, perplexity=per, random_state=sem,
                       init='pca', learning_rate='auto').fit_transform(Xp)
            ax = axes[i][j]
            for cl, lab in ((0, 'Normal'), (1, 'Chiari')):
                m = y == cl
                ax.scatter(emb[m, 0], emb[m, 1], s=14, c=COLOR[cl], label=lab,
                           alpha=.75, edgecolors='none')
            ax.set_title(f'perplejidad={per} · semilla={sem}', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc='best')
            tw = float(trustworthiness(Xp, emb, n_neighbors=K_VECINOS))
            coh = cohesion_clase(emb, y)
            embs[(per, sem)] = emb
            res.append(dict(perplejidad=per, semilla=sem,
                            trustworthiness=round(tw, 4),
                            vecinos_misma_clase=round(coh, 4)))
            print(f'    perplejidad={per:>2} semilla={sem} -> trustworthiness={tw:.4f} '
                  f'· vecinos de la misma clase={coh:.4f}', flush=True)

    fig.suptitle('t-SNE del conjunto de desarrollo y prueba — VISUALIZACIÓN EXPLORATORIA\n'
                 'La estructura aparente cambia con la perplejidad y con la semilla: '
                 'no permite inferir viabilidad ni separabilidad',
                 fontsize=10.5, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    guardar(fig, FIG / 'tsne_sensibilidad.png')

    # --- cuantificacion de "la estructura cambia" ---
    az = azar_clase(y)
    print(chr(10) + '  === ESTABILIDAD ENTRE SEMILLAS (misma perplejidad) ===')
    est = []
    for per in PERPLEJIDADES:
        d = discrepancia(embs[(per, SEMILLAS[0])], embs[(per, SEMILLAS[1])])
        est.append(dict(perplejidad=per, procrustes_entre_semillas=round(d, 4)))
        print(f'    perplejidad={per:>2}: disimilitud de Procrustes entre semilla '
              f'{SEMILLAS[0]} y {SEMILLAS[1]} = {d:.4f}')
    print(chr(10) + '  === ESTABILIDAD ENTRE PERPLEJIDADES (misma semilla) ===')
    est_per = []
    for sem in SEMILLAS:
        for i in range(len(PERPLEJIDADES) - 1):
            a, b = PERPLEJIDADES[i], PERPLEJIDADES[i + 1]
            d = discrepancia(embs[(a, sem)], embs[(b, sem)])
            est_per.append(dict(semilla=sem, de=a, a=b, procrustes=round(d, 4)))
            print(f'    semilla={sem}: perplejidad {a} vs {b} = {d:.4f}')

    tws = [r['trustworthiness'] for r in res]
    cohs = [r['vecinos_misma_clase'] for r in res]
    print(chr(10) + f'  trustworthiness (k={K_VECINOS}): entre {min(tws):.4f} y {max(tws):.4f}')
    print(f'  vecinos de la misma clase: entre {min(cohs):.4f} y {max(cohs):.4f} '
          f'(al azar seria {az:.4f})')

    json.dump(dict(n_imagenes=int(len(X)), dim_vector=int(X.shape[1]),
                   pca_componentes=int(Xp.shape[1]), perplejidades=PERPLEJIDADES,
                   semillas=SEMILLAS, init='pca', learning_rate='auto',
                   k_vecinos=K_VECINOS, proyecciones=res,
                   vecinos_misma_clase_al_azar=round(az, 4),
                   trustworthiness_min=round(min(tws), 4),
                   trustworthiness_max=round(max(tws), 4),
                   vecinos_misma_clase_min=round(min(cohs), 4),
                   vecinos_misma_clase_max=round(max(cohs), 4),
                   estabilidad_entre_semillas=est,
                   estabilidad_entre_perplejidades=est_per,
                   criterio='La afirmacion "la estructura cambia" NO es inspeccion visual: se '
                            'mide con la disimilitud de Procrustes entre proyecciones del mismo '
                            'conjunto de puntos (0 = identicas salvo rotacion/escala/traslacion, '
                            '1 = sin relacion). La lectura "las clases se separan" se mide con la '
                            'fraccion de los k vecinos mas cercanos EN LA PROYECCION que son de la '
                            'misma clase, contra el valor esperado al azar.',
                   nota='t-SNE es visualizacion exploratoria. No preserva distancias globales, '
                        'depende de la perplejidad y de la semilla, y NO permite inferencia sobre '
                        'separabilidad ni sobre la viabilidad de un clasificador.'),
              open(V2 / 'tsne_parametros.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n  figura: {FIG / "tsne_sensibilidad.png"}')
    print(f'  JSON  : {V2 / "tsne_parametros.json"}')


if __name__ == '__main__':
    main()
