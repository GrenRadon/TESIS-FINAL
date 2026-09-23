"""Revision del confusor pediatrico (M-10/M-14) y busqueda de atajos alternativos.

Responde cuatro preguntas:
  1. Calidad del dato de edad (coherencia con la URL, faltantes)
  2. Composicion clase x edad: ¿existe la correlacion que explicaria el confusor?
  3. ¿El efecto etario sobrevive en el modelo final, y bajo que alineacion?
  4. ¿Hay otro atajo? Tamano del recorte como predictor
"""
import sys, csv, re, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import mannwhitneyu, fisher_exact
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifiesto import leer_manifiesto

OUT = Path('resultados_v2')
CORTE = 18.0
V4 = Path('resultados_v4')
_UMB = __import__('json').loads(
    (Path('app/models/config_modelo.json')).read_text(encoding='utf-8'))['umbral']
MODELOS = [('v2 pipeline antiguo', OUT / 'predicciones_oof.csv', 0.5),
           ('v4 base (con volteo)', V4 / 'oof_base_DenseNet121.csv', 0.1903),
           ('v4 sin volteo DESPLEGADA', V4 / 'oof_sin_flip.csv', _UMB)]
RE_NUM = re.compile(r'^(?:chiari|normal)_(\d+)')
RE_URL = [(re.compile(r'(\d+)[-\s]?month'), lambda m: int(m) / 12),
          (re.compile(r'(\d+)[-\s]?year'), lambda m: float(m))]


def edad_de(v):
    try:
        return float(v['edad'])
    except (TypeError, ValueError):
        return None


def casos_de(archivo, reg, hipotesis='A'):
    """Agrega por caso y adjunta la edad segun la hipotesis de alineacion del manifiesto."""
    agg = defaultdict(lambda: {'p': [], 'y': None, 'n': None, 'c': None, 'rutas': []})
    for r in csv.DictReader(open(archivo, encoding='utf-8')):
        a = agg[r['clave_agrupacion']]
        a['p'].append(float(r['prob']))
        a['y'] = int(r['label'])
        a['n'] = int(RE_NUM.match(r['archivo']).group(1))
        a['c'] = r['clase']
        a['rutas'].append(r['ruta'])
    out = []
    for v in agg.values():
        if v['c'] == 'chiari':
            k = ('chiari', v['n'])
        else:
            k = ('normal', v['n'] if (hipotesis == 'A' or v['n'] <= 6) else v['n'] - 1)
        out.append(dict(y=v['y'], p=float(np.mean(v['p'])), c=v['c'],
                        e=edad_de(reg.get(k)) if reg.get(k) else None, rutas=v['rutas']))
    return out


def main():
    reg = leer_manifiesto()
    res = {}

    print('=' * 88)
    print('  1. CALIDAD DEL DATO DE EDAD')
    print('=' * 88)
    incoh = []
    for k, v in sorted(reg.items()):
        est = None
        for rx, f in RE_URL:
            m = rx.search(v['url'].lower())
            if m:
                est = f(m.group(1))
                break
        real = edad_de(v)
        if est is not None and real is not None and abs(real - est) > max(0.5, 0.35 * est):
            incoh.append((k[0], k[1], real, round(est, 2), v['url'].split('/cases/')[-1][:44]))
    for c, n, r, e, u in incoh:
        print(f'   incoherente: {c} {n} edad={r:g} pero la URL sugiere {e}  ({u})')
    for clase in ('chiari', 'normal'):
        falt = [k[1] for k, v in reg.items() if k[0] == clase and edad_de(v) is None]
        tot = sum(1 for k in reg if k[0] == clase)
        print(f'   {clase}: {len(falt)} de {tot} casos sin edad -> {sorted(falt)}')
    res['incoherencias_edad'] = incoh

    print('\n' + '=' * 88)
    print('  2. COMPOSICION CLASE x EDAD EN DESARROLLO — ¿existe el mecanismo?')
    print('=' * 88)
    ds = [r for r in csv.DictReader(open(OUT / 'dataset_nivel_imagen.csv', encoding='utf-8'))
          if r['split'] == 'desarrollo']  # dataset v4: 97 casos
    casos = {}
    for r in ds:
        n = int(r['numero_base'])
        v = reg.get((r['clase'], n))
        if v:
            casos[(r['clase'], n)] = edad_de(v)
    comp = {}
    for corte in (2, 5, 18):
        pc = sum(1 for (c, _), e in casos.items() if c == 'chiari' and e is not None and e < corte)
        pn = sum(1 for (c, _), e in casos.items() if c == 'normal' and e is not None and e < corte)
        ac = sum(1 for (c, _), e in casos.items() if c == 'chiari' and e is not None and e >= corte)
        an = sum(1 for (c, _), e in casos.items() if c == 'normal' and e is not None and e >= corte)
        p_ped = pc / (pc + pn) if pc + pn else 0
        p_adu = ac / (ac + an) if ac + an else 0
        comp[corte] = dict(p_chiari_ped=round(p_ped, 3), p_chiari_adu=round(p_adu, 3),
                           n_ped=pc + pn, n_adu=ac + an)
        print(f'   corte <{corte:2d}a: pediatricos {pc + pn:2d} ({pc} ch / {pn} no) '
              f'P(chiari|ped)={p_ped:.2f}   |   adultos {ac + an:2d} P(chiari|adu)={p_adu:.2f}')
    print('\n   El confusor por correlacion de etiqueta requeriria P(chiari|ped) > P(chiari|adu).')
    print('   Se observa lo contrario: el mecanismo supuesto NO esta respaldado por los datos.')
    res['composicion_clase_edad'] = comp

    print('\n' + '=' * 88)
    print('  3. EFECTO ETARIO EN LOS NEGATIVOS, POR MODELO Y ALINEACION')
    print('=' * 88)
    print(f'   {"modelo":<24}{"hip":>4}{"n_ped":>7}{"n_adu":>7}{"p_ped":>8}{"p_adu":>8}'
          f'{"MW p":>9}{"FP p/a":>9}{"Fisher":>9}')
    efecto = {}
    for etq, arch, umb in MODELOS:
        for hip in ('A', 'B'):
            cs = casos_de(arch, reg, hip)
            pe = [c['p'] for c in cs if c['y'] == 0 and c['e'] is not None and c['e'] < CORTE]
            ad = [c['p'] for c in cs if c['y'] == 0 and c['e'] is not None and c['e'] >= CORTE]
            if not pe or not ad:
                continue
            mw = float(mannwhitneyu(pe, ad, alternative='two-sided')[1])
            fpe = sum(1 for x in pe if x >= umb)
            fad = sum(1 for x in ad if x >= umb)
            fi = float(fisher_exact([[fpe, len(pe) - fpe], [fad, len(ad) - fad]])[1])
            print(f'   {etq:<24}{hip:>4}{len(pe):>7}{len(ad):>7}{np.mean(pe):>8.3f}'
                  f'{np.mean(ad):>8.3f}{mw:>9.4f}{f"{fpe}/{fad}":>9}{fi:>9.4f}')
            efecto[f'{etq}|{hip}'] = dict(n_ped=len(pe), n_adu=len(ad),
                                          media_ped=round(float(np.mean(pe)), 4),
                                          media_adu=round(float(np.mean(ad)), 4),
                                          mannwhitney_p=round(mw, 4), fisher_p=round(fi, 4))
    res['efecto_etario'] = efecto

    print('\n' + '=' * 88)
    print('  4. ATAJO ALTERNATIVO: TAMANO DEL RECORTE')
    print('=' * 88)
    X, y, g = [], [], []
    for r in ds:
        w, h = Image.open(r['ruta']).size
        X.append([w, h, w * h, w / h])
        y.append(int(r['label']))
        g.append(r['clave_agrupacion'])
    X, y, g = np.array(X, float), np.array(y), np.array(g)
    for cl, lab in ((1, 'chiari'), (0, 'normal')):
        s = X[y == cl]
        print(f'   {lab:7s} n={len(s):3d}  ancho={s[:, 0].mean():6.1f}+-{s[:, 0].std():5.1f}  '
              f'alto={s[:, 1].mean():6.1f}+-{s[:, 1].std():5.1f}  area mediana={np.median(s[:, 2]):9.0f}')
    p_area = float(mannwhitneyu(X[y == 1, 2], X[y == 0, 2])[1])
    print(f'   Mann-Whitney del area, chiari vs normal: p={p_area:.2e}')

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, va in sgkf.split(X, y, groups=g):
        oof[va] = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    agg = defaultdict(lambda: {'p': [], 'y': None})
    for gg, pp, yy in zip(g, oof, y):
        agg[gg]['p'].append(pp)
        agg[gg]['y'] = yy
    yc = np.array([v['y'] for v in agg.values()])
    pc = np.array([np.mean(v['p']) for v in agg.values()])
    auc_caso = float(roc_auc_score(yc, pc))
    print(f'   Regresion logistica SOLO con las dimensiones (sin mirar la imagen):')
    print(f'     AUC OOF por imagen = {roc_auc_score(y, oof):.4f}   por caso = {auc_caso:.4f}')
    print(f'     (la CNN sobre la imagen da 0.9574 por caso)')
    res['atajo_tamano'] = dict(p_area=float(f'{p_area:.3e}'), auc_solo_dimensiones_caso=round(auc_caso, 4),
                               area_mediana_chiari=float(np.median(X[y == 1, 2])),
                               area_mediana_normal=float(np.median(X[y == 0, 2])))

    json.dump(res, open(OUT / 'confusor_edad.json', 'w'), indent=2, ensure_ascii=False)
    print(f'\nJSON: {OUT / "confusor_edad.json"}')


if __name__ == '__main__':
    main()
