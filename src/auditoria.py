"""Auditoria de los archivos en disco: inventario, deduplicacion y deteccion de fuga.

Reproduce la auditoria original y debe re-ejecutarse cada vez que cambien las imagenes.
Salidas: auditoria_nivel_imagen.csv, auditoria_nivel_caso.csv
"""
import csv, hashlib, re
from collections import defaultdict
from pathlib import Path

OUT = Path('resultados_v2')
OUT.mkdir(exist_ok=True)
RE = re.compile(r'^(?:(chiari|normal)_)?(\d+)([a-zA-Z]*)\.(jpg|jpeg|png)$', re.I)
FOLDERS = {
    ('entrenamiento', 'chiari'): Path('data/vf/cropped/chiari'),
    ('entrenamiento', 'normal'): Path('data/vf/cropped/normal'),
    ('externo', 'chiari'): Path('data/vf/val_externo/cropped/chiari'),
    ('externo', 'normal'): Path('data/vf/val_externo/cropped/normal'),
}
# Archivos byte-identicos a imagenes Chiari, colocados en la carpeta normal/.
# Confirmado con matchTemplate contra data/vf/raw/chiari (corr >= 0.9987).
IDENT_MAL_ETIQ = {('chiari', '25'), ('chiari', '53'), ('chiari', '58'),
                  ('chiari', '59'), ('chiari', '70')}


def main():
    rows, sinmatch = [], []
    for (carpeta, clase), p in FOLDERS.items():
        if not p.exists():
            continue
        for f in sorted(p.iterdir()):
            if not f.is_file():
                continue
            m = RE.match(f.name)
            if not m:
                sinmatch.append((carpeta, clase, f.name))
                continue
            _, num, suf, _ = m.groups()
            rows.append(dict(archivo=f.name, numero_base=int(num), sufijo=suf.lower(),
                             clave_agrupacion=f'{clase}_{int(num)}',
                             rID=f'{clase.upper()[:1]}{int(num):03d}', clase=clase,
                             carpeta=carpeta, ruta=str(f).replace('\\', '/'),
                             sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                             bytes=f.stat().st_size))

    carp, hashes = defaultdict(set), defaultdict(lambda: defaultdict(set))
    for r in rows:
        k = (r['clase'], str(r['numero_base']))
        carp[k].add(r['carpeta'])
        hashes[k][r['carpeta']].add(r['sha256'])
    for r in rows:
        k = (r['clase'], str(r['numero_base']))
        ambas = len(carp[k]) == 2
        r['fuga_train_externo'] = 'SI' if ambas else 'NO'
        if not ambas:
            r['tipo_solape'] = ''
        elif hashes[k]['entrenamiento'] & hashes[k]['externo']:
            r['tipo_solape'] = 'archivo_identico_sha256'
        elif k in IDENT_MAL_ETIQ:
            r['tipo_solape'] = 'mismo_raw_recorte_distinto'
        else:
            r['tipo_solape'] = 'mismo_caso_corte_distinto'
        r['manifiesto'] = 'ver manifiesto_final_casos.csv'

    cols = ['archivo', 'numero_base', 'clave_agrupacion', 'rID', 'clase', 'carpeta', 'sufijo',
            'fuga_train_externo', 'tipo_solape', 'sha256', 'bytes', 'ruta', 'manifiesto']
    with open(OUT / 'auditoria_nivel_imagen.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r['clase'], r['numero_base'],
                                                r['sufijo'], r['carpeta'])))

    casos = {}
    for r in rows:
        k = (r['clase'], r['numero_base'])
        c = casos.setdefault(k, dict(clase=r['clase'], numero_base=k[1],
                                     clave_agrupacion=r['clave_agrupacion'], rID=r['rID'],
                                     n_train=0, n_externo=0, fuga=r['fuga_train_externo'],
                                     tipo=r['tipo_solape']))
        c['n_train' if r['carpeta'] == 'entrenamiento' else 'n_externo'] += 1
    with open(OUT / 'auditoria_nivel_caso.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['rID', 'clave_agrupacion', 'clase', 'numero_base',
                                           'n_train', 'n_externo', 'fuga', 'tipo'])
        w.writeheader()
        for k in sorted(casos):
            w.writerow(casos[k])

    for (carpeta, clase), p in FOLDERS.items():
        sub = [r for r in rows if r['carpeta'] == carpeta and r['clase'] == clase]
        print(f'  {carpeta:14s} {clase:7s}: {len(sub):3d} img | '
              f'{len({r["numero_base"] for r in sub}):3d} casos')
    nf = sum(1 for c in casos.values() if c['fuga'] == 'SI')
    print(f'  casos totales: {len(casos)} | con solape train/externo: {nf}')
    print(f'  imagenes: {len(rows)} | sin match de patron: {len(sinmatch)}')
    if sinmatch:
        print(f'    {sinmatch}')


if __name__ == '__main__':
    main()
