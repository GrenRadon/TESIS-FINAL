"""M-01 / C-10: manifiesto reproducible a nivel de imagen, con atribucion y licencia.

Se regenera siempre desde dataset_nivel_imagen.csv, para que no quede desfasado cuando
cambian las exclusiones o entran imagenes nuevas.
"""
import csv, datetime, os, re
from pathlib import Path

OUT = Path('resultados_v2')
# Terminos verificados el 2026-09-06 en la politica oficial de Radiopaedia
# "Using and attributing images from Radiopaedia" (https://doi.org/10.53347/rID-89930):
#   - licencia UNICA DEL SITIO ("the Creative Commons license we use"), no por caso
#   - uso limitado a fines NO COMERCIALES
#   - atribucion obligatoria al colaborador y a Radiopaedia; se recomienda incluir el rID
#   - "you do not copyright the material" (clausula de no apropiacion)
# VERSION CONFIRMADA el 2026-09-06 en la pagina de licencia de Radiopaedia, que declara
# literalmente: "we make all our cases available under a Creative Commons Non-Commercial
# Attribution, Sharealike license (currently CC-NC-BY-SA 3.0)".
# DOS SALVEDADES que deben figurar en el texto legal:
#   - "currently": la version es la vigente en la fecha de acceso, puede cambiar.
#   - los Terminos de Uso de Radiopaedia contienen MODIFICACIONES sobre la licencia base y
#     son, segun el propio sitio, "the single source of truth".
LICENCIA = 'CC BY-NC-SA 3.0 (Radiopaedia, licencia unica del sitio, con las modificaciones de sus Terminos de Uso)'
POLITICA = 'https://doi.org/10.53347/rID-89930'
RE_FECHA = re.compile(r'Accessed on (\d{1,2} \w{3} \d{4})')

COLS = ['archivo', 'sha256', 'clase', 'split', 'clave_agrupacion', 'rID', 'numero_base',
        'edad', 'sexo', 'diagnostico', 'cita', 'atribucion_ok', 'licencia',
        'licencia_alcance', 'licencia_verificada', 'politica_url',
        'fecha_acceso', 'fecha_acceso_origen', 'fuente', 'url_caso', 'ruta']

# Radiopaedia ofrece dos formatos de atribucion. El largo ("Autor, Titulo. Case study,
# Radiopaedia.org (Accessed on DD Mmm YYYY)") lleva la fecha dentro; el corto
# ("Case courtesy of Autor, Radiopaedia.org, rID: N") NO la lleva. Los casos capturados con
# el formato corto no tienen fecha que extraer, asi que se toma la del archivo en disco y se
# MARCA como inferida: es una evidencia mas debil y el lector debe poder distinguirla.


def fecha_de_archivo(ruta):
    if not os.path.exists(ruta):
        return ''
    d = datetime.date.fromtimestamp(os.path.getmtime(ruta))
    return d.strftime('%d %b %Y')


def main():
    man = {r['clave_agrupacion']: r for r in
           csv.DictReader(open(OUT / 'manifiesto_final_casos.csv', encoding='utf-8'))}
    ds = list(csv.DictReader(open(OUT / 'dataset_nivel_imagen.csv', encoding='utf-8')))

    n_attr = n_fecha = n_infer = 0
    with open(OUT / 'procedencia_imagenes.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in ds:
            m = man.get(r['clave_agrupacion'], {})
            cita = (m.get('cita') or '').strip()
            ok = bool(cita) and not cita.startswith('FALTA')
            f = RE_FECHA.search(cita)
            if f:
                fecha, origen = f.group(1), 'cita'
            else:
                fecha = fecha_de_archivo(r['ruta'])
                origen = 'fecha de descarga del archivo (inferida)' if fecha else ''
            n_attr += ok
            n_fecha += bool(f)
            n_infer += bool(fecha) and origen != 'cita'
            w.writerow(dict(
                archivo=r['archivo'], sha256=r['sha256'], clase=r['clase'], split=r['split'],
                clave_agrupacion=r['clave_agrupacion'], rID=r['rID'],
                numero_base=r['numero_base'], edad=m.get('edad', ''), sexo=m.get('sexo', ''),
                diagnostico=m.get('diagnostico', ''), cita=cita,
                atribucion_ok='SI' if ok else 'NO', licencia=LICENCIA,
                licencia_alcance='sitio (no por caso)',
                licencia_verificada='SI (2026-09-06)',
                politica_url=POLITICA, fecha_acceso=fecha,
                fecha_acceso_origen=origen,
                fuente='Radiopaedia.org', url_caso=m.get('url', ''), ruta=r['ruta']))

    print(f'  procedencia_imagenes.csv: {len(ds)} imagenes')
    print(f'  con atribucion de autor  : {n_attr}')
    print(f'  SIN atribucion           : {len(ds) - n_attr}')
    print(f'  fecha extraida de la cita: {n_fecha}')
    print(f'  fecha inferida del archivo: {n_infer}')
    print(f'  sin fecha                : {len(ds) - n_fecha - n_infer}')
    print(f'  licencia                 : {LICENCIA}  (verificada = NO en todas)')


if __name__ == '__main__':
    main()
