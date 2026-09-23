"""Regla de reparto PREESPECIFICADA para los casos nuevos (escrita antes de recibirlos).

Se fija aqui, con semilla, ANTES de ver ningun resultado del modelo sobre esas imagenes.
Registrarla por adelantado es lo que impide elegir a posteriori la particion que mejor luce
(la critica C-04 / M-05 del informe).

REGLAS
  R1. El test sellado de 20 casos (test_sellado.json) NO se toca: ni se amplia ni se sustituye.
  R2. El destino depende del numero de casos NORMAL nuevos, porque la clase negativa es el
      cuello de botella de todas las metricas:
        - normales nuevos <  UMBRAL_HOLDOUT  -> TODOS al desarrollo.
          Un holdout con menos de 10 negativos tiene un IC de ancho > 0,45: no informa,
          y resta datos al entrenamiento, que es donde hacen falta.
        - normales nuevos >= UMBRAL_HOLDOUT  -> se aparta un "holdout interno adicional"
          con OBJETIVO_NEG_HOLDOUT negativos y tantos positivos como haga falta para
          prevalencia ~50 %, mas defendible que el 75 % del test actual (M-03).
  R3. El reparto es aleatorio estratificado por clase, agrupado por caso (nunca por imagen),
      con SEED. Todas las imagenes de un caso van juntas.
  R4. El resultado se escribe en holdout_v2.json ANTES de entrenar, y no se vuelve a tocar.

NOMENCLATURA: al provenir de Radiopaedia, este conjunto es un HOLDOUT INTERNO ADICIONAL.
No es validacion externa (M-02) y no debe llamarse asi en el documento.
"""
import csv, json, random
from pathlib import Path

SEED = 42
UMBRAL_HOLDOUT = 15        # casos normal nuevos por debajo de los cuales no se aparta nada
OBJETIVO_NEG_HOLDOUT = 10  # negativos que tendria el holdout adicional si se crea
OUT = Path('resultados_v2')
SALIDA = OUT / 'holdout_v2.json'


def casos_nuevos(umbral_chiari=77, umbral_normal=27):
    """Casos presentes en el dataset cuya numeracion es posterior a la del corte."""
    ds = list(csv.DictReader(open(OUT / 'dataset_nivel_imagen.csv', encoding='utf-8')))
    nuevos = {'chiari': {}, 'normal': {}}
    for r in ds:
        n = int(r['numero_base'])
        lim = umbral_chiari if r['clase'] == 'chiari' else umbral_normal
        if n >= lim:
            nuevos[r['clase']].setdefault(r['clave_agrupacion'], []).append(r['archivo'])
    return nuevos


def repartir(nuevos):
    rng = random.Random(SEED)
    ch = sorted(nuevos['chiari'])
    no = sorted(nuevos['normal'])
    plan = dict(seed=SEED, umbral_holdout=UMBRAL_HOLDOUT,
                objetivo_neg_holdout=OBJETIVO_NEG_HOLDOUT,
                n_casos_nuevos=dict(chiari=len(ch), normal=len(no)),
                nomenclatura='holdout interno adicional (NO validacion externa, M-02)')

    if len(no) < UMBRAL_HOLDOUT:
        plan['decision'] = (f'{len(no)} casos normal nuevos < {UMBRAL_HOLDOUT}: '
                            f'todos al desarrollo, no se crea holdout adicional')
        plan['desarrollo'] = dict(chiari=ch, normal=no)
        plan['holdout_v2'] = dict(chiari=[], normal=[])
        return plan

    rng.shuffle(no)
    rng.shuffle(ch)
    h_no = sorted(no[:OBJETIVO_NEG_HOLDOUT])
    h_ch = sorted(ch[:min(len(ch), OBJETIVO_NEG_HOLDOUT)])  # prevalencia ~50 %
    plan['decision'] = (f'{len(no)} casos normal nuevos >= {UMBRAL_HOLDOUT}: se aparta un '
                        f'holdout adicional de {len(h_no)} negativos y {len(h_ch)} positivos')
    plan['holdout_v2'] = dict(chiari=h_ch, normal=h_no)
    plan['desarrollo'] = dict(chiari=sorted(set(ch) - set(h_ch)),
                              normal=sorted(set(no) - set(h_no)))
    return plan


def main():
    nuevos = casos_nuevos()
    plan = repartir(nuevos)
    print(f'  casos nuevos: {plan["n_casos_nuevos"]}')
    print(f'  decision    : {plan["decision"]}')
    print(f'  desarrollo  : {len(plan["desarrollo"]["chiari"])} chiari + '
          f'{len(plan["desarrollo"]["normal"])} normal')
    print(f'  holdout_v2  : {len(plan["holdout_v2"]["chiari"])} chiari + '
          f'{len(plan["holdout_v2"]["normal"])} normal')
    if SALIDA.exists():
        print(f'\n  {SALIDA} YA EXISTE: no se sobrescribe (el reparto ya estaba fijado).')
        return
    SALIDA.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n  reparto fijado en {SALIDA}')


if __name__ == '__main__':
    main()
