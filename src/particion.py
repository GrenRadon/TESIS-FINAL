"""Paso 1-3: manifiesto final a nivel de caso, partición limpia por caso y verificación de fuga."""
import sys, csv, json, random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from manifiesto import leer_manifiesto

SEED = 42
# Archivos excluidos: byte-identicos a imagenes Chiari del mismo conjunto y confirmados
# por matchTemplate contra data/vf/raw/chiari (corr >= 0.9987) como recortes de una raw
# Chiari. Estaban colocados en val_externo/cropped/normal/ con nombre "normal_*".
EXCLUIDOS_SHA = {
    "6163ad6ffd63b87f": "normal_25a.jpeg == chiari_24a.jpeg (raw chiari_24a, corr 0.9987)",
    "66291783b27449e6": "normal_26a.jpg  == chiari_25a.jpg  (raw chiari_25a, corr 0.9998)",
    "f5871fe2194fe845": "normal_24a.png  == chiari_28a.png  (raw chiari_28a, corr 1.0000)",
}
# Casos excluidos por procedencia no documentable (M-01/C-10): sin fila en el manifiesto
# de origen y sin URL de caso recuperable, por lo que no puede acreditarse autoria ni licencia.
# normal_26 estuvo excluido por falta de procedencia; el 2026-09-05 se documento
# (rID 43937, meningioma frontal) y se sustituyo la imagen, por lo que vuelve a entrar.
EXCLUIDOS_CASO = {
    # Clase negativa redefinida como "sin hallazgos compatibles con MC-I": admite patologia
    # extra-fosa posterior, pero NO patologia EN la fosa posterior, que es la region evaluada.
    ("normal", 36): "patologia en fosa posterior: atrofia cerebelosa, hipocampal y de nucleos basales",
    ("normal", 38): "patologia en fosa posterior: meduloblastoma leptomeningeo primario",
    ("normal", 40): "patologia en fosa posterior: demencia britanica familiar con afectacion cerebelosa",
}


def _excluido(r):
    if (r["clase"], int(r["numero_base"])) in EXCLUIDOS_CASO:
        return True
    return r["clase"] == "normal" and any(r["sha256"].startswith(h) for h in EXCLUIDOS_SHA)
N_TEST_CHIARI, N_TEST_NORMAL = 15, 5
OUT = Path("resultados_v2")

def construir():
    reg = leer_manifiesto()
    casos = {(r["clase"], int(r["numero_base"])): r
             for r in csv.DictReader(open(OUT/"auditoria_nivel_caso.csv", encoding="utf-8"))}
    imgs = list(csv.DictReader(open(OUT/"auditoria_nivel_imagen.csv", encoding="utf-8")))

    # unión de claves: filas del manifiesto + casos detectados en disco
    todas = sorted(set(reg) | set(casos))
    manifiesto, sin_match, sin_fila = {}, [], []
    for (clase, num) in todas:
        m = reg.get((clase, num))
        a = casos.get((clase, num))
        if m is None:
            sin_fila.append((clase, num))
            m = dict(clave_agrupacion="", rID="", edad="", sexo="", diagnostico="",
                     sintoma="", cita="", url="")
        clave = m["clave_agrupacion"] or f"PROV_{clase}_{num}"
        origen = "manifiesto"
        if not m["clave_agrupacion"]:
            origen = "provisional"; sin_match.append((clase, num))
        manifiesto[(clase, num)] = dict(
            clave_agrupacion=clave, origen_clave=origen,
            rID=m["rID"] or (a["rID"] if a else f"{clase.upper()[:1]}{num:03d}"),
            rID_origen="manifiesto" if m["rID"] else "provisional",
            clase=clase, numero_base=num, edad=m["edad"], sexo=m["sexo"],
            diagnostico=m["diagnostico"], sintoma=m["sintoma"], cita=m["cita"], url=m["url"],
            n_train_orig=int(a["n_train"]) if a else 0,
            n_externo_orig=int(a["n_externo"]) if a else 0,
            fuga_orig=a["fuga"] if a else "SIN_IMAGENES")
    # imágenes por caso, deduplicadas por sha256 (los 27 archivos idénticos cuentan una sola vez)
    pool = defaultdict(dict)
    n_exc = 0
    for r in imgs:
        if _excluido(r):
            n_exc += 1
            continue
        pool[(r["clase"], int(r["numero_base"]))].setdefault(r["sha256"], r)
    print(f"  [limpieza] entradas 'normal' excluidas por ser Chiari mal etiquetadas: {n_exc}")
    for k, v in manifiesto.items():
        v["n_imagenes"] = len(pool.get(k, {}))
    return manifiesto, pool, sin_match, sin_fila

SELLADO = OUT / "test_sellado.json"


def particionar(manifiesto):
    """El test esta SELLADO en test_sellado.json y no se re-sortea.

    Si un caso se excluye del dataset y estaba en test, sale del test y no se sustituye:
    sustituirlo equivaldria a re-sortear el conjunto de prueba cada vez que cambian los datos.
    """
    import json as _json
    disponibles = {v["clave_agrupacion"] for v in manifiesto.values() if v["n_imagenes"] > 0}
    if SELLADO.exists():
        sello = _json.loads(SELLADO.read_text(encoding="utf-8"))
        test = set(sello["chiari"]) | set(sello["normal"])
        perdidas = sorted(test - disponibles)
        if perdidas:
            print(f"  [test sellado] casos del test ya no disponibles, se retiran: {perdidas}")
        return {c: ("test" if c in test else "desarrollo") for c in disponibles}
    rng = random.Random(SEED)
    split = {}
    for clase, n_test in (("chiari", N_TEST_CHIARI), ("normal", N_TEST_NORMAL)):
        claves = sorted({v["clave_agrupacion"] for v in manifiesto.values()
                         if v["clase"] == clase and v["n_imagenes"] > 0})
        rng.shuffle(claves)
        for c in claves[:n_test]: split[c] = "test"
        for c in claves[n_test:]: split[c] = "desarrollo"
    return split

def verificar(manifiesto, split, pool):
    dev = {c for c, s in split.items() if s == "desarrollo"}
    tst = {c for c, s in split.items() if s == "test"}
    inter = dev & tst
    # verificación por sha256: ninguna imagen física puede estar en ambos lados
    h_dev, h_tst = set(), set()
    for k, v in manifiesto.items():
        s = split.get(v["clave_agrupacion"])
        (h_dev if s == "desarrollo" else h_tst if s == "test" else set()).update(pool.get(k, {}).keys())
    return inter, h_dev & h_tst, dev, tst

if __name__ == "__main__":
    manifiesto, pool, sin_match, sin_fila = construir()
    split = particionar(manifiesto)
    inter, inter_h, dev, tst = verificar(manifiesto, split, pool)

    OUT.mkdir(exist_ok=True)
    cols = ["clave_agrupacion","origen_clave","rID","rID_origen","clase","numero_base","split",
            "n_imagenes","n_train_orig","n_externo_orig","fuga_orig","edad","sexo","diagnostico","sintoma","cita","url"]
    with open(OUT/"manifiesto_final_casos.csv","w",newline="",encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for k in sorted(manifiesto, key=lambda k:(k[0],k[1])):
            v = dict(manifiesto[k]); v["split"] = split.get(v["clave_agrupacion"], "sin_imagenes")
            w.writerow({c: v.get(c,"") for c in cols})
    with open(OUT/"dataset_nivel_imagen.csv","w",newline="",encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["archivo","ruta","clase","label","numero_base","clave_agrupacion","rID","split","sha256","carpeta_origen"])
        for k in sorted(pool, key=lambda k:(k[0],k[1])):
            v = manifiesto[k]; s = split.get(v["clave_agrupacion"], "sin_imagenes")
            for h, r in sorted(pool[k].items(), key=lambda x: x[1]["archivo"]):
                w.writerow([r["archivo"], r["ruta"], k[0], 1 if k[0]=="chiari" else 0, k[1],
                            v["clave_agrupacion"], v["rID"], s, h, r["carpeta"]])

    print("=== 1. MANIFIESTO FINAL A NIVEL DE CASO ===")
    for clase in ("chiari","normal"):
        vs = [v for v in manifiesto.values() if v["clase"]==clase]
        print(f"  {clase:7s}: {len(vs)} casos | claves del manifiesto: {sum(1 for v in vs if v['origen_clave']=='manifiesto')}"
              f" | provisionales: {sum(1 for v in vs if v['origen_clave']=='provisional')}"
              f" | rID real: {sum(1 for v in vs if v['rID_origen']=='manifiesto')}")
        print(f"           imágenes únicas (dedup sha256): {sum(v['n_imagenes'] for v in vs)}"
              f" | casos sin imagen: {sum(1 for v in vs if v['n_imagenes']==0)}")
    print("")
    print("  --- desajustes manifiesto <-> archivos ---")
    print(f"  archivos en disco SIN fila en el manifiesto : {sin_fila or 'ninguno'}")
    huerfanos = [(v["clase"], v["numero_base"], v["clave_agrupacion"]) for v in manifiesto.values() if v["n_imagenes"] == 0]
    print(f"  filas del manifiesto SIN archivo en disco   : {huerfanos or 'ninguna'}")

    print("\n=== 2. PARTICIÓN POR CASO (semilla 42) ===")
    for clase in ("chiari","normal"):
        for s in ("desarrollo","test"):
            vs = [v for v in manifiesto.values() if v["clase"]==clase and split.get(v["clave_agrupacion"])==s]
            print(f"  {clase:7s} {s:11s}: {len(vs):3d} casos | {sum(v['n_imagenes'] for v in vs):3d} imágenes")
    print("\n  claves de TEST reservadas:")
    for clase in ("chiari","normal"):
        cl = sorted(v["clave_agrupacion"] for v in manifiesto.values()
                    if v["clase"]==clase and split.get(v["clave_agrupacion"])=="test")
        print(f"    {clase}: {cl}")

    print("\n=== 3. VERIFICACIÓN OBLIGATORIA ===")
    print(f"  claves desarrollo: {len(dev)} | claves test: {len(tst)}")
    print(f"  intersección clave_agrupacion(desarrollo) ∩ clave_agrupacion(test) = {sorted(inter) if inter else 'VACÍA'}  -> {'FALLA' if inter else 'OK'}")
    print(f"  intersección sha256(desarrollo) ∩ sha256(test)                     = {len(inter_h)} archivos -> {'FALLA' if inter_h else 'OK'}")
    ok = not inter and not inter_h
    print(f"\n  RESULTADO: {'PASA — se puede entrenar' if ok else 'NO PASA — detener'}")
    json.dump({"interseccion_claves": sorted(inter), "interseccion_sha256": len(inter_h), "ok": ok},
              open(OUT/"verificacion_particion.json","w"), indent=2)
    sys.exit(0 if ok else 1)
