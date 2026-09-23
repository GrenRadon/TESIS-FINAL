"""Lector del manifiesto Radiopaedia + merge con la auditoría a nivel de caso."""
import zipfile, re, html, csv
from pathlib import Path

XLSX = Path("data/vf/Manifiesto_Datos_Radiopaedia.xlsx")
# El manifiesto usa dos formatos de cita segun cuando se capturo:
#   "...https://doi.org/10.53347/rID-39310"   y   "...Radiopaedia.org, rID: 172037"
RE_RID = re.compile(r"rID[-:\s]+(\d+)")

def _sheet(z, name, strs):
    xml = z.read(name).decode("utf8")
    out = []
    for rm in re.finditer(r'<row[^>]*r="(\d+)"[^>]*>(.*?)</row>', xml, re.S):
        cells = {}
        for cm in re.finditer(r'<c r="([A-Z]+)\d+"([^>]*)>(.*?)</c>', rm.group(2), re.S):
            col, attrs, body = cm.group(1), cm.group(2), cm.group(3)
            v = re.search(r'<v>(.*?)</v>', body, re.S)
            if v:
                cells[col] = strs[int(v.group(1))] if 't="s"' in attrs else v.group(1)
            else:
                t = re.search(r'<t[^>]*>(.*?)</t>', body, re.S)
                cells[col] = html.unescape(t.group(1)) if t else ""
        out.append(cells)
    return out

def leer_manifiesto(path=XLSX):
    z = zipfile.ZipFile(path)
    ss = z.read("xl/sharedStrings.xml").decode("utf8")
    strs = [html.unescape(re.sub(r'<[^>]+>', '', s)) for s in re.findall(r'<si>(.*?)</si>', ss, re.S)]
    ch = _sheet(z, "xl/worksheets/sheet2.xml", strs)
    no = _sheet(z, "xl/worksheets/sheet3.xml", strs)
    reg = {}
    for r in ch[1:]:
        n = r.get("A", "").strip()
        if not n.isdigit(): continue
        reg[("chiari", int(n))] = dict(
            imagen_num=int(n), clase="chiari", rID=r.get("B", "").strip(),
            clave_agrupacion=r.get("C", "").strip(), diagnostico=r.get("D", "").strip(),
            edad=r.get("E", "").strip(), sexo=r.get("F", "").strip(),
            sintoma=r.get("G", "").strip(), hallazgo=r.get("H", "").strip(),
            cita=r.get("I", "").strip(), url=r.get("J", "").strip())
    for r in no[1:]:
        n = r.get("A", "").strip()
        if not n.isdigit(): continue
        cita = r.get("G", "").strip()
        # La hoja Normal no tiene columna rID, pero la cita trae el DOI de Radiopaedia.
        m = RE_RID.search(cita)
        reg[("normal", int(n))] = dict(
            imagen_num=int(n), clase="normal", rID=m.group(1) if m else "",
            clave_agrupacion=r.get("B", "").strip(), diagnostico=r.get("C", "").strip(),
            edad=r.get("D", "").strip(), sexo=r.get("E", "").strip(),
            sintoma=r.get("F", "").strip(), hallazgo="",
            cita=cita, url=r.get("H", "").strip())
    return reg
