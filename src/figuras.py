"""Salida unica de figuras: PNG a 300 dpi para impresion + PDF vectorial.

Motivo: las figuras se guardaban con el dpi que cada script traia (140-160), suficiente en
pantalla e insuficiente para el documento impreso, donde se pide 300 dpi o mas. Ademas, en
las figuras con texto pequeno --curvas, diagramas de dispersion, matrices-- rasterizar
degrada la tipografia sin necesidad: matplotlib puede escribir PDF vectorial directamente,
y en ese formato el texto sigue siendo texto.

Las figuras cuyo contenido son imagenes medicas (muestras, Grad-CAM, chequeos) llevan el
raster dentro del PDF, pero sus rotulos siguen siendo vectoriales.

Uso en cada script:
    from figuras import guardar
    guardar(fig, FIG / 'nombre.png')
"""
from pathlib import Path

DPI = 300


def guardar(fig, ruta, dpi=DPI, pdf=True, **kw):
    """Escribe <ruta> en PNG a `dpi` y, salvo que se pida lo contrario, su gemelo en PDF."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    opciones = dict(bbox_inches='tight', facecolor='white')
    opciones.update(kw)
    fig.savefig(str(ruta), dpi=dpi, **opciones)
    salidas = [ruta]
    if pdf:
        p = ruta.with_suffix('.pdf')
        fig.savefig(str(p), **opciones)      # el PDF no lleva dpi: es vectorial
        salidas.append(p)
    return salidas
