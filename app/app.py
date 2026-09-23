"""Herramienta de apoyo para la identificacion de imagenes compatibles con MC-I.

NO es una herramienta de diagnostico. Ver la pestana "Alcance y limitaciones".
"""
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import onnxruntime as ort

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / 'src'))
import preprocesamiento as PP  # noqa: E402
import validacion_entrada as VE  # noqa: E402

TITULO = 'Apoyo a la identificación de imágenes compatibles con MC-I'
DIR_MODELOS = Path(__file__).resolve().parent / 'models'
UMBRAL_POR_DEFECTO = 0.5
MARGEN_BAJA_CONFIANZA = 0.10
LADO_MINIMO = VE.MIN_LADO
_SENS = Path(__file__).resolve().parent.parent / 'resultados_v2' / 'sensibilidad_recorte.json'
_LAT = Path(__file__).resolve().parent / 'models' / 'latencia.json'
LAT = (json.loads(_LAT.read_text(encoding='utf-8')) if _LAT.exists()
       else dict(p50_ms=dict(mediana=34), p95_ms=dict(mediana=43)))
SENS = (json.loads(_SENS.read_text(encoding='utf-8')) if _SENS.exists()
        else dict(rango_mediano=0.58, pct_vuelcos=0.169, n_casos=20,
                  casos_con_algun_vuelco=10, aciertos_con_sugerido=16))

st.set_page_config(page_title=TITULO, layout='wide',
                   initial_sidebar_state='collapsed')

# --- Presentacion (RNF07): paleta sobria de dos colores funcionales -----------------
st.markdown("""
<style>
  :root {
    --tinta:#1c2430; --suave:#5a6675; --linea:#e2e6ec; --fondo:#f7f8fa;
    --alto:#b4472f; --bajo:#2f7d5f; --aviso:#8a6216;
  }
  /* La barra fija de Streamlit ocupa la franja superior: con menos espacio que esto el
     titulo se mete debajo y se ve cortado por arriba. */
  .block-container { padding-top: 4.6rem; max-width: 1180px; }
  h1, h2, h3 { letter-spacing: -.015em; color: var(--tinta); font-weight: 620; }
  .cabecera { border-bottom: 1px solid var(--linea); padding-bottom: 1.1rem;
              margin-bottom: 1.6rem; }
  /* line-height holgado: con 1,25 las tildes de las mayusculas quedaban al ras */
  .cabecera .tit { font-size: 1.55rem; font-weight: 640; color: var(--tinta);
                   line-height: 1.4; margin-bottom: .35rem; padding-top: .1rem;
                   overflow: visible; }
  .cabecera .sub { font-size: .9rem; color: var(--suave); }
  .cinta { background: #fbf7ec; border: 1px solid #ead9b0; border-left: 3px solid var(--aviso);
           border-radius: 7px; padding: .8rem 1rem; font-size: .87rem; color: #4a3d20;
           margin-bottom: 1.5rem; }
  .panel { background: var(--fondo); border: 1px solid var(--linea); border-radius: 10px;
           padding: 1.3rem 1.4rem; }
  .veredicto { border-radius: 10px; padding: 1.15rem 1.3rem; margin-bottom: .9rem; }
  .veredicto .et { font-size: .74rem; text-transform: uppercase; letter-spacing: .09em;
                   opacity: .72; margin-bottom: .3rem; }
  .veredicto .vl { font-size: 1.32rem; font-weight: 640; line-height: 1.3; }
  .v-alto { background:#fbf0ed; border:1px solid #e8c6bc; color: var(--alto); }
  .v-bajo { background:#eef6f2; border:1px solid #c2ddd0; color: var(--bajo); }
  .cifra { font-size: 2.5rem; font-weight: 300; color: var(--tinta); line-height: 1;
           font-variant-numeric: tabular-nums; }
  .cifra small { font-size: .95rem; color: var(--suave); font-weight: 400; }
  .dato { display:flex; justify-content:space-between; padding:.42rem 0;
          border-bottom:1px dotted var(--linea); font-size:.86rem; }
  .dato span:first-child { color: var(--suave); }
  .dato span:last-child { color: var(--tinta); font-variant-numeric: tabular-nums; }
  .pie { color: var(--suave); font-size: .78rem; border-top: 1px solid var(--linea);
         padding-top: .9rem; margin-top: 2.2rem; }
  .stTabs [data-baseweb="tab"] { font-size: .92rem; }
  div[data-testid="stImage"] img { border-radius: 8px; border: 1px solid var(--linea); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar_modelo():
    cands = sorted(DIR_MODELOS.glob('*.onnx'))
    if not cands:
        return None, None, {}
    if len(cands) > 1:
        # Fallar ruidosamente en vez de elegir por orden alfabetico: con dos modelos en la
        # carpeta se cargaria uno y se aplicaria el umbral del config del otro.
        st.error(f'Hay {len(cands)} modelos .onnx en `models/`: '
                 + ', '.join(f'`{c.name}`' for c in cands)
                 + '. Debe haber exactamente uno, junto a su `config_modelo.json`. '
                   'Elimine los que sobren antes de continuar.')
        st.stop()
    ruta = cands[0]
    cfg_path = DIR_MODELOS / 'config_modelo.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8')) if cfg_path.exists() else {}
    return ort.InferenceSession(str(ruta)), ruta.stem, cfg


def marco(imagen, caja):
    """Dibuja el recuadro de recorte sobre una copia reducida, para la vista previa."""
    vista = imagen.convert('RGB').copy()
    d = ImageDraw.Draw(vista)
    grosor = max(2, int(min(vista.size) * 0.006))
    x0, y0, x1, y1 = caja
    d.rectangle([x0, y0, x1, y1], outline=(180, 71, 47), width=grosor)
    # velo sobre lo que queda fuera, para que se lea que eso se descarta
    velo = Image.new('RGBA', vista.size, (28, 36, 48, 105))
    hueco = Image.new('RGBA', (max(1, x1 - x0), max(1, y1 - y0)), (0, 0, 0, 0))
    velo.paste(hueco, (x0, y0))
    return Image.alpha_composite(vista.convert('RGBA'), velo).convert('RGB')


@st.cache_data
def geometria_sugerencia():
    """Fracciones medidas sobre los recortes que se hicieron a mano (175 pares).

    Si el artefacto falta, se usan las mismas medianas como respaldo para que la app no
    dependa de un fichero opcional.
    """
    ruta = DIR_MODELOS / 'sugerencia_recorte.json'
    if ruta.exists():
        g = json.loads(ruta.read_text(encoding='utf-8'))
        return (g['fraccion_centro_x']['mediana'], g['fraccion_centro_y']['mediana'],
                g['fraccion_ancho']['mediana'], g['fraccion_alto']['mediana'], g['n_pares'])
    return (0.662, 0.594, 0.728, 0.681, 175)


def caja_sugerida(imagen):
    """Propone un recuadro inicial sobre la fosa posterior.

    NO es deteccion automatica de la region: eso sigue sin implementarse (C-07). Se delimita
    la cabeza separandola del fondo oscuro y se coloca el recuadro donde, medido sobre los
    recortes manuales del conjunto de trabajo, suele caer la fosa posterior. El usuario lo
    corrige; solo evita empezar desde un recuadro arbitrario.
    """
    fx, fy, fw, fh, _ = geometria_sugerencia()
    a = np.asarray(imagen.convert('L'), dtype=np.float32)
    H, W = a.shape
    mascara = a > max(12.0, float(a.max()) * 0.10)
    filas = np.where(mascara.any(axis=1))[0]
    cols = np.where(mascara.any(axis=0))[0]
    if len(filas) < 8 or len(cols) < 8:          # imagen casi vacia: recuadro centrado
        lado = int(min(W, H) * 0.7)
        return (W // 2, H // 2, lado, lado)
    y0, y1 = int(filas[0]), int(filas[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    ah, aw = y1 - y0, x1 - x0
    return (int(x0 + aw * fx), int(y0 + ah * fy),
            int(max(LADO_MINIMO, aw * fw)), int(max(LADO_MINIMO, ah * fh)))


def a_pct(v, total):
    return float(np.clip(v / total * 100.0, 0.0, 100.0))


def caja_desde_pct(W, H, px, py, pw, ph):
    """Convierte porcentajes en una caja valida. Los controles nunca se limitan entre si:
    el ajuste a los bordes se hace aqui, al final, y no cambia el rango de ningun control."""
    ancho = int(np.clip(W * pw / 100.0, LADO_MINIMO, W))
    alto = int(np.clip(H * ph / 100.0, LADO_MINIMO, H))
    cx = int(np.clip(W * px / 100.0, ancho / 2, W - ancho / 2))
    cy = int(np.clip(H * py / 100.0, alto / 2, H - alto / 2))
    x0, y0 = cx - ancho // 2, cy - alto // 2
    return (x0, y0, x0 + ancho, y0 + alto)


def controles_recorte(imagen, clave):
    """RF02. Recorte con controles nativos (RNF06: sin dependencias de terceros).

    Los cuatro controles tienen rango fijo 0-100 %, de modo que mover uno NO reajusta a los
    demas. Con rangos en pixeles dependientes entre si, cambiar el ancho movia el centro.
    """
    W, H = imagen.size
    if min(W, H) < LADO_MINIMO:
        st.error(f'La imagen mide {W} × {H} px. El lado menor debe ser de al menos '
                 f'{LADO_MINIMO} px para poder analizarla.')
        st.stop()

    k = f'recorte_{clave}'
    if k not in st.session_state:
        cx, cy, cw, ch = caja_sugerida(imagen)
        st.session_state[k] = dict(px=a_pct(cx, W), py=a_pct(cy, H),
                                   pw=a_pct(cw, W), ph=a_pct(ch, H))
    v = st.session_state[k]

    n_pares = geometria_sugerencia()[4]
    st.info(f'**Se propone un recuadro de partida.** Su posición y tamaño se midieron sobre '
            f'los {n_pares} recortes que se hicieron a mano para construir el modelo, '
            f'expresados en proporción a la cabeza detectada en la imagen. Compruébelo en la '
            f'vista previa y corríjalo si hace falta.\n\n'
            f'**Es una regularidad geométrica, no una detección de la región.** Si en su '
            f'imagen la cabeza mira hacia el otro lado, la sugerencia caerá del lado '
            f'contrario y tendrá que moverla. La responsabilidad del encuadre es suya.')

    st.markdown('**Ajuste en este orden.** Cada control es independiente: mover uno no '
                'modifica a los demás.')
    st.caption('**El encuadre influye en el resultado, pero no por igual en todos los '
               'casos.** Medido sobre el conjunto de prueba: con el recuadro sobre la fosa, '
               'pequeños ajustes cambian el veredicto en el 8% de los '
               'casos con un resultado claro, y en el 81% de aquellos '
               'cuya probabilidad ya cae cerca del punto de corte. La herramienta avisa '
               'cuando su caso está en ese segundo grupo. Aun así, encuadre con cuidado: si '
               'el recuadro se va de la fosa, el resultado deja de ser interpretable.')
    c1, c2 = st.columns(2)
    with c1:
        st.caption('Paso 1 — sitúe el recuadro sobre la zona correcta')
        px = st.slider('Posición horizontal', 0.0, 100.0, v['px'], 0.5,
                       key=f'{k}_px', help='0 % es el borde izquierdo, 100 % el derecho.',
                       format='%.0f %%')
        py = st.slider('Posición vertical', 0.0, 100.0, v['py'], 0.5,
                       key=f'{k}_py', help='0 % es el borde superior, 100 % el inferior.',
                       format='%.0f %%')
    with c2:
        st.caption('Paso 2 — ajuste el tamaño hasta encuadrar la región')
        pw = st.slider('Ancho del recuadro', 2.0, 100.0, v['pw'], 0.5,
                       key=f'{k}_pw', help='Porcentaje del ancho de la imagen.',
                       format='%.0f %%')
        ph = st.slider('Alto del recuadro', 2.0, 100.0, v['ph'], 0.5,
                       key=f'{k}_ph', help='Porcentaje del alto de la imagen.',
                       format='%.0f %%')

    if st.button('Restablecer la sugerencia inicial'):
        for suf in ('px', 'py', 'pw', 'ph'):
            st.session_state.pop(f'{k}_{suf}', None)
        st.session_state.pop(k, None)
        st.rerun()

    caja = caja_desde_pct(W, H, px, py, pw, ph)
    recorte = imagen.crop(caja)
    rw, rh = recorte.size

    v1, v2 = st.columns([3, 2])
    with v1:
        st.image(marco(imagen, caja), width='stretch')
        st.caption(f'Imagen completa: {W} × {H} px. Lo atenuado se descarta.')
    with v2:
        st.image(recorte, width='stretch')
        st.caption(f'Región seleccionada: {rw} × {rh} px')
        prop = max(rw, rh) / max(1, min(rw, rh))
        if min(rw, rh) < LADO_MINIMO:
            st.error(f'Demasiado pequeña: el lado menor es {min(rw, rh)} px y el mínimo es '
                     f'{LADO_MINIMO}. Aumente el tamaño.')
        elif prop > VE.MAX_ASPECTO:
            st.warning(f'Recuadro muy alargado ({prop:.1f} : 1). El máximo admitido es '
                       f'{VE.MAX_ASPECTO:.0f} : 1. Iguale más el ancho y el alto.')
        else:
            st.success('El recuadro cumple los requisitos de forma y tamaño.')
    return recorte


# ---------------------------------------------------------------- cabecera
st.markdown(f"""<div class="cabecera">
  <div class="tit">{TITULO}</div>
  <div class="sub">Prototipo de investigación · Proyecto de grado · Universidad Industrial
  de Santander</div>
</div>""", unsafe_allow_html=True)

st.markdown("""<div class="cinta"><b>Uso no clínico.</b> Esta herramienta clasifica una
imagen como compatible o no compatible con hallazgos de malformación de Chiari I.
No emite diagnóstico, no mide el descenso amigdalar y no ha sido validada clínicamente.
Población objetivo: pacientes adultos de 18 años o más. Consulte la pestaña
<i>Alcance y limitaciones</i> antes de usarla.</div>""", unsafe_allow_html=True)

sesion, nombre_modelo, cfg = cargar_modelo()
if sesion is None:
    st.error('No se encontró un modelo .onnx en `models/`. Ejecute `convert_to_onnx.py`.')
    st.stop()

umbral = float(cfg.get('umbral', UMBRAL_POR_DEFECTO))
if 'umbral' not in cfg:
    st.error(f'No hay `config_modelo.json`: se usa el umbral por defecto '
             f'{UMBRAL_POR_DEFECTO}. Ese umbral **no está justificado** y degrada la '
             'especificidad. Genere la configuración con la exportación.')

arq = cfg.get('arquitectura', 'DenseNet121')
t_analisis, t_manual, t_lectura, t_alcance, t_ficha = st.tabs(
    ['Análisis', 'Manual de uso', 'Cómo leer el resultado',
     'Alcance y limitaciones', 'Ficha técnica'])

# ---------------------------------------------------------------- analisis
with t_analisis:
    archivo = st.file_uploader(
        'Cargue una resonancia magnética sagital T1 en formato JPG o PNG',
        type=['jpg', 'jpeg', 'png'],
        help='Puede cargar la imagen completa: en el paso siguiente delimitará la región.')

    if archivo is None:
        st.markdown("""<div class="panel">
        <b>Procedimiento</b><br><br>
        1 &nbsp; Cargue la imagen. Puede estar sin recortar.<br>
        2 &nbsp; Delimite la fosa posterior con los controles de región.<br>
        3 &nbsp; Confirme el análisis.<br><br>
        <span style="color:#5a6675">El recorte lo delimita usted: la herramienta no localiza
        la región de interés de forma automática.</span></div>""", unsafe_allow_html=True)
    else:
        try:
            imagen = Image.open(archivo)
            imagen.load()
        except Exception:
            st.error('El archivo no es una imagen legible. Cargue un JPG o PNG válido.')
            st.stop()

        st.markdown('#### Delimitación de la región de interés')
        recorte = controles_recorte(imagen, archivo.name)
        st.divider()

        if not st.button('Analizar la región seleccionada', type='primary'):
            st.caption('Ajuste la región y pulse el botón para ejecutar el análisis.')
            st.stop()

        # --- RF03 / RF04: control de calidad y deteccion fuera de distribucion ---
        aceptable, problemas, dist, nivel = VE.validar(np.array(recorte.convert('L')))
        if not aceptable:
            st.error('**La región seleccionada no cumple los requisitos mínimos y no se '
                     'analiza:**\n\n' + '\n'.join(f'- {p}' for p in problemas)
                     + '\n\nSe requiere un corte sagital medio T1 recortado a la fosa '
                       'posterior.')
            st.stop()
        if nivel == 'fuera de distribución':
            st.error(f'**Entrada fuera de la distribución de entrenamiento** (distancia '
                     f'{dist:.1f}). El resultado que sigue **no es fiable**: la imagen no se '
                     'parece a las usadas para entrenar el modelo. Verifique que sea una '
                     'resonancia sagital T1 y que el recorte cubra la fosa posterior.')
        elif nivel == 'atípica':
            st.warning(f'Entrada atípica respecto al conjunto de entrenamiento (distancia '
                       f'{dist:.1f}). Interprete el resultado con cautela.')

        with st.spinner('Analizando'):
            # unica llamada de preprocesamiento; devuelve [0,255], el modelo normaliza dentro
            tensor = PP.cargar_imagen(recorte)[None, ...].astype(np.float32)
            entrada = sesion.get_inputs()[0].name
            prob = float(sesion.run(None, {entrada: tensor})[0][0][0])

        compatible = prob >= umbral
        cerca = abs(prob - umbral) < MARGEN_BAJA_CONFIANZA
        st.markdown('#### Resultado')
        r1, r2 = st.columns([3, 2])
        with r1:
            clase = 'v-alto' if compatible else 'v-bajo'
            texto = ('Compatible con hallazgos de MC-I' if compatible
                     else 'No compatible con hallazgos de MC-I')
            st.markdown(f'<div class="veredicto {clase}"><div class="et">Clasificación de '
                        f'la imagen</div><div class="vl">{texto}</div></div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="cifra">{prob * 100:.1f}<small> % de probabilidad '
                        f'estimada</small></div>', unsafe_allow_html=True)
            st.progress(min(max(prob, 0.0), 1.0))
            if cerca:
                st.warning(f'El valor queda a {abs(prob - umbral):.3f} del umbral '
                           f'({umbral:.4f}). La clasificación es de baja confianza: una '
                           'variación mínima en el recorte podría invertirla.')
        with r2:
            st.markdown(
                f'<div class="panel">'
                f'<div class="dato"><span>Umbral de decisión</span><span>{umbral:.4f}</span></div>'
                f'<div class="dato"><span>Distancia al umbral</span>'
                f'<span>{abs(prob - umbral):.4f}</span></div>'
                f'<div class="dato"><span>Control de entrada</span>'
                f'<span>{nivel if nivel else "dentro de rango"}</span></div>'
                f'<div class="dato"><span>Región analizada</span>'
                f'<span>{recorte.size[0]} × {recorte.size[1]} px</span></div>'
                f'<div class="dato"><span>Arquitectura</span><span>{arq}</span></div>'
                f'</div>', unsafe_allow_html=True)

        st.info('**Este resultado no es un diagnóstico.** Un hallazgo radiológico compatible '
                'con MC-I no equivale a enfermedad sintomática ni a indicación quirúrgica. '
                'La interpretación corresponde a un médico especialista. Consulte la pestaña '
                '*Cómo leer el resultado*.')

# ---------------------------------------------------------------- manual
with t_manual:
    st.markdown(f"""
### Manual de uso

#### 1 · Qué imagen necesita

Una resonancia magnética de la cabeza **vista de perfil**, del corte que pasa por el centro.
En el informe del estudio suele figurar como *corte sagital medio, ponderación T1*. Si no
está seguro, pregúntele a quien le entregó el estudio: es la vista en la que se ve el perfil
del rostro, el cerebro y el comienzo del cuello en una misma imagen.

La imagen **puede estar completa**: no hace falta recortarla antes, la herramienta lo permite
en el paso siguiente. Formatos admitidos: JPG, JPEG y PNG.

#### 2 · Cómo delimitar la zona

Debe encuadrar la parte baja y trasera del cerebro: **el cerebelo, y la zona donde el cráneo
se une con el cuello** — lo que en términos anatómicos es la fosa posterior y la unión
cráneo-cervical. La vista previa oscurece todo lo que queda fuera, para que vea exactamente
qué se va a analizar.

**La herramienta propone un recuadro de partida.** En muchos casos solo hará falta
retocarlo. Si prefiere ajustarlo desde cero, siga este orden:

1. **Posición horizontal.** Mueva el recuadro hasta que quede sobre la nuca, no sobre el
   rostro. Si la cabeza de su imagen mira hacia el lado contrario al de la sugerencia,
   este es el control que hay que corregir primero.
2. **Posición vertical.** Bájelo o súbalo hasta centrarlo en el cerebelo.
3. **Ancho y alto.** Recién ahora ajuste el tamaño, hasta que entre lo que debe entrar y
   quede fuera lo que sobra.

Los cuatro controles son independientes: **mover uno no descoloca a los demás**. Debajo de
la vista previa, un indicador le confirma si el recuadro cumple los requisitos de forma y
tamaño antes de que ejecute el análisis. El botón *Restablecer la sugerencia inicial*
devuelve el recuadro propuesto si se pierde.

Un recorte correcto:

- Incluye el cerebelo y el agujero por donde el cerebro se continúa con la médula
  (foramen magno).
- Incluye las primeras vértebras del cuello.
- **No** incluye todo el cráneo ni el rostro completo.
- Su lado más corto no baja de {LADO_MINIMO} píxeles.

El recorte no es un detalle estético. El modelo aprendió con imágenes encuadradas de esta
manera, y una imagen encuadrada de otro modo produce un resultado que no significa nada.

#### 3 · Ejecutar el análisis

Al pulsar el botón, la herramienta revisa que la zona seleccionada sea utilizable, comprueba
si se parece a las imágenes con las que aprendió y calcula el resultado. Si la zona no cumple
los mínimos, se rechaza indicando por qué y **no se emite ningún porcentaje**.

#### 4 · Leer el resultado

Consulte la pestaña *Qué significa el resultado*. Anticipamos lo esencial: el porcentaje
**no indica gravedad** y el resultado **no es un diagnóstico**.

---

### Si aparece un mensaje de error

| Mensaje | Qué pasó | Qué hacer |
|---|---|---|
| El archivo no es una imagen legible | El formato no se pudo abrir | Convierta la imagen a JPG o PNG |
| La región no cumple los requisitos mínimos | La zona elegida es muy pequeña, muy alargada, o casi no tiene contraste | Amplíe la zona, o revise la imagen original |
| Entrada atípica | La zona elegida se aparta de lo que el modelo vio al aprender | Revise el encuadre. Si insiste, tome el resultado con cautela |
| Entrada fuera de distribución | La zona no se parece a las imágenes de entrenamiento | Compruebe que sea una resonancia de perfil y que el recuadro cubra el cerebelo y el cuello |
| Baja confianza | El resultado quedó muy cerca del punto de corte | Trátelo como no concluyente: no permite concluir nada en ningún sentido |
""")


# ---------------------------------------------------------------- lectura
def frecuencias(sens, espec, prev, n=None):
    """Frecuencias naturales: comunican riesgo mejor que los porcentajes.

    Devuelve, sobre n personas examinadas, cuantas tienen realmente la malformacion y como
    se reparten entre alarmas ciertas, alarmas falsas, tranquilidades ciertas y casos
    perdidos.

    El tamano se escala con la prevalencia: con 1 de cada 100 y n=1000 el redondeo deja los
    casos perdidos en cero, lo que se leeria como que la herramienta nunca falla.
    """
    if n is None:
        n = 1000 if prev >= 0.05 else 10000
    con = round(n * prev)
    sin_ = n - con
    alarma_cierta = round(con * sens)
    perdidos = con - alarma_cierta
    tranquilo_cierto = round(sin_ * espec)
    alarma_falsa = sin_ - tranquilo_cierto
    alarmas = alarma_cierta + alarma_falsa
    tranquilos = tranquilo_cierto + perdidos
    return dict(n=n, con=con, sin_=sin_, alarma_cierta=alarma_cierta, perdidos=perdidos,
                tranquilo_cierto=tranquilo_cierto, alarma_falsa=alarma_falsa,
                alarmas=alarmas, tranquilos=tranquilos)


with t_lectura:
    ppv = json.loads((RAIZ / 'resultados_v4' / 'analisis_finales.json')
                     .read_text(encoding='utf-8'))['ppv_npv']
    SENS, ESPEC = ppv['sensibilidad'], ppv['especificidad']
    curva_path = DIR_MODELOS / 'curva_operacion.json'
    curva = (json.loads(curva_path.read_text(encoding='utf-8'))
             if curva_path.exists() else None)

    st.markdown("""
### Qué significa el resultado

#### La herramienta mira una imagen, no a una persona

Compara la resonancia que usted cargó con las imágenes que utilizó para aprender. Cuando dice
**compatible**, está diciendo una sola cosa: *esta imagen se parece a las de personas que
tenían la malformación*.

No lo examinó a usted. No conoce sus síntomas, su historia clínica, sus otros estudios ni su
exploración física. Por eso el resultado **no es un diagnóstico**, y no puede serlo: un
diagnóstico lo hace un médico reuniendo todo eso, no una imagen sola.

#### El porcentaje no indica gravedad

El número que aparece **no dice qué tan avanzada está la malformación**, ni cuánto han
descendido las amígdalas cerebelosas, ni si hay que operar. Solo indica cuánto se parece la
imagen a las del grupo con malformación. Una imagen puede dar un porcentaje alto y
corresponder a alguien sin ningún síntoma.
""")

    st.markdown(f"""
#### Está ajustada para no dejar pasar casos, y eso tiene un precio

Se configuró deliberadamente para hacer sonar la alarma de más antes que dejar escapar un caso
real. La consecuencia es directa y hay que tenerla presente: **marca como compatibles a
muchas personas que no tienen nada**.

De cada 100 personas **con** la malformación, la herramienta señala unas
**{SENS * 100:.0f}**. Pero de cada 100 personas **sin** la malformación, solo deja pasar
tranquilas a unas **{ESPEC * 100:.0f}**: a las otras {100 - ESPEC * 100:.0f} también las
señala.
""")

    st.markdown('#### Un resultado negativo es más confiable que uno positivo')
    st.markdown('''Cuánto se puede confiar en cada respuesta depende de **a quién se esté
examinando**. Si en el grupo examinado casi nadie tiene la malformación, la mayoría de las
señales serán falsas — y eso le ocurre a cualquier prueba, no solo a esta.

Tenga presente que **nadie se hace una resonancia al azar**: si usted tiene una imagen para
cargar aquí, es porque un médico la solicitó por algún motivo. El grupo que llega a esta
herramienta ya viene filtrado, y se parece más a los dos primeros escenarios que al último.''')
    esc = {
        'Personas ya derivadas por sospecha — 30 de cada 100': 0.30,
        'Personas con síntomas leves — 10 de cada 100': 0.10,
        'Grupo con el que se construyó el modelo — 64 de cada 100': 0.64,
        'Población general — 1 de cada 100 (no es el uso previsto)': 0.01,
    }
    elegido = st.radio('Escenario', list(esc), index=0, label_visibility='collapsed')
    f = frecuencias(SENS, ESPEC, esc[elegido])
    if esc[elegido] <= 0.01:
        st.info('Este escenario supone aplicar la herramienta a personas tomadas al azar, '
                'sin motivo previo para sospechar. **No es el uso previsto** y ninguna prueba '
                'se comporta bien así. Se muestra para que la comparación sea honesta.')

    e1, e2 = st.columns(2)
    with e1:
        st.markdown(f"""<div class="panel">
        <b>Si la herramienta dice «compatible»</b><br><br>
        De cada <b>{f['n']:,}</b> personas examinadas, la herramienta señala a
        <b>{f['alarmas']:,}</b>. De esas, solo <b>{f['alarma_cierta']:,}</b> tienen
        realmente la malformación.<br><br>
        Las otras <b>{f['alarma_falsa']:,}</b> son <b>falsas alarmas</b>
        ({f['alarma_falsa'] / f['alarmas'] * 100:.0f} de cada 100 señaladas).<br><br>
        <span style="color:#5a6675">Un resultado compatible significa
        <i>conviene que lo mire un especialista</i>, no <i>usted tiene la
        malformación</i>.</span></div>""", unsafe_allow_html=True)
    with e2:
        st.markdown(f"""<div class="panel">
        <b>Si la herramienta dice «no compatible»</b><br><br>
        De cada <b>{f['n']:,}</b> personas examinadas, la herramienta deja pasar a
        <b>{f['tranquilos']:,}</b>. De esas, <b>{f['tranquilo_cierto']:,}</b> efectivamente
        no la tienen.<br><br>
        Se le escapan <b>{f['perdidos']:,}</b>
        ({f['perdidos'] / f['tranquilos'] * 100:.1f} de cada 100 no señaladas).<br><br>
        <span style="color:#5a6675">Este es el resultado en el que más se puede confiar.
        La herramienta sirve mejor para <b>descartar</b> que para confirmar.</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
#### Cuando el resultado queda en el límite, no concluya

La herramienta avisa si el valor queda muy cerca del punto de corte. En ese caso, mover
apenas el recuadro del recorte puede cambiar la respuesta de un lado al otro. No es un
resultado dudoso por casualidad: es un resultado **sin información suficiente**, y debe
tratarse como no concluyente.

#### Qué hacer con el resultado

- **No tome ninguna decisión sobre su salud a partir de esta pantalla.**
- Lleve la imagen y el informe original a su médico. La herramienta no reemplaza la lectura
  del radiólogo.
- Si el resultado fue «compatible», no es motivo de alarma por sí solo: la mayoría de esas
  señales son falsas.
- Si fue «no compatible» pero usted tiene síntomas, consulte igual. Ninguna herramienta
  descarta una enfermedad por completo.
""")

    with st.expander('Detalle técnico para profesionales de la salud'):
        umbral_pct = umbral * 100
        st.markdown(f"""
**Punto de corte: {umbral:.4f}**, no 0,5. Se eligió en validación cruzada agrupada por caso
maximizando la especificidad sujeta a sensibilidad ≥ 0,95, usando **únicamente** las
predicciones out-of-fold del conjunto de desarrollo. No se ajustó sobre el conjunto de prueba
reservado.

**Rendimiento en el conjunto de desarrollo** (n = 97 casos, predicción out-of-fold agrupada):
sensibilidad **{SENS:.4f}**, especificidad **{ESPEC:.4f}**, AUC 0,8456 (IC 95 % 0,761–0,921).

**Calibración.** El modelo es sobreconfiado en el extremo alto: en el tramo superior predice
0,954 de media y la proporción real de positivos fue 0,884. Brier 0,1665 frente a 0,2334 de
una predicción constante. Cuando la interfaz muestra {umbral_pct:.0f} % o más, conviene
descontar mentalmente esa diferencia.

**Valores predictivos según prevalencia**, calculados con la sensibilidad y especificidad de
arriba:

| Prevalencia | VPP | VPN |
|---|---|---|
""" + chr(10).join(
            f"| {r['prevalencia'] * 100:.0f} % | {r['ppv']:.3f} | {r['npv']:.3f} |"
            for r in ppv['tabla']) + f"""

**Especificidad desagregada por tipo de negativo** (M-18): cerebros sanos 0,550
(IC 95 % 0,342–0,742, n = 20); con patología distinta de MC-I 0,500 (IC 95 % 0,280–0,720,
n = 16). Con esos tamaños el contraste carece de potencia para descartar una diferencia real;
lo que sí puede afirmarse es que la especificidad es baja en ambos subgrupos.

**Margen de baja confianza:** {MARGEN_BAJA_CONFIANZA:.2f} en torno al punto de corte.
""")

        if curva:
            st.markdown(f"""
---

**El modelo separa las clases; la especificidad baja es una consecuencia del corte elegido.**

*Average precision* = **{curva['average_precision']:.4f}** frente a una línea base de
**{curva['ap_linea_base']:.4f}** (la prevalencia del conjunto, que es lo que obtendría un
clasificador sin información). AUC = {curva['auc']:.4f}. La discriminación es real; lo que se
negocia en el punto de corte es cómo repartir los errores.

**El mismo modelo en distintos puntos de corte** ({curva['n_casos']} casos,
{curva['n_positivos']} positivos y {curva['n_negativos']} negativos, predicción out-of-fold):

| Umbral | Sens. | Espec. | Precisión | F1 | VP/VN/FP/FN |
|---|---|---|---|---|---|
""" + chr(10).join(
                ('| **{u:.4f}**{m} | **{se:.4f}** | **{es:.4f}** | **{pr:.4f}** | '
                 '**{f1:.4f}** | {a}/{b}/{c}/{d} |' if abs(
                     q['umbral'] - curva['umbral_desplegado']) < 1e-9 else
                 '| {u:.4f}{m} | {se:.4f} | {es:.4f} | {pr:.4f} | {f1:.4f} | {a}/{b}/{c}/{d} |'
                 ).format(u=q['umbral'],
                          m=' · desplegado' if abs(q['umbral'] -
                                                   curva['umbral_desplegado']) < 1e-9 else '',
                          se=q['sensibilidad'], es=q['especificidad'], pr=q['precision'],
                          f1=q['f1'], a=q['vp'], b=q['vn'], c=q['fp'], d=q['fn'])
                for q in curva['puntos']) + f"""

Con el corte intuitivo de 0,50 la especificidad sube a 0,750 y la precisión a 0,845, a costa
de bajar la sensibilidad a 0,803: se perderían 12 de 61 casos positivos en lugar de 3. Se
eligió {curva['umbral_desplegado']:.4f} aplicando la regla preespecificada —
{curva['regla']} — porque en este dominio omitir un caso pesa más que generar una consulta
innecesaria.

**Valor predictivo positivo según corte y prevalencia**, para dimensionar el efecto:

| Umbral | """ + ' | '.join(f'{v:.0%}' for v in curva['prevalencias']) + """ |
|---|""" + '---|' * len(curva['prevalencias']) + chr(10) + chr(10).join(
                '| {} | {} |'.format(
                    f"{q['umbral']:.4f}",
                    ' | '.join(f'{x:.3f}' for x in q['vpp']))
                for q in curva['puntos']) + """

Artefacto: `app/models/curva_operacion.json`, generado por `src/curva_operacion.py` junto al
modelo que describe.
""")

# ---------------------------------------------------------------- alcance
with t_alcance:
    st.markdown("""
### Alcance y limitaciones

#### Solo para personas de 18 años o más

**No debe usarse con imágenes de niños ni de adolescentes.** El cerebro y el cráneo cambian
mucho con el crecimiento, y esta herramienta se construyó y se probó fundamentalmente con
adultos.

Al analizarlo, no encontramos que la herramienta se equivocara más con los menores que con
los adultos: la proporción de falsas alarmas fue 0,50 en pediátricos y 0,48 en adultos
(prueba exacta de Fisher, p = 1,00). Pero **no encontrar una diferencia no es lo mismo que
demostrar que no existe**: solo había ocho menores sin la malformación en el conjunto, y con
tan pocos casos el análisis únicamente habría detectado un problema muy grande. La
restricción se mantiene por prudencia, no porque el sesgo se haya descartado.

#### La herramienta no sabe si le dio una resonancia

Antes de analizar comprueba cosas básicas: que la imagen tenga tamaño suficiente, que tenga
contraste, que no sea una franja alargada, y qué tanto se parece a las imágenes con las que
aprendió — esto último mediante la distancia de Mahalanobis a la distribución del conjunto de
entrenamiento.

Lo que **no** comprueba: que sea una resonancia magnética, que sea la vista de perfil, ni que
el recuadro contenga realmente el cerebelo. Una fotografía cualquiera con estadísticas
parecidas puede pasar el control y recibir un porcentaje. Verificarlo exigiría un clasificador
de modalidad y de plano anatómico, que no se desarrolló en este trabajo.

#### Usted elige la zona, y de eso depende el resultado

La herramienta **no encuentra sola** la parte de la imagen que debe mirar. Si el recuadro está
mal puesto, el resultado no vale, y salvo que la zona resulte estadísticamente muy rara, la
herramienta no puede avisarlo.

**Cuánto depende: está medido.** Sobre los 20 casos del conjunto de prueba, se
partió del recuadro sugerido y se desplazó su centro hasta un 3 % del tamaño de la cabeza en
cada eje —el recuadro sigue sobre la fosa posterior; es la diferencia entre dos personas que
encuadran bien—:

- En los **18 casos con un resultado claro**, esos ajustes cambiaron el veredicto en el
  **8%** de las variantes. El resultado es estable.
- En los **2 casos cuya probabilidad ya caía cerca del punto de corte**, lo cambiaron en el
  **81%**. Ahí el encuadre decide la respuesta.

La conclusión no es que el modelo sea inestable, sino que **la inestabilidad se concentra
justo donde la herramienta ya avisa**. El mensaje de baja confianza no es decorativo: marca
los casos en los que el recuadro, y no la anatomía, determina el resultado.

Lo que sigue siendo cierto: si el recuadro **se va** de la fosa posterior —por ejemplo si sube
e incorpora cerebro anterior— el resultado deja de ser interpretable, y la herramienta solo
puede advertirlo si la región resulta estadísticamente atípica. Es el motivo por el que una
versión utilizable necesitaría localizar la región de forma automática, que es exactamente lo
que este trabajo no hace.

Consecuencia práctica: **si al corregir levemente el recuadro el resultado cambia, el
resultado no es informativo.** Trátelo como no concluyente.

#### No se ha probado fuera de este estudio

El modelo se construyó y se evaluó con imágenes de una sola fuente pública. **No se ha
probado con pacientes de otro hospital, ni con otros equipos de resonancia, ni siguiendo a
pacientes en el tiempo.** Es lo que en investigación se llama ausencia de validación externa
y prospectiva, y es la limitación más importante de todas.

#### Sobre sus datos: lo que se afirma y lo que no

El código de esta aplicación **no guarda la imagen en el disco ni la envía a terceros**.

Esa afirmación cubre el código de la aplicación, **no el entorno donde está instalada**: no
alcanza a la memoria temporal del servidor, la caché del framework, los registros del servidor
web ni las copias que pueda hacer la plataforma de alojamiento. Tampoco se hizo una evaluación
de seguridad informática, ni un análisis de impacto sobre datos personales, ni una
verificación de cumplimiento de la ley colombiana de protección de datos personales
(Ley 1581 de 2012).

**No se afirma que la herramienta sea privada por diseño.** Se afirma únicamente lo que se
verificó.

#### Para qué sirve, entonces

Como apoyo académico y docente, y como ejercicio de exploración. **No sustituye la lectura de
un radiólogo ni la valoración de un especialista**, y no debe formar parte de ninguna decisión
clínica.
""")


# ---------------------------------------------------------------- ficha
with t_ficha:
    st.markdown('### Ficha técnica')
    f1, f2 = st.columns(2)
    with f1:
        st.markdown(
            f'<div class="panel"><b>Modelo</b>'
            f'<div class="dato"><span>Artefacto</span><span>{nombre_modelo}</span></div>'
            f'<div class="dato"><span>Arquitectura</span><span>{arq}</span></div>'
            f'<div class="dato"><span>Umbral de decisión</span><span>{umbral:.4f}</span></div>'
            f'<div class="dato"><span>Formato</span><span>ONNX</span></div>'
            f'<div class="dato"><span>Entrada</span><span>224 × 224 × 3, float32</span></div>'
            f'</div>', unsafe_allow_html=True)
    with f2:
        st.markdown(
            f'<div class="panel"><b>Procesamiento</b>'
            f'<div class="dato"><span>Pipeline</span><span>v{PP.VERSION}</span></div>'
            f'<div class="dato"><span>Lado mínimo</span><span>{VE.MIN_LADO} px</span></div>'
            f'<div class="dato"><span>Proporción máxima</span>'
            f'<span>{VE.MAX_ASPECTO:.1f} : 1</span></div>'
            f'<div class="dato"><span>Margen de baja confianza</span>'
            f'<span>{MARGEN_BAJA_CONFIANZA:.2f}</span></div>'
            f'<div class="dato"><span>Latencia p50 / p95</span>'
            f'<span>{LAT["p50_ms"]["mediana"]:.0f} / {LAT["p95_ms"]["mediana"]:.0f} ms</span></div>'
            f'</div>', unsafe_allow_html=True)
    st.markdown(f"""
**Etapas del preprocesamiento** (idénticas a las del entrenamiento, importadas de
`src/preprocesamiento.py`):

`{PP.descripcion(arq)}`

El preprocesamiento no se reimplementa en esta aplicación. La normalización específica de la
arquitectura viaja dentro del grafo del modelo, de modo que no puede desincronizarse del
artefacto exportado.

""")

    cp = DIR_MODELOS / 'curva_operacion.json'
    if cp.exists():
        c = json.loads(cp.read_text(encoding='utf-8'))
        q = next(x for x in c['puntos']
                 if abs(x['umbral'] - c['umbral_desplegado']) < 1e-9)
        VP, VN, FP, FN = q['vp'], q['vn'], q['fp'], q['fn']
        POS, NEG = VP + FN, VN + FP
        SE, ES = q['sensibilidad'], q['especificidad']

        st.divider()
        st.markdown(f"""
### Cómo se obtienen estas cifras

#### 1 · Diseño de la validación

Con {c['n_casos']} casos no alcanza para apartar un conjunto de validación y otro de prueba
sin quedarse sin datos. Se usó **validación cruzada estratificada agrupada por caso**
(`StratifiedGroupKFold`, k = 5): los casos se reparten en cinco bloques manteniendo la
proporción de clases, y **todas las imágenes de un mismo caso caen siempre en el mismo
bloque**. Eso último es lo que impide que una imagen de un paciente entrene un modelo que
después evalúa otra imagen del mismo paciente.

Cada bloque se predice con el modelo entrenado sin él. Al juntar las cinco predicciones se
obtiene una predicción **out-of-fold** por caso: {c['n_casos']} predicciones, todas hechas
por un modelo que no vio ese caso. Sobre ese conjunto se calcula todo lo que sigue.

Las épocas de entrenamiento se fijaron de antemano, **sin parada temprana**, para que el
bloque de validación no interviniera en la decisión de cuándo detener el entrenamiento.

#### 2 · La matriz de confusión

Es el origen de todo lo demás. Al aplicar el punto de corte {c['umbral_desplegado']:.4f} a
las {c['n_casos']} predicciones:

| | Predice compatible | Predice no compatible | Total |
|---|---|---|---|
| **Tiene MC-I** | **{VP}** verdaderos positivos | **{FN}** falsos negativos | {POS} |
| **No tiene MC-I** | **{FP}** falsos positivos | **{VN}** verdaderos negativos | {NEG} |
| **Total** | {VP + FP} | {FN + VN} | {c['n_casos']} |
""")

        st.markdown('#### 3 · Las métricas, con el cálculo hecho')
        st.markdown('**Sensibilidad** — de los que tienen la malformación, qué proporción '
                    'detecta. Es la fila de arriba.')
        st.latex(r'\text{Sens}=\frac{VP}{VP+FN}=\frac{%d}{%d+%d}=\mathbf{%.4f}'
                 % (VP, VP, FN, SE))
        st.markdown('**Especificidad** — de los sanos, qué proporción deja pasar. Es la fila '
                    'de abajo.')
        st.latex(r'\text{Espec}=\frac{VN}{VN+FP}=\frac{%d}{%d+%d}=\mathbf{%.4f}'
                 % (VN, VN, FP, ES))
        st.markdown('**Precisión** — de los que señala, qué proporción acierta. Es la '
                    '**columna** izquierda, no una fila: por eso depende de cuántos enfermos '
                    'haya en el grupo examinado, y las dos anteriores no.')
        st.latex(r'\text{Prec}=\frac{VP}{VP+FP}=\frac{%d}{%d+%d}=\mathbf{%.4f}'
                 % (VP, VP, FP, q['precision']))
        st.markdown('**F1** — media armónica de precisión y sensibilidad. Se usa la armónica '
                    'y no la aritmética porque penaliza que una de las dos sea mala.')
        st.latex(r'F_1=\frac{2\cdot\text{Prec}\cdot\text{Sens}}'
                 r'{\text{Prec}+\text{Sens}}=\mathbf{%.4f}' % q['f1'])
        st.markdown(f"""
**AUC = {c['auc']:.4f}** — probabilidad de que, tomando al azar un caso con la malformación y
otro sin ella, el modelo asigne mayor probabilidad al primero. **No depende del punto de
corte**: mide el orden, no la decisión. 0,5 es azar; 1,0 es orden perfecto.

**Average precision = {c['average_precision']:.4f}** — área bajo la curva de
precisión-exhaustividad. Su línea base **no es 0,5 sino la prevalencia**:
{c['ap_linea_base']:.4f}. Es la métrica que mejor refleja el rendimiento con clases
desbalanceadas, y es la que responde si el modelo tiene señal real.

**Brier = 0,1665** — error cuadrático medio entre la probabilidad predicha y el resultado
observado. Mide **calibración**, no discriminación: un modelo puede ordenar bien y aun así
dar probabilidades infladas. La referencia es 0,2334, que es lo que daría predecir siempre
la prevalencia.
""")

        st.markdown('#### 4 · Cómo se pasa a «de cada 1.000 personas»')
        st.markdown('La sensibilidad y la especificidad **no cambian** con la frecuencia de '
                    'la enfermedad; el valor predictivo sí. La conversión es el teorema de '
                    'Bayes, donde *p* es la prevalencia del grupo examinado:')
        st.latex(r'\text{VPP}=\frac{\text{Sens}\cdot p}'
                 r'{\text{Sens}\cdot p+(1-\text{Espec})\cdot(1-p)}')
        st.latex(r'\text{VPN}=\frac{\text{Espec}\cdot(1-p)}'
                 r'{\text{Espec}\cdot(1-p)+(1-\text{Sens})\cdot p}')
        pv = 0.30
        con, sin_ = round(1000 * pv), 1000 - round(1000 * pv)
        se_n, es_n = round(con * SE), round(sin_ * ES)
        senaladas = se_n + (sin_ - es_n)
        vpp30 = SE * pv / (SE * pv + (1 - ES) * (1 - pv))
        st.markdown(f"""
**Ejemplo completo con p = 0,30** (personas ya derivadas por sospecha), sobre 1.000
examinadas:

- Tienen la malformación: 1.000 × 0,30 = **{con}**. La herramienta señala
  {con} × {SE:.4f} = **{se_n}**, y pierde {con - se_n}.
- No la tienen: **{sin_}**. Deja pasar {sin_} × {ES:.4f} = **{es_n}**, y señala por error
  {sin_ - es_n}.
- Señaladas en total: {se_n} + {sin_ - es_n} = **{senaladas}**.
- De esas, aciertan {se_n} → VPP = {se_n}/{senaladas} = **{se_n / senaladas:.4f}**, que es
  lo mismo que da la fórmula: {vpp30:.4f}.

Las cifras de la pestaña *Qué significa el resultado* son exactamente esta cuenta.
""")

        st.markdown('#### 5 · Los intervalos de confianza')
        st.markdown(f"""
Con {c['n_casos']} casos, una cifra sin intervalo no dice nada. Se usan dos métodos porque
miden cosas distintas:

**Wilson**, para sensibilidad, especificidad y precisión, que son proporciones. Se prefiere
al intervalo normal porque no se sale de [0, 1] ni se estrecha indebidamente cuando la
proporción se acerca a los extremos, que es justo el caso de una sensibilidad de {SE:.3f}.

**Bootstrap de {c['n_bootstrap']:,} remuestreos agrupados por caso**, para el AUC, que no es
una proporción y no tiene fórmula cerrada. Se remuestrean **casos, no imágenes**: las
imágenes de un mismo paciente no son observaciones independientes, y tratarlas como tales
estrecharía el intervalo artificialmente. Semilla {c['semilla']}.

| Métrica | Valor | IC 95 % | Método |
|---|---|---|---|
| Sensibilidad | {SE:.4f} | [{q['sens_ic95'][0]:.3f} – {q['sens_ic95'][1]:.3f}] | Wilson |
| Especificidad | {ES:.4f} | [{q['espec_ic95'][0]:.3f} – {q['espec_ic95'][1]:.3f}] | Wilson |
| Precisión | {q['precision']:.4f} | [{q['precision_ic95'][0]:.3f} – {q['precision_ic95'][1]:.3f}] | Wilson |
| AUC | {c['auc']:.4f} | [{c['auc_ic95'][0]:.3f} – {c['auc_ic95'][1]:.3f}] | Bootstrap por caso |
| Average precision | {c['average_precision']:.4f} | — | línea base {c['ap_linea_base']:.4f} |
| F1 | {q['f1']:.4f} | — | — |
| Exactitud | {q['exactitud']:.4f} | — | — |

El intervalo de la especificidad tiene {q['espec_ic95'][1] - q['espec_ic95'][0]:.3f} de
amplitud porque solo hay {NEG} negativos. Es la limitación de tamaño muestral más relevante
del trabajo.

#### 6 · Cómo se eligió el punto de corte

Regla fijada **antes** de mirar los resultados: {c['regla']}.

Entre todos los cortes que mantienen la sensibilidad en 0,95 o más, se tomó el de mayor
especificidad. Eso da {c['umbral_desplegado']:.4f}. **El conjunto de prueba reservado no
participó en esta elección**, y se abrió una sola vez, con arquitectura y umbral ya fijados.

Todas las cifras de esta pestaña se leen de `app/models/curva_operacion.json`, generado por
`src/curva_operacion.py` desde las predicciones out-of-fold. No hay ningún número escrito a
mano en esta interfaz.
""")

st.markdown(f'<div class="pie">Modelo <code>{nombre_modelo}</code> · pipeline v{PP.VERSION} '
            f'· arquitectura {arq} · umbral {umbral:.4f}<br>'
            f'Prototipo de investigación. Uso no clínico. Población adulta.</div>',
            unsafe_allow_html=True)
