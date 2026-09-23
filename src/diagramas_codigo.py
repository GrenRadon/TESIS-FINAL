"""Genera los diagramas estructurales como CÓDIGO (Mermaid / Graphviz), no como PNG.

Los diagramas de flujo y de bloques quedan mejor renderizados por una herramienta que los
dibuje a partir de código: se pueden reeditar, escalan sin perder calidad y el texto es
seleccionable. Las cifras se leen de los artefactos, igual que las tablas.

Dónde pegar el código:
  Mermaid   -> https://mermaid.live
  Graphviz  -> `dot -Tpng archivo.dot -o salida.png`
"""
import csv, json
from collections import defaultdict
from pathlib import Path

V2 = Path('resultados_v2')
V4 = Path('resultados_v4')
SALIDA = V2 / 'DIAGRAMAS_CODIGO.txt'
EXCL_FOSA = {36: 'atrofia cerebelosa', 38: 'meduloblastoma leptomeníngeo',
             40: 'degeneración cerebelosa'}
MAL_ETIQ = {'6163ad6ffd63b87f', '66291783b27449e6', 'f5871fe2194fe845'}


def cifras():
    """Todas las cifras del diagrama, contadas sobre los artefactos."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from manifiesto import leer_manifiesto
    reg = leer_manifiesto()
    aud = list(csv.DictReader(open(V2 / 'auditoria_nivel_imagen.csv', encoding='utf-8')))
    ds = list(csv.DictReader(open(V2 / 'dataset_nivel_imagen.csv', encoding='utf-8')))
    h = defaultdict(list)
    for r in aud:
        h[r['sha256']].append(r)
    g = defaultdict(lambda: [0, set()])
    for r in ds:
        k = (r['split'], r['clase'])
        g[k][0] += 1
        g[k][1].add(r['clave_agrupacion'])
    return dict(
        casos_man=len(reg),
        ch_man=sum(1 for k in reg if k[0] == 'chiari'),
        no_man=sum(1 for k in reg if k[0] == 'normal'),
        entradas=len(aud), unicas=len(h),
        dup=sum(len(v) - 1 for v in h.values()),
        mal=len({k for k in h if any(k.startswith(m) for m in MAL_ETIQ)}),
        excl=len(EXCL_FOSA),
        img=len(ds), casos=len({r['clave_agrupacion'] for r in ds}),
        dev_ch_c=len(g[('desarrollo', 'chiari')][1]), dev_ch_i=g[('desarrollo', 'chiari')][0],
        dev_no_c=len(g[('desarrollo', 'normal')][1]), dev_no_i=g[('desarrollo', 'normal')][0],
        te_ch_c=len(g[('test', 'chiari')][1]), te_ch_i=g[('test', 'chiari')][0],
        te_no_c=len(g[('test', 'normal')][1]), te_no_i=g[('test', 'normal')][0])


def consort_mermaid(c):
    return f"""flowchart TD
    A["<b>Casos documentados en el manifiesto Radiopaedia</b><br/>{c['casos_man']} casos · {c['ch_man']} Chiari + {c['no_man']} Normal"]
    B["Entradas de archivo auditadas<br/>{c['entradas']}<br/><i>el mismo archivo podía estar en varias carpetas</i>"]
    C["<b>Imágenes físicas únicas</b><br/>{c['unicas']} · deduplicadas por SHA-256"]
    D["Evaluación de elegibilidad<br/>por clase y región anatómica"]
    E["<b>INCLUIDAS</b><br/>{c['img']} imágenes · {c['casos']} casos"]
    F["<b>DESARROLLO</b><br/>{c['dev_ch_c'] + c['dev_no_c']} casos · {c['dev_ch_i'] + c['dev_no_i']} imágenes<br/>Chiari {c['dev_ch_c']} ({c['dev_ch_i']} img)<br/>Normal {c['dev_no_c']} ({c['dev_no_i']} img)"]
    G["<b>TEST SELLADO</b><br/>{c['te_ch_c'] + c['te_no_c']} casos · {c['te_ch_i'] + c['te_no_i']} imágenes<br/>Chiari {c['te_ch_c']} ({c['te_ch_i']} img)<br/>Normal {c['te_no_c']} ({c['te_no_i']} img)"]
    H["StratifiedGroupKFold k=5<br/>agrupado por caso<br/>Predicción out-of-fold"]
    I["Evaluado UNA vez<br/>con arquitectura y umbral<br/>ya fijados en el desarrollo"]

    X1["− {c['dup']} duplicados exactos<br/>mismo SHA-256"]
    X2["{c['mal']} imágenes aparecían con AMBAS clases<br/>reasignadas a Chiari (no se eliminan)"]
    X3["− {c['excl']} casos excluidos: patología EN fosa posterior<br/>casos 36, 38 y 40"]

    A --> B
    B --> C
    B -.-> X1
    C --> D
    C -.-> X2
    D --> E
    D -.-> X3
    E --> F
    E --> G
    F --> H
    G --> I

    classDef incl fill:#eef7f2,stroke:#1D9E75,stroke-width:2px
    classDef excl fill:#fdeeee,stroke:#c44,stroke-dasharray: 4 3
    classDef norm fill:#eef3fa,stroke:#333
    class E,F,G incl
    class X1,X2,X3 excl
    class A,B,C,D,H,I norm"""


def pipeline_mermaid():
    return """flowchart LR
    subgraph FUENTE["Fuente única · src/preprocesamiento.py v2.0.0"]
        P1["1 · Escala de grises"] --> P2["2 · CLAHE<br/>clip 2,0 · tile 8×8<br/><i>sobre resolución original</i>"]
        P2 --> P3["3 · Resize 224×224<br/>LANCZOS"]
        P3 --> P4["4 · 3 canales<br/>float32 en [0,255]"]
    end

    subgraph MODELO["Dentro del modelo · viaja con el .keras y el .onnx"]
        M1["preprocess_input<br/>específico de la arquitectura"] --> M2["DenseNet121<br/>pesos ImageNet"]
        M2 --> M3["GAP + BatchNorm<br/>Dense(32) + Dropout(0,5)"]
        M3 --> M4["Sigmoide"]
    end

    E1["Entrenamiento<br/>FoldGen"] --> P1
    E2["Exportación ONNX<br/>convert_to_onnx.py"] --> P1
    E3["Aplicación<br/>app.py"] --> P1
    P4 --> M1
    M4 --> S["Probabilidad<br/>umbral 0,1484"]

    classDef fuente fill:#eef7f2,stroke:#1D9E75,stroke-width:2px
    classDef modelo fill:#eef3fa,stroke:#378ADD
    classDef cons fill:#fff8e6,stroke:#D98324
    class P1,P2,P3,P4 fuente
    class M1,M2,M3,M4 modelo
    class E1,E2,E3 cons"""


def app_mermaid():
    return """flowchart TD
    U["Usuario carga JPG/PNG<br/><i>puede estar sin recortar</i>"] --> V0{"¿Imagen legible?"}
    V0 -->|no| R0["Rechazo:<br/>archivo no es una imagen válida"]
    V0 -->|sí| SG["<b>RF15 · Recuadro sugerido</b><br/>cabeza detectada por umbral de fondo<br/>geometría medida sobre 175 recortes manuales"]
    SG --> RC["<b>RF02 · El usuario ajusta y decide</b><br/>4 controles independientes en %<br/>indicador de forma y tamaño en vivo"]
    RC --> CF{"El usuario confirma<br/>el análisis"}
    CF --> V1{"Control de calidad<br/>resolución ≥ 64 px · proporción ≤ 3:1<br/>desv. típica ≥ 8 · ≥ 16 niveles de gris"}
    V1 -->|no cumple| R1["<b>RECHAZO</b><br/>se indica el motivo concreto"]
    V1 -->|cumple| V2["Distancia de Mahalanobis<br/>a la distribución de entrenamiento"]
    V2 --> V3{"¿Nivel?"}
    V3 -->|"≥ p99,5"| A1["Aviso: FUERA DE DISTRIBUCIÓN<br/>el resultado no es fiable"]
    V3 -->|"≥ p95"| A2["Aviso: entrada atípica"]
    V3 -->|"dentro"| P["Preprocesamiento<br/>fuente única"]
    A1 --> P
    A2 --> P
    P --> I["Inferencia ONNX<br/>p50 34 ms · p95 43 ms"]
    I --> D{"prob ≥ 0,1484"}
    D -->|sí| C1["Compatible con MC-I"]
    D -->|no| C2["No compatible con MC-I"]
    C1 --> W["Advertencias:<br/>uso no clínico · población adulta<br/>aviso si prob a menos de 0,10 del umbral"]
    C2 --> W

    classDef rech fill:#fdeeee,stroke:#c44
    classDef avis fill:#fff8e6,stroke:#D98324
    classDef ok fill:#eef7f2,stroke:#1D9E75
    classDef nuevo fill:#eef3fa,stroke:#378ADD,stroke-width:2px
    class SG,RC,CF nuevo
    class R0,R1 rech
    class A1,A2,W avis
    class C1,C2 ok"""


def casos_uso_mermaid():
    """Reemplaza la Figura 10. Cambios: desaparece la promesa de privacidad por diseño
    (M-19) y aparece la validación de entrada como sub-caso obligatorio (C-07)."""
    return """flowchart LR
    U(["Usuario<br/><i>sin registro ni autenticación</i>"])

    subgraph S["Sistema · aplicación web"]
        UC1(["Cargar imagen MRI<br/>sagital T1, completa o recortada"])
        UC0(["Ajustar el recuadro<br/>sobre la fosa posterior"])
        UC9(["Restablecer el<br/>recuadro sugerido"])
        UC2(["Obtener estimación<br/>de probabilidad"])
        UC3(["Validar la entrada"])
        UC4(["Preprocesar imagen"])
        UC5(["Ejecutar inferencia ONNX"])
        UC6(["Ver resultado y<br/>distancia al umbral"])
        UC7(["Consultar advertencias<br/>de alcance y uso"])
        UC8(["Ver modelo activo,<br/>umbral y versión"])
    end

    U --- UC1
    U --- UC0
    U --- UC2
    U --- UC8
    UC1 -. "«extend»" .-> UC0
    UC0 -. "«extend»" .-> UC9
    UC2 -. "«include»" .-> UC3
    UC2 -. "«include»" .-> UC4
    UC2 -. "«include»" .-> UC5
    UC2 -. "«extend»" .-> UC6
    UC2 -. "«extend»" .-> UC7

    classDef inc fill:#eef7f2,stroke:#1D9E75
    classDef ext fill:#fff8e6,stroke:#D98324
    class UC3,UC4,UC5 inc
    class UC0,UC6,UC7,UC9 ext"""


def secuencia_mermaid():
    """Reemplaza la Figura 11."""
    return """sequenceDiagram
    autonumber
    actor U as Usuario
    participant UI as Streamlit · app.py
    participant V as validacion_entrada.validar
    participant P as preprocesamiento.preprocesar
    participant S as ort.InferenceSession
    participant M as modelo .onnx + config_modelo.json

    rect rgb(238,243,250)
    note over UI,M: Arranque del servidor — una sola vez
    UI->>UI: cargar_modelo()  @st.cache_resource
    UI->>M: comprueba que hay UN solo .onnx
    alt hay 0 o más de 1
        M-->>UI: aborta con error visible
    end
    M-->>UI: sesión + umbral 0,1484 + versión del pipeline
    end

    rect rgb(255,248,230)
    note over U,V: Validación — puede detener el flujo
    U->>UI: st.file_uploader con la imagen
    UI->>UI: caja_sugerida() — RF15
    UI-->>U: recuadro propuesto + vista previa
    U->>UI: ajusta los 4 controles y confirma
    note over UI: RF02 — se analiza el RECORTE,<br/>no la imagen completa
    UI->>V: validar(arr del recorte)
    V->>V: control de calidad (lado, proporción, desv. típica, niveles)
    V->>V: distancia de Mahalanobis a la referencia de desarrollo
    alt no supera el control de calidad
        V-->>UI: RECHAZO con motivo
        UI-->>U: no se emite predicción
    else atípica (≥ p95) o fuera de distribución (≥ p99,5)
        V-->>UI: aviso, continúa
    end
    end

    rect rgb(238,247,242)
    note over UI,S: Predicción
    UI->>P: preprocesar(imagen)
    P->>P: gris → CLAHE(2,0 · 8×8) → resize 224 LANCZOS → 3 canales [0,255]
    P-->>UI: tensor [1,224,224,3] float32
    UI->>S: session.run(tensor)
    note right of S: preprocess_input va DENTRO del grafo
    S-->>UI: probabilidad p
    UI-->>U: p ≥ 0,1484 → «compatible con MC-I»
    UI-->>U: advertencias: uso no clínico · adultos · aviso si p está a menos de 0,10 del umbral
    end"""


def componentes_mermaid():
    """Reemplaza la Figura 12."""
    return """flowchart TB
    subgraph DAT["Datos"]
        D1["auditoria.py<br/>hashes · duplicados · fuga"]
        D2["particion.py<br/>manifiesto por caso<br/><b>aborta si dev ∩ test ≠ ∅</b>"]
        D3[("test_sellado.json<br/>congelado")]
        D1 --> D2 --> D3
    end

    subgraph ENT["Entrenamiento · única parte que depende de TensorFlow"]
        E1["entrenamiento_v4.py<br/>StratifiedGroupKFold k=5"]
        E2["umbral_y_edad.py<br/>umbral SOLO con OOF"]
        E3["entrenamiento_final_v4.py --sin-flip"]
        E1 --> E2
        E1 --> E3
    end

    subgraph CNV["Conversión"]
        C1["convert_to_onnx.py<br/><b>aborta si |Keras − ONNX| > 1e−4</b>"]
    end

    subgraph PRO["Producción · sin dependencia de TensorFlow"]
        R1["cargar_modelo()<br/>@st.cache_resource"]
        R2["validacion_entrada.py<br/>calidad + Mahalanobis"]
        R3["ort.InferenceSession"]
        R4["Interfaz Streamlit"]
    end

    PP["<b>src/preprocesamiento.py v2.0.0</b><br/>fuente única del preprocesamiento"]

    D2 --> E1
    E3 --> C1
    E2 -->|umbral| C1
    C1 --> A[("modelo .onnx<br/>+ config_modelo.json<br/><i>en la misma carpeta</i>")]
    A --> R1 --> R3
    R2 --> R3 --> R4
    PP -.-> E1
    PP -.-> C1
    PP -.-> R3

    classDef fuente fill:#eef7f2,stroke:#1D9E75,stroke-width:2px
    class PP,A fuente"""


def actividad_mermaid():
    """Reemplaza la Figura 13. El orden CLAHE→resize y el rango [0,255] son los cambios
    respecto a la versión del libro (C-06/C-08)."""
    return """flowchart TD
    I(["PIL.Image en modo arbitrario<br/>L · RGB · RGBA"]) --> RC

    subgraph REC["Recorte · RF02 y RF15, lo delimita el usuario"]
        RC0["Detección de la cabeza<br/>umbral de fondo"] --> RC1["Recuadro sugerido<br/>fx 0,662 · fy 0,594<br/>fw 0,728 · fh 0,681"]
        RC1 --> RC["Ajuste del usuario<br/>4 controles independientes"]
        RC --> RC2["imagen.crop(caja)"]
    end

    RC2 --> A1

    subgraph VAL["Validación · validacion_entrada.py"]
        A1["np.array → escala de grises uint8"] --> A2{"lado ≥ 64 ·<br/>proporción ≤ 3:1 ·<br/>desv. típica ≥ 8 ·<br/>≥ 16 niveles"}
        A2 -->|no| F(["RECHAZO<br/>no se emite predicción"])
        A2 -->|sí| A3["Mahalanobis sobre 6 rasgos"]
    end

    subgraph PRE["Preprocesamiento · preprocesamiento.py v2.0.0"]
        B1["1 · cv2.cvtColor → gris"]
        B2["2 · CLAHE clip 2,0 tile 8×8<br/><b>sobre la resolución original</b>"]
        B3["3 · resize 224×224 LANCZOS"]
        B4["4 · np.stack ×3 canales"]
        B5["5 · float32 en <b>[0,255]</b>"]
        B1 --> B2 --> B3 --> B4 --> B5
    end

    subgraph MOD["Dentro del grafo del modelo"]
        C1["preprocess_input de la arquitectura<br/>como capas Keras"]
    end

    A3 --> B1
    B5 --> D(["tensor [1,224,224,3] float32"])
    D --> C1 --> E(["probabilidad"])

    classDef mal fill:#fdeeee,stroke:#c44
    classDef ok fill:#eef7f2,stroke:#1D9E75
    class F mal
    class B2,B3,B5 ok"""


def consort_dot(c):
    return f"""digraph CONSORT {{
    rankdir=TB;
    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];
    edge [fontname="Helvetica", fontsize=9];

    A [label="Casos documentados en el manifiesto Radiopaedia\\n{c['casos_man']} casos ({c['ch_man']} Chiari + {c['no_man']} Normal)", fillcolor="#eef3fa"];
    B [label="Entradas de archivo auditadas\\n{c['entradas']}", fillcolor="#eef3fa"];
    C [label="Imágenes físicas únicas\\n{c['unicas']} (deduplicadas por SHA-256)", fillcolor="#eef3fa"];
    D [label="Evaluación de elegibilidad\\npor clase y región anatómica", fillcolor="#eef3fa"];
    E [label="INCLUIDAS\\n{c['img']} imágenes / {c['casos']} casos", fillcolor="#eef7f2", color="#1D9E75", penwidth=2];
    F [label="DESARROLLO\\n{c['dev_ch_c'] + c['dev_no_c']} casos / {c['dev_ch_i'] + c['dev_no_i']} img\\nChiari {c['dev_ch_c']} · Normal {c['dev_no_c']}", fillcolor="#eef7f2", color="#1D9E75"];
    G [label="TEST SELLADO\\n{c['te_ch_c'] + c['te_no_c']} casos / {c['te_ch_i'] + c['te_no_i']} img\\nChiari {c['te_ch_c']} · Normal {c['te_no_c']}", fillcolor="#eef7f2", color="#1D9E75"];

    X1 [label="− {c['dup']} duplicados exactos (SHA-256)", fillcolor="#fdeeee", color="#c44"];
    X2 [label="{c['mal']} imágenes con AMBAS clases\\nreasignadas a Chiari", fillcolor="#fdeeee", color="#c44"];
    X3 [label="− {c['excl']} casos: patología EN fosa posterior\\n(36, 38, 40)", fillcolor="#fdeeee", color="#c44"];

    A -> B; B -> C; C -> D; D -> E; E -> F; E -> G;
    B -> X1 [style=dashed, arrowhead=none];
    C -> X2 [style=dashed, arrowhead=none];
    D -> X3 [style=dashed, arrowhead=none];
    {{rank=same; B; X1}} {{rank=same; C; X2}} {{rank=same; D; X3}} {{rank=same; F; G}}
}}"""


def main():
    c = cifras()
    L = ['# Diagramas como código — para renderizar, no como imagen', '',
         'Las cifras se leen de los artefactos del pipeline. Regenerable con '
         '`python src/diagramas_codigo.py`.', '',
         '**Dónde pegarlo:** Mermaid en <https://mermaid.live>. Graphviz con '
         '`dot -Tpng archivo.dot -o salida.png`.', '',
         '---', '',
         '## 1. Flujo de selección del conjunto de datos (CONSORT/STARD) — C-02', '',
         'Para **Cap. 4 §4.2 Dataset**.', '', '### Mermaid', '', '```mermaid',
         consort_mermaid(c), '```', '',
         '### Graphviz (alternativa, mejor control del trazado)', '', '```dot',
         consort_dot(c), '```', '',
         '---', '',
         '## 2. Pipeline de preprocesamiento — C-06 / C-08', '',
         'Para **Cap. 4 §4.2 Preprocesamiento**. Muestra que las tres rutas —entrenamiento, '
         'exportación y aplicación— convergen en la misma fuente, y que `preprocess_input` va '
         'dentro del modelo.', '', '```mermaid', pipeline_mermaid(), '```', '',
         '---', '',
         '## 3. Flujo de la aplicación con validación de entrada — M-17 / M-18 / C-07', '',
         'Para **Cap. 4 §4.4 Servicio web**. Incluye el control de calidad y la detección fuera '
         'de distribución.', '', '```mermaid', app_mermaid(), '```', '',
         '---', '',
         '## 4. Diagramas UML del libro que hay que rehacer', '',
         'Las Figuras 10 a 13 del libro describen un sistema que ya no es el vigente: '
         'preprocesamiento en otro orden y otro rango, umbral 0,5, sin validación de entrada y '
         'con una promesa de «privacidad por diseño» que M-19 retira. Estos cuatro reemplazos '
         'mantienen el mismo tipo de diagrama UML y la misma numeración.', '',
         '### 4.1 Casos de uso — reemplaza la Figura 10 (§4.4.4)', '', '```mermaid',
         casos_uso_mermaid(), '```', '',
         '### 4.2 Secuencia del flujo de predicción — reemplaza la Figura 11 (§4.4.5)', '',
         '```mermaid', secuencia_mermaid(), '```', '',
         '### 4.3 Componentes — reemplaza la Figura 12 (§4.4.6)', '', '```mermaid',
         componentes_mermaid(), '```', '',
         '### 4.4 Actividad de `preprocesar()` — reemplaza la Figura 13 (§4.4.7)', '',
         '```mermaid', actividad_mermaid(), '```', '']
    SALIDA.write_text('\n'.join(L), encoding='utf-8')
    print(f'  escrito: {SALIDA}')
    print(f'  cifras del CONSORT: {c["casos_man"]} casos documentados -> {c["unicas"]} únicas '
          f'-> {c["img"]} imágenes / {c["casos"]} casos incluidos')


if __name__ == '__main__':
    main()
