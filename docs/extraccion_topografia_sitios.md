# Extracción topográfica de sitios lunares y marcianos

## Objetivo

`scripts/extraer_topografia_sitios.py` adapta el lector PDS3 y el procesamiento
topográfico del notebook
[`notebooks/ODE_REST_Extraccion_Luna_Marte.ipynb`](../notebooks/ODE_REST_Extraccion_Luna_Marte.ipynb).
Descarga siempre desde PDS los DEM LOLA `LDEM_64` y MOLA `MEGT90N000EB`, incluso
si ya existen localmente. Lee las primeras cinco filas de `moon` y las primeras
cinco de `mars`, según el orden de `app/data/input/sitios.csv`, y resume los DEM en una ventana cuadrada de
25 × 25 km centrada en cada ubicación. El resultado tiene diez filas y solo
estas columnas: `nombre`, `cuerpo`, `relief_m`, `slope_mean_deg`,
`slope_p90_deg`, `ruggedness_tri_m`, `roughness_rms_m` y
`hypsometric_integral`.

## Ejecución

Desde la raíz del repositorio, con el entorno del proyecto activo:

```powershell
python scripts/extraer_topografia_sitios.py
```

El destino predeterminado es
`app/data/processed/topografia_sitios_25km.csv`. Cada ejecución vuelve a descargar
y reemplaza ambos DEM y sus etiquetas PDS3. Se guardan en
`app/data/raw/moon/topography/LUN-TOP-01/` y
`app/data/raw/mars/topography/MAR-TOP-01/`. Hace falta conexión a Internet. Se puede
cambiar la ventana, cantidad de sitios, entrada o salida:

```powershell
python scripts/extraer_topografia_sitios.py `
  --side-km 25 `
  --per-body 5 `
  --sites app/data/input/sitios.csv `
  --output app/data/processed/topografia_sitios_25km.csv
```

Este es el modo reproducible desde cero. Para procesar DEM locales sin
descargar ni sobrescribir, usar `--no-download` y `--moon-dem` / `--mars-dem`
para indicar etiquetas `.LBL` locales. El `.IMG` referenciado por `^IMAGE` debe
quedar junto a la etiqueta.

## Características y definiciones

Las elevaciones se convierten a metros aplicando `SCALING_FACTOR`; un `OFFSET`
de radio planetario se ignora al trabajar con alturas relativas. El tamaño del
píxel se calcula localmente con el radio planetario y la resolución de la
proyección cilíndrica. Se agregan las siguientes métricas sobre los píxeles
válidos de la ventana:

| Columna | Definición | Unidad |
|---|---|---|
| `relief_m` | Elevación máxima menos mínima | m |
| `slope_mean_deg` | Promedio de la pendiente calculada con gradientes centrales | grados |
| `slope_p90_deg` | Percentil 90 de la pendiente | grados |
| `ruggedness_tri_m` | Promedio del TRI de Riley: raíz de la suma de diferencias al cuadrado con los ocho vecinos | m |
| `roughness_rms_m` | Promedio de la desviación RMS local de elevación en vecindarios 3 × 3 | m |
| `hypsometric_integral` | `(elevación media - elevación mínima) / relief`; 0 si el relieve es cero | adimensional |

Para calcular derivadas en el borde del recorte se lee un píxel de margen; el
resumen se calcula dentro del rectángulo nominal. En ventanas cercanas a un
polo, la anchura longitudinal aumenta para conservar aproximadamente los 25 km
de lado este-oeste, con un límite de 359°.

## Selección de ubicaciones

La selección es determinista: las cinco primeras filas cuyo `cuerpo` sea
`moon` y las cinco primeras filas `mars`. En el CSV vigente son:

- Moon: `crater_shackleton`, `planalto_aristarchus_meseta`, `crater_aristarchus`, `colinas_de_marius`, `anomalia_reiner_gamma`.
- Mars: `crater_jezero_centro`, `crater_jezero_delta`, `crater_gale_centro`, `crater_gale_curiosity`, `oxia_planum_centro`.

Cambiar el orden en `sitios.csv` cambia cuáles ubicaciones se seleccionan.

## Insumos actuales y limitación conocida

El DEM lunar predeterminado es
`app/data/raw/moon/topography/LUN-TOP-01/ldem_64.lbl` (LOLA LDEM 64). El archivo
MOLA local previo `megc90n000cb.lbl` declara `NAME=COUNTS_PER_BIN` (conteos);
el script lo reemplaza con `MEGT90N000EB`, el DEM de topografía. En la
nomenclatura MOLA, `T` indica topografía y `C`, conteos. El DEM marciano tiene
16 píxeles por grado (aprox. 3.7 km por píxel en el ecuador), así que las
métricas reflejan el relieve a esa escala. Las URLs fuente están definidas en
`DEM_URLS` dentro del script.

El lector admite rásteres PDS3 de una banda en proyección cilíndrica o
equirectangular. Lee del archivo solamente las filas de cada ventana, sin
mapear el DEM lunar completo en memoria. Si los dos DEM ya se descargaron antes
y solo se necesita reintentar el procesamiento, ejecutar con `--no-download`.
Las celdas nulas se excluyen de las estadísticas de elevación;
si la cobertura válida es inferior al 80 %, la extracción se detiene para ese
insumo. La rutina requiere NumPy, ya incluido en las dependencias del proyecto.
