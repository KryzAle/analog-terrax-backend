# Organización de `app/data/`

```text
app/data/
├── profiles/    Perfiles compactos usados en tiempo de ejecución
├── *.json       Taxonomía, sitios e índices estadísticos del backend
├── *.npz        Índices compactos de similitud del backend
├── input/       Entradas pequeñas versionadas, como sitios.csv
├── raw/         Rásteres y archivos originales descargados (ignorado por Git)
├── processed/   El CSV topografia_sitios_25km.csv se versiona; otros resultados se ignoran
├── catalogs/    Catálogos, planes y validaciones (ignorado por Git)
└── reference/   Fuentes auxiliares externas (ignorado por Git)
```

Los JSON y NPZ de la raíz de `app/data/` son artefactos compactos requeridos por
la API y se mantienen versionados. `input/sitios.csv` es la configuración de
sitios usada por los procesos científicos. Sólo `processed/topografia_sitios_25km.csv` se versiona para el despliegue;
los demás resultados procesados se ignoran. Los rásteres originales
pueden ocupar cientos de megabytes; `raw/`, `catalogs/` y `reference/` se
ignoran en Git y sus carpetas se crean al ejecutar los flujos.
