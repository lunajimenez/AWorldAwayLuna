# A World Away: Hunting for Exoplanets with AI 🌌

> Primer análisis y exploración de los datos para el NASA Space Apps Challenge 2025 del grupo AstrobytesUTB!

## 🚀 Descripción

Este proyecto aborda el desafío de identificación automatizada de exoplanetas utilizando técnicas de Machine Learning e Inteligencia Artificial. El objetivo es analizar y procesar datos de diferentes misiones espaciales (Kepler, K2 y TESS) para crear modelos predictivos que puedan identificar nuevos exoplanetas de manera automática.

## 📊 Estructura del Proyecto

```
.
├── Data/                         # Datos originales de las misiones
│   ├── cumulative_2025.10.01.csv # Dataset Kepler
│   ├── k2pandc_2025.10.01.csv    # Dataset K2
│   └── TOI_2025.10.01.csv        # Dataset TESS
│
├── FilteredData/                 # Datos procesados y filtrados
│   ├── K2_All_Filtrated.csv
│   ├── K2Filtrated.csv
│   ├── k2_iterative_imputer.joblib
│   ├── K2Filtrated_imputed_iterative.csv
│   ├── KOI_All_Filtrated.csv
│   ├── KOI_Filtrated.csv
│   ├── koi_iterative_imputer.joblib
│   ├── KOIFiltrated_imputed_iterative.csv
│   ├── TOI_All_Filtrated.csv
│   ├── toi_iterative_imputer.joblib
│   └── TOIFiltrated_imputed_iterative.csv
│
├── Notebooks/                    # Jupyter notebooks para análisis
│   └── analisis_exploratorio.ipynb
│
└── Luna/                        # Entorno virtual de Python
```

## 🛠️ Procesamiento de Datos

### Análisis Exploratorio (EDA)
- Exploración detallada de las características de cada dataset
- Análisis de valores nulos y outliers
- Visualización de distribuciones y correlaciones
- Identificación de variables clave para la detección de exoplanetas

### Preprocesamiento
1. **Filtrado de Datos**
   - Eliminación de columnas con alta proporción de valores nulos
   - Remoción de identificadores y variables no relevantes
   - Tratamiento de valores atípicos

2. **Imputación de Datos**
   - Implementación de imputación iterativa para manejar valores faltantes
   - Generación de modelos de imputación específicos para cada dataset
   - Validación de la calidad de las imputaciones

### Variables Clave Identificadas
- **Parámetros Orbitales**: período, duración del tránsito, profundidad
- **Parámetros Estelares**: temperatura efectiva, radio estelar
- **Parámetros Planetarios**: radio del planeta, temperatura de equilibrio

## 📈 Características del Análisis

- Implementación de técnicas de visualización avanzadas
- Análisis de correlaciones entre variables
- Estudio de la distribución de clases (confirmados/candidatos/falsos positivos)
- Evaluación de la calidad de los datos por misión

## 🔍 Próximos Pasos

1. Desarrollo de modelos de Machine Learning
2. Implementación de una interfaz web
3. Validación cruzada y optimización de modelos
4. Integración de nuevos datos por parte del usuario para actualización continua 

## 🛠️ Tecnologías Utilizadas

- Python 3.12
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebooks

## 📝 Notas

Este mini-proyecto forma parte del desafío "A World Away: Hunting for Exoplanets with AI" del NASA Space Apps Challenge 2025, que busca automatizar la identificación de exoplanetas utilizando técnicas de IA/ML sobre datos de misiones espaciales de la NASA.

## 👥 Autores

- Luna Jiménez

*"Explorando nuevos mundos a través de los datos"* 🌍✨