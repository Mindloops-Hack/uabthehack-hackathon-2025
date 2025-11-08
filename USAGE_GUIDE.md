# Usage Guide - UAB WiFi Dataset

**Evento:** UAB THE HACK! 2025 (8-9 de noviembre)
**Challenge:** DTIC WiFi Network Analysis
**Nivel:** Todos los niveles (Rookie → Advanced)

---

## Tabla de Contenidos

1. [Quick Start](#quick-start)
2. [Estructura del Dataset](#estructura-del-dataset)
3. [Carga de Datos](#carga-de-datos)
4. [Análisis Básico (Nivel Rookie)](#análisis-básico-nivel-rookie)
5. [Análisis Avanzado (Nivel Intermedio)](#análisis-avanzado-nivel-intermedio)
6. [Machine Learning y LLMs (Nivel Avanzado)](#machine-learning-y-llms-nivel-avanzado)
7. [Tips y Trucos](#tips-y-trucos)
8. [Problemas Comunes](#problemas-comunes)
9. [Recursos](#recursos)

---

## Quick Start

### Instalación Rápida

```bash
# Clonar o descargar el dataset
cd dtic-wifi-analysis

# Crear entorno virtual (recomendado)
python -m venv venv

# Activar entorno
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Instalar dependencias básicas
pip install pandas matplotlib seaborn jupyter

# O instalar todo desde requirements.txt
pip install -r requirements.txt

# Lanzar Jupyter
jupyter notebook
```

### Primeros Pasos en 5 Minutos

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar un archivo de APs
with open('anonymized_data/AP-info-v2-2025-06-13T14_45_01+02_00-ANON.json', 'r') as f:
    aps = json.load(f)

# 2. Convertir a DataFrame
df_aps = pd.DataFrame(aps)

# 3. Ver las primeras filas
print(df_aps.head())

# 4. Estadísticas básicas
print(f"Total APs: {len(df_aps)}")
print(f"APs activos: {(df_aps['status'] == 'Up').sum()}")
print(f"Total dispositivos: {df_aps['client_count'].sum()}")

# 5. Visualización rápida
df_aps['client_count'].hist(bins=50)
plt.xlabel('Dispositivos por AP')
plt.ylabel('Frecuencia')
plt.title('Distribución de Carga en Access Points')
plt.show()
```

---

## Estructura del Dataset

### Archivos Disponibles

```
dtic-wifi-analysis/
├── README.md                      # Descripción general del challenge
├── DATA_DICTIONARY.md             # Diccionario de datos (este archivo es tu biblia!)
├── USAGE_GUIDE.md                 # Esta guía
├── requirements.txt               # Dependencias Python
│
├── anonymized_data/               # DATOS PRINCIPALES (copiar aquí desde Google Drive)
│   ├── AP-info-v2-[timestamp]-ANON.json       # 7.229 archivos (~10GB)
│   └── client-info-[timestamp]-ANON.json      # 3.205 archivos
│
├── anonymized_samples/            # Muestras para desarrollo rápido
│   ├── AP-info-v2-2025-06-13T14_45_01+02_00-ANON.json
│   └── client-info-2025-04-09T11_47_24+02_00-10487-ANON.json
│
├── starter_kits/                  # Notebooks de ejemplo
│   ├── 01_rookie_basic_analysis.ipynb
│   ├── 02_intermediate_mobility.ipynb
│   └── 03_advanced_ml_llm.ipynb
│
└── utils/                         # Funciones auxiliares
    ├── data_loader.py             # Funciones para cargar datos eficientemente
    ├── visualization.py           # Funciones de visualización
    └── preprocessing.py           # Limpieza y transformación
```

### Nomenclatura de Archivos

#### Access Points
```
AP-info-v2-[TIMESTAMP].json

Ejemplos:
AP-info-v2-2025-04-05T10_00_01+02_00-ANON.json
             ↑                ↑
             fecha/hora       zona horaria
```

#### Clientes
```
client-info-[TIMESTAMP]-[COUNT].json

Ejemplos:
client-info-2025-04-09T11_47_24+02_00-10487-ANON.json
                                             ↑
                                             número de dispositivos
```

### Tamaño de Datos

| Tipo | Cantidad | Tamaño Total | Tamaño Promedio |
|------|----------|--------------|-----------------|
| APs | 7.229 archivos | ~10 GB | ~1.4 MB/archivo |
| Clientes | 3.205 archivos | ~15 GB | ~5 MB/archivo |
| **Total** | **10.434 archivos** | **~25 GB** | - |

---

## Carga de Datos

### Opción 1: Cargar un Solo Archivo (Desarrollo)

```python
import json
import pandas as pd

# Cargar APs
with open('anonymized_samples/AP-info-v2-2025-06-13T14_45_01+02_00-ANON.json', 'r', encoding='utf-8') as f:
    aps_data = json.load(f)

df_aps = pd.DataFrame(aps_data)

# Cargar Clientes
with open('anonymized_samples/client-info-2025-04-09T11_47_24+02_00-10487-ANON.json', 'r', encoding='utf-8') as f:
    clients_data = json.load(f)

df_clients = pd.DataFrame(clients_data)
```

### Opción 2: Cargar Múltiples Archivos (Análisis Completo)

```python
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # Para barra de progreso

def load_all_json_files(directory, pattern):
    """
    Carga todos los archivos JSON que coinciden con un patrón.

    Args:
        directory: Ruta al directorio
        pattern: Patrón glob (ej: "AP-info-v2-*.json")

    Returns:
        DataFrame combinado
    """
    files = list(Path(directory).glob(pattern))
    print(f"Encontrados {len(files)} archivos")

    all_data = []

    for file in tqdm(files):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Añadir timestamp del nombre del archivo
                timestamp = extract_timestamp_from_filename(file.name)

                for record in data:
                    record['_file_timestamp'] = timestamp

                all_data.extend(data)
        except Exception as e:
            print(f"Error cargando {file}: {e}")

    return pd.DataFrame(all_data)

def extract_timestamp_from_filename(filename):
    """
    Extrae el timestamp del nombre del archivo.

    Ejemplo:
    'AP-info-v2-2025-04-05T10_00_01+02_00-ANON.json'
    → '2025-04-05T10:00:01+02:00'
    """
    import re
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\+\d{2}_\d{2})', filename)
    if match:
        return match.group(1).replace('_', ':')
    return None

# Usar la función
df_aps_all = load_all_json_files('anonymized_data/', 'AP-info-v2-*-ANON.json')
df_clients_all = load_all_json_files('anonymized_data/', 'client-info-*-ANON.json')
```

### Opción 3: Cargar Selectivamente por Fecha/Hora

```python
from datetime import datetime, timedelta

def load_files_in_range(directory, pattern, start_date, end_date):
    """
    Carga archivos en un rango de fechas específico.

    Args:
        start_date: datetime object
        end_date: datetime object
    """
    files = Path(directory).glob(pattern)
    selected_files = []

    for file in files:
        timestamp_str = extract_timestamp_from_filename(file.name)
        if timestamp_str:
            file_date = datetime.fromisoformat(timestamp_str)
            if start_date <= file_date <= end_date:
                selected_files.append(file)

    print(f"Cargando {len(selected_files)} archivos entre {start_date} y {end_date}")

    # Cargar igual que antes...
    # (código similar a load_all_json_files)

# Ejemplo: Solo cargar datos de una semana
start = datetime(2025, 4, 10)
end = datetime(2025, 4, 17)
df_week = load_files_in_range('anonymized_data/', 'client-info-*.json', start, end)
```

### Opción 4: Sampling Aleatorio (Para Prototipado Rápido)

```python
import random

def load_random_sample(directory, pattern, n_files=10):
    """Carga N archivos aleatorios para desarrollo rápido."""
    files = list(Path(directory).glob(pattern))
    sample_files = random.sample(files, min(n_files, len(files)))

    print(f"Cargando muestra de {len(sample_files)} archivos")

    # Cargar igual que antes...

# Ejemplo: Cargar 20 archivos aleatorios
df_sample = load_random_sample('anonymized_data/', 'client-info-*.json', n_files=20)
```

---

## Análisis Básico (Nivel Rookie)

### 1. Exploración Inicial de Datos

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Cargar datos (usar Opción 1 para empezar)
df_aps = pd.read_json('anonymized_samples/AP-info-v2-2025-06-13T14_45_01+02_00-ANON.json')
df_clients = pd.read_json('anonymized_samples/client-info-2025-04-09T11_47_24+02_00-10487-ANON.json')

# Información general
print("=== ACCESS POINTS ===")
print(f"Total APs: {len(df_aps)}")
print(f"Columnas: {df_aps.columns.tolist()}")
print(df_aps.info())

print("\n=== CLIENTES ===")
print(f"Total clientes: {len(df_clients)}")
print(f"Columnas: {df_clients.columns.tolist()}")
print(df_clients.info())
```

### 2. Estadísticas Descriptivas

```python
# APs
print("=== Estadísticas de Access Points ===")
print(df_aps[['client_count', 'cpu_utilization', 'mem_free']].describe())

# Clientes
print("\n=== Estadísticas de Clientes ===")
print(df_clients[['signal_db', 'snr', 'speed', 'health']].describe())
```

### 3. Identificar Zonas Hotspot

```python
# Extraer edificio del nombre del AP
df_aps['building'] = df_aps['name'].str.extract(r'AP-([A-Z]+)\d+')[0]

# Agrupar por edificio
hotspots = df_aps.groupby('building').agg({
    'client_count': 'sum',
    'name': 'count'  # Número de APs por edificio
}).rename(columns={'name': 'num_aps'})

hotspots['avg_clients_per_ap'] = hotspots['client_count'] / hotspots['num_aps']
hotspots = hotspots.sort_values('client_count', ascending=False)

print("=== Top 10 Edificios con Más Dispositivos ===")
print(hotspots.head(10))

# Visualización
plt.figure(figsize=(14, 6))
hotspots.head(15)['client_count'].plot(kind='bar')
plt.title('Top 15 Edificios por Densidad de Dispositivos')
plt.xlabel('Edificio')
plt.ylabel('Total Dispositivos Conectados')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 4. Distribución de Calidad de Señal

```python
# Histograma de RSSI
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df_clients['signal_db'].hist(bins=50, edgecolor='black')
plt.xlabel('RSSI (dBm)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Potencia de Señal')
plt.axvline(-70, color='red', linestyle='--', label='Umbral "débil"')
plt.axvline(-50, color='green', linestyle='--', label='Umbral "excelente"')
plt.legend()

plt.subplot(1, 2, 2)
df_clients['signal_strength'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Signal Strength (1-5)')
plt.ylabel('Número de Clientes')
plt.title('Distribución de Fuerza de Señal Simplificada')

plt.tight_layout()
plt.show()

# Porcentaje con señal pobre
poor_signal_pct = (df_clients['signal_db'] < -70).mean() * 100
print(f"Porcentaje de clientes con señal pobre (<-70 dBm): {poor_signal_pct:.1f}%")
```

### 5. Análisis de Bandas (2.4 GHz vs 5 GHz)

```python
# Distribución de clientes por banda
band_distribution = df_clients['band'].value_counts()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
band_distribution.plot(kind='pie', autopct='%1.1f%%', labels=['5 GHz', '2.4 GHz'])
plt.title('Distribución de Clientes por Banda')
plt.ylabel('')

plt.subplot(1, 2, 2)
df_clients.boxplot(column='speed', by='band')
plt.xlabel('Banda (GHz)')
plt.ylabel('Velocidad (Mbps)')
plt.title('Velocidad por Banda')
plt.suptitle('')  # Quitar título automático

plt.tight_layout()
plt.show()

# Comparación de velocidad promedio
print("=== Velocidad Promedio por Banda ===")
print(df_clients.groupby('band')['speed'].agg(['mean', 'median', 'max']))
```

### 6. Tipos de Dispositivos

```python
# Top 10 fabricantes
top_manufacturers = df_clients['manufacturer'].value_counts().head(10)

plt.figure(figsize=(12, 6))
top_manufacturers.plot(kind='barh')
plt.title('Top 10 Fabricantes de Dispositivos')
plt.xlabel('Número de Dispositivos')
plt.ylabel('Fabricante')
plt.tight_layout()
plt.show()

# Sistemas operativos
os_distribution = df_clients['os_type'].value_counts()
print("=== Distribución de Sistemas Operativos ===")
print(os_distribution)

# Categorías de dispositivos
category_distribution = df_clients['client_category'].value_counts()
print("\n=== Categorías de Dispositivos ===")
print(category_distribution)
```

### 7. Red UAB vs eduroam

```python
network_stats = df_clients.groupby('network').agg({
    'macaddr': 'count',
    'signal_db': 'mean',
    'speed': 'mean',
    'health': 'mean'
}).rename(columns={'macaddr': 'num_clients'})

print("=== Comparación UAB vs eduroam ===")
print(network_stats)

# Visualización
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

network_stats['num_clients'].plot(kind='bar', ax=axes[0])
axes[0].set_title('Número de Clientes')
axes[0].set_ylabel('Clientes')

network_stats['signal_db'].plot(kind='bar', ax=axes[1])
axes[1].set_title('RSSI Promedio')
axes[1].set_ylabel('dBm')

network_stats['speed'].plot(kind='bar', ax=axes[2])
axes[2].set_title('Velocidad Promedio')
axes[2].set_ylabel('Mbps')

plt.tight_layout()
plt.show()
```

### 8. Rendimiento de Access Points

```python
# Top 10 APs con más clientes
top_aps = df_aps.nlargest(10, 'client_count')[['name', 'client_count', 'cpu_utilization', 'status']]
print("=== Top 10 APs con Más Clientes ===")
print(top_aps)

# APs con problemas
problematic_aps = df_aps[
    (df_aps['status'] == 'Down') |
    (df_aps['cpu_utilization'] > 80)
][['name', 'status', 'client_count', 'cpu_utilization']]

print("\n=== APs con Posibles Problemas ===")
print(problematic_aps)

# Scatter plot: carga vs CPU
plt.figure(figsize=(10, 6))
plt.scatter(df_aps['client_count'], df_aps['cpu_utilization'], alpha=0.5)
plt.xlabel('Número de Clientes')
plt.ylabel('CPU Utilization (%)')
plt.title('Relación entre Carga de Clientes y Uso de CPU')
plt.axhline(80, color='red', linestyle='--', label='Umbral crítico (80%)')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Análisis Avanzado (Nivel Intermedio)

### 1. Análisis Temporal (Series de Tiempo)

```python
import pandas as pd
from datetime import datetime

# Cargar múltiples archivos con timestamps
df_clients_temporal = load_all_json_files('anonymized_data/', 'client-info-*.json')

# Convertir timestamp a datetime
df_clients_temporal['datetime'] = pd.to_datetime(
    df_clients_temporal['last_connection_time'],
    unit='ms'
)

# Extraer características temporales
df_clients_temporal['hour'] = df_clients_temporal['datetime'].dt.hour
df_clients_temporal['day_of_week'] = df_clients_temporal['datetime'].dt.dayofweek
df_clients_temporal['date'] = df_clients_temporal['datetime'].dt.date

# Análisis por hora del día
hourly_activity = df_clients_temporal.groupby('hour').size()

plt.figure(figsize=(12, 5))
hourly_activity.plot(kind='bar')
plt.xlabel('Hora del Día')
plt.ylabel('Número de Conexiones')
plt.title('Actividad de Red por Hora del Día')
plt.xticks(range(24), range(24), rotation=0)
plt.axvspan(9, 21, alpha=0.2, color='green', label='Horario lectivo')
plt.legend()
plt.tight_layout()
plt.show()

# Heatmap: Día de semana vs Hora
pivot_table = df_clients_temporal.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)

plt.figure(figsize=(14, 6))
sns.heatmap(
    pivot_table,
    cmap='YlOrRd',
    annot=False,
    fmt='d',
    yticklabels=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
)
plt.title('Heatmap de Actividad: Día de Semana vs Hora')
plt.xlabel('Hora del Día')
plt.ylabel('Día de la Semana')
plt.tight_layout()
plt.show()
```

### 2. Análisis de Movilidad (Device Tracking)

```python
# Cargar dos snapshots temporales diferentes
df_t1 = pd.read_json('anonymized_data/client-info-2025-04-09T10_00_00+02_00-ANON.json')
df_t2 = pd.read_json('anonymized_data/client-info-2025-04-09T14_00_00+02_00-ANON.json')

# Merge por MAC del dispositivo
df_t1_subset = df_t1[['macaddr', 'associated_device_name', 'signal_db']].rename(
    columns=lambda x: x + '_t1' if x != 'macaddr' else x
)
df_t2_subset = df_t2[['macaddr', 'associated_device_name', 'signal_db']].rename(
    columns=lambda x: x + '_t2' if x != 'macaddr' else x
)

mobility = df_t1_subset.merge(df_t2_subset, on='macaddr', how='inner')

# Filtrar dispositivos que cambiaron de AP
moved = mobility[mobility['associated_device_name_t1'] != mobility['associated_device_name_t2']]

print(f"Total dispositivos en ambos snapshots: {len(mobility)}")
print(f"Dispositivos que se movieron: {len(moved)} ({len(moved)/len(mobility)*100:.1f}%)")

# Top movimientos (transiciones más comunes)
movements = moved.groupby(['associated_device_name_t1', 'associated_device_name_t2']).size()
movements = movements.sort_values(ascending=False).head(10)

print("\n=== Top 10 Transiciones Más Comunes ===")
for (ap1, ap2), count in movements.items():
    print(f"{ap1} → {ap2}: {count} dispositivos")
```

### 3. Análisis de Grafo de Movilidad (NetworkX)

```python
import networkx as nx

# Crear grafo dirigido de movimientos
G = nx.DiGraph()

# Añadir edges (transiciones)
for (ap_from, ap_to), count in movements.items():
    G.add_edge(ap_from, ap_to, weight=count)

# Visualización
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=2, iterations=50)

# Tamaño de nodos proporcional al grado
node_sizes = [G.degree(node) * 100 for node in G.nodes()]

# Grosor de edges proporcional al peso
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8)
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edges(G, pos, width=[w/10 for w in weights], alpha=0.5, arrows=True)

plt.title('Grafo de Movilidad entre Access Points')
plt.axis('off')
plt.tight_layout()
plt.show()

# Métricas de centralidad
betweenness = nx.betweenness_centrality(G, weight='weight')
print("\n=== Top 5 APs con Mayor Centralidad (hubs de tránsito) ===")
for ap, score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{ap}: {score:.4f}")
```

### 4. Mapa de Calor de Densidad + Calidad

```python
# Agrupar por AP
ap_quality = df_clients.groupby('associated_device_name').agg({
    'macaddr': 'count',  # Número de clientes
    'signal_db': 'mean',  # RSSI promedio
    'snr': 'mean',
    'health': 'mean'
}).rename(columns={'macaddr': 'num_clients'})

# Extraer edificio
ap_quality['building'] = ap_quality.index.to_series().str.extract(r'AP-([A-Z]+)\d+')[0]

# Agrupar por edificio
building_quality = ap_quality.groupby('building').agg({
    'num_clients': 'sum',
    'signal_db': 'mean',
    'snr': 'mean',
    'health': 'mean'
}).sort_values('num_clients', ascending=False)

# Visualización con dos ejes
fig, ax1 = plt.subplots(figsize=(14, 6))

x = range(len(building_quality.head(15)))
ax1.bar(x, building_quality.head(15)['num_clients'], alpha=0.7, label='Densidad (clientes)')
ax1.set_xlabel('Edificio')
ax1.set_ylabel('Número de Clientes', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticks(x)
ax1.set_xticklabels(building_quality.head(15).index, rotation=45)

ax2 = ax1.twinx()
ax2.plot(x, building_quality.head(15)['signal_db'], color='red', marker='o', label='RSSI promedio')
ax2.set_ylabel('RSSI (dBm)', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.axhline(-70, color='darkred', linestyle='--', alpha=0.5)

plt.title('Densidad de Dispositivos vs Calidad de Señal por Edificio')
fig.tight_layout()
plt.show()
```

### 5. Detección de Anomalías (Outliers)

```python
from scipy import stats

# Detección de APs anómalos usando Z-score
df_aps['z_score_cpu'] = stats.zscore(df_aps['cpu_utilization'])
df_aps['z_score_clients'] = stats.zscore(df_aps['client_count'])

# Anomalías: Z-score > 3 (más de 3 desviaciones estándar)
anomalous_aps = df_aps[
    (df_aps['z_score_cpu'].abs() > 3) |
    (df_aps['z_score_clients'].abs() > 3)
][['name', 'client_count', 'cpu_utilization', 'status']]

print("=== APs Anómalos Detectados ===")
print(anomalous_aps)

# Visualización
plt.figure(figsize=(10, 6))
plt.scatter(
    df_aps['client_count'],
    df_aps['cpu_utilization'],
    alpha=0.5,
    label='Normal'
)
plt.scatter(
    anomalous_aps['client_count'],
    anomalous_aps['cpu_utilization'],
    color='red',
    s=100,
    label='Anómalo',
    marker='X'
)
plt.xlabel('Número de Clientes')
plt.ylabel('CPU Utilization (%)')
plt.title('Detección de APs Anómalos')
plt.legend()
plt.tight_layout()
plt.show()
```

### 6. Dashboard Interactivo con Plotly

```python
import plotly.express as px
import plotly.graph_objects as go

# Scatter interactivo
fig = px.scatter(
    df_clients,
    x='signal_db',
    y='speed',
    color='band',
    size='snr',
    hover_data=['associated_device_name', 'os_type', 'health'],
    labels={
        'signal_db': 'RSSI (dBm)',
        'speed': 'Velocidad (Mbps)',
        'band': 'Banda (GHz)'
    },
    title='Relación entre Señal y Velocidad por Banda'
)
fig.show()

# Mapa de calor interactivo por edificio
building_stats = df_clients.copy()
building_stats['building'] = building_stats['associated_device_name'].str.extract(r'AP-([A-Z]+)\d+')[0]

building_pivot = building_stats.groupby(['building', 'band']).size().unstack(fill_value=0)

fig = px.imshow(
    building_pivot.T,
    labels=dict(x="Edificio", y="Banda (GHz)", color="Dispositivos"),
    x=building_pivot.index,
    y=['2.4 GHz', '5 GHz'],
    title='Distribución de Bandas por Edificio',
    aspect='auto'
)
fig.show()
```

---

## Machine Learning y LLMs (Nivel Avanzado)

### 1. Predicción de Carga de APs (Regresión)

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Preparar features
# (Necesitas cargar datos temporales con _file_timestamp)
df_ml = df_aps_temporal.copy()

# Extraer características temporales
df_ml['hour'] = pd.to_datetime(df_ml['_file_timestamp']).dt.hour
df_ml['day_of_week'] = pd.to_datetime(df_ml['_file_timestamp']).dt.dayofweek
df_ml['is_weekend'] = df_ml['day_of_week'].isin([5, 6]).astype(int)

# Extraer edificio
df_ml['building'] = df_ml['name'].str.extract(r'AP-([A-Z]+)\d+')[0]

# One-hot encoding del edificio
df_ml = pd.get_dummies(df_ml, columns=['building'], prefix='building')

# Features y target
feature_cols = ['hour', 'day_of_week', 'is_weekend'] + [col for col in df_ml.columns if col.startswith('building_')]
X = df_ml[feature_cols]
y = df_ml['client_count']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación
print("=== Evaluación del Modelo ===")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} clientes")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print("\n=== Top 10 Features Más Importantes ===")
print(feature_importance)

# Visualización
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Carga Real (clientes)')
plt.ylabel('Carga Predicha (clientes)')
plt.title('Predicción de Carga de Access Points')
plt.tight_layout()
plt.show()
```

### 2. Clustering de Comportamiento de Dispositivos

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Features de comportamiento
behavior_features = df_clients[['signal_db', 'snr', 'speed', 'maxspeed', 'health']].copy()

# Eliminar NaNs
behavior_features = behavior_features.dropna()

# Normalizar
scaler = StandardScaler()
behavior_scaled = scaler.fit_transform(behavior_features)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(behavior_scaled)

behavior_features['cluster'] = clusters

# Análisis de clusters
print("=== Perfil de Clusters ===")
for cluster_id in range(4):
    cluster_data = behavior_features[behavior_features['cluster'] == cluster_id]
    print(f"\n--- Cluster {cluster_id} ({len(cluster_data)} dispositivos) ---")
    print(cluster_data.describe())

# Visualización
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(
    behavior_features['signal_db'],
    behavior_features['speed'],
    c=behavior_features['cluster'],
    cmap='viridis',
    alpha=0.6
)
plt.xlabel('RSSI (dBm)')
plt.ylabel('Velocidad (Mbps)')
plt.title('Clusters de Comportamiento')
plt.colorbar(label='Cluster')

plt.subplot(1, 2, 2)
behavior_features.groupby('cluster').size().plot(kind='bar')
plt.xlabel('Cluster')
plt.ylabel('Número de Dispositivos')
plt.title('Distribución de Clusters')

plt.tight_layout()
plt.show()
```

### 3. Chatbot con RAG (LangChain + Claude/GPT)

```python
# IMPORTANTE: Necesitas API key de Anthropic o OpenAI
# pip install langchain anthropic chromadb

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatAnthropic
from langchain.chains import RetrievalQA

# 1. Preparar documentos (estadísticas del dataset)
documents = []

# Crear resumen textual de cada edificio
for building in df_clients['building'].unique():
    building_data = df_clients[df_clients['building'] == building]

    summary = f"""
    Edificio: {building}
    Total dispositivos: {len(building_data)}
    RSSI promedio: {building_data['signal_db'].mean():.1f} dBm
    Velocidad promedio: {building_data['speed'].mean():.1f} Mbps
    Health promedio: {building_data['health'].mean():.1f}/100
    Distribución de bandas: {building_data['band'].value_counts().to_dict()}
    """
    documents.append(summary)

# 2. Crear embeddings y vector store
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.create_documents(documents)

embeddings = OpenAIEmbeddings()  # O usar sentence-transformers para local
vectorstore = Chroma.from_documents(docs, embeddings)

# 3. Crear chain de QA
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 4. Hacer preguntas
def ask_wifi_assistant(question):
    result = qa_chain({"query": question})
    print(f"Pregunta: {question}")
    print(f"Respuesta: {result['result']}\n")
    return result

# Ejemplos
ask_wifi_assistant("¿Qué edificio tiene mejor calidad de señal?")
ask_wifi_assistant("¿Dónde hay más congestión de dispositivos?")
ask_wifi_assistant("¿Qué edificios tienen velocidades más bajas?")
```

### 4. Sistema de Recomendaciones (Optimización de Canales)

```python
import numpy as np
from scipy.optimize import minimize

# Extraer datos de radios
radios_data = []
for _, ap in df_aps.iterrows():
    for radio in ap['radios']:
        radios_data.append({
            'ap_name': ap['name'],
            'band': radio['band'],
            'channel': int(radio['channel']),
            'utilization': radio['utilization'],
            'tx_power': radio['tx_power']
        })

df_radios = pd.DataFrame(radios_data)

# Análisis de congestión por canal
channel_congestion = df_radios.groupby(['band', 'channel']).agg({
    'utilization': 'mean',
    'ap_name': 'count'
}).rename(columns={'ap_name': 'num_aps'})

print("=== Canales Más Congestionados (5 GHz) ===")
print(channel_congestion.loc[1].sort_values('utilization', ascending=False).head(10))

# Recomendaciones de reasignación
def recommend_channel_reassignment(band=1, threshold=50):
    """
    Recomienda APs que deberían cambiar de canal.

    Args:
        band: 1 (5 GHz) o 0 (2.4 GHz)
        threshold: Umbral de utilización para considerar congestionado
    """
    band_radios = df_radios[df_radios['band'] == band]

    # Encontrar canales congestionados
    congested = band_radios[band_radios['utilization'] > threshold]

    # Encontrar canales libres
    channel_util = band_radios.groupby('channel')['utilization'].mean().sort_values()
    free_channels = channel_util[channel_util < threshold].index.tolist()

    recommendations = []
    for _, ap_radio in congested.iterrows():
        current_channel = ap_radio['channel']
        current_util = ap_radio['utilization']

        # Recomendar canal más libre
        if free_channels:
            best_channel = free_channels[0]
            recommendations.append({
                'ap_name': ap_radio['ap_name'],
                'current_channel': current_channel,
                'current_util': current_util,
                'recommended_channel': best_channel,
                'target_util': channel_util[best_channel]
            })

    return pd.DataFrame(recommendations)

recommendations = recommend_channel_reassignment(band=1, threshold=50)
print("\n=== Recomendaciones de Cambio de Canal ===")
print(recommendations.head(10))
```

### 5. Detección de Anomalías con Isolation Forest

```python
from sklearn.ensemble import IsolationForest

# Features de comportamiento
anomaly_features = df_clients[['signal_db', 'snr', 'speed', 'health']].dropna()

# Entrenar Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_labels = iso_forest.fit_predict(anomaly_features)

# -1 = anomalía, 1 = normal
anomaly_features['is_anomaly'] = anomaly_labels == -1

print(f"Dispositivos anómalos detectados: {anomaly_features['is_anomaly'].sum()}")

# Analizar anomalías
anomalies = df_clients.loc[anomaly_features[anomaly_features['is_anomaly']].index]

print("\n=== Características de Dispositivos Anómalos ===")
print(anomalies[['signal_db', 'snr', 'speed', 'health', 'associated_device_name']].describe())

# Visualización
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(
    anomaly_features[~anomaly_features['is_anomaly']]['signal_db'],
    anomaly_features[~anomaly_features['is_anomaly']]['speed'],
    alpha=0.3,
    label='Normal'
)
plt.scatter(
    anomaly_features[anomaly_features['is_anomaly']]['signal_db'],
    anomaly_features[anomaly_features['is_anomaly']]['speed'],
    color='red',
    alpha=0.7,
    label='Anomalía'
)
plt.xlabel('RSSI (dBm)')
plt.ylabel('Velocidad (Mbps)')
plt.title('Detección de Anomalías - Isolation Forest')
plt.legend()

plt.subplot(1, 2, 2)
anomaly_features.boxplot(column='health', by='is_anomaly', ax=plt.gca())
plt.xlabel('Es Anomalía')
plt.ylabel('Health Score')
plt.title('Health Score: Normal vs Anómalo')
plt.suptitle('')

plt.tight_layout()
plt.show()
```

---

## Tips y Trucos

### Optimización de Memoria

```python
# Cargar solo columnas necesarias
import json

def load_json_columns(file_path, columns):
    """Carga solo columnas específicas de un JSON."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Filtrar solo columnas requeridas
    filtered = []
    for record in data:
        filtered.append({k: record.get(k) for k in columns})

    return pd.DataFrame(filtered)

# Ejemplo
df_clients_light = load_json_columns(
    'anonymized_data/client-info-XXX.json',
    columns=['macaddr', 'associated_device_name', 'signal_db', 'speed']
)
```

### Procesamiento Paralelo

```python
from multiprocessing import Pool
from functools import partial

def process_file(file_path):
    """Procesa un archivo individual."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    # ... procesamiento ...
    return result

def parallel_load(directory, pattern, n_workers=4):
    """Carga archivos en paralelo."""
    files = list(Path(directory).glob(pattern))

    with Pool(processes=n_workers) as pool:
        results = pool.map(process_file, files)

    return pd.concat(results)

# Usar
df = parallel_load('anonymized_data/', 'client-info-*.json', n_workers=8)
```

### Exportar Resultados

```python
# CSV
df_results.to_csv('results/hotspots_analysis.csv', index=False)

# Excel con múltiples hojas
with pd.ExcelWriter('results/wifi_analysis.xlsx') as writer:
    hotspots.to_excel(writer, sheet_name='Hotspots')
    ap_quality.to_excel(writer, sheet_name='Quality')
    building_quality.to_excel(writer, sheet_name='Building Stats')

# JSON
df_results.to_json('results/analysis.json', orient='records', indent=2)

# Pickle (para guardar DataFrames grandes rápidamente)
df.to_pickle('processed_data/clients_all.pkl')
# Cargar: df = pd.read_pickle('processed_data/clients_all.pkl')
```

---

## Problemas Comunes

### 1. "FileNotFoundError: No such file or directory"

**Solución:**
```python
from pathlib import Path

# Verificar que el archivo existe
file_path = Path('anonymized_data/AP-info-v2-XXX.json')
if not file_path.exists():
    print(f"Archivo no encontrado: {file_path}")
    print("Archivos disponibles:")
    for f in Path('anonymized_data/').glob('*.json'):
        print(f" - {f.name}")
```

### 2. "JSONDecodeError: Expecting value"

**Solución:** Archivo JSON corrupto o vacío
```python
import json

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(f"Error parseando {file_path}: {e}")
    # Intentar leer línea por línea para identificar el problema
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                json.loads(line)
            except:
                print(f"Error en línea {i}: {line[:100]}")
```

### 3. "MemoryError" al cargar todos los archivos

**Solución:** Cargar en chunks o usar Dask
```python
import dask.dataframe as dd

# Opción 1: Cargar solo muestra aleatoria
df_sample = load_random_sample('anonymized_data/', 'client-info-*.json', n_files=50)

# Opción 2: Usar Dask para procesamiento lazy
# (Requiere: pip install dask)
# ddf = dd.read_json('anonymized_data/client-info-*.json')
# result = ddf.groupby('associated_device_name').size().compute()
```

### 4. KeyError al acceder a campos

**Solución:** Algunos registros pueden no tener todos los campos
```python
# Opción 1: Usar .get() con default
ap_name = record.get('name', 'UNKNOWN')

# Opción 2: Verificar existencia
if 'down_reason' in record:
    print(f"AP caído: {record['down_reason']}")

# Opción 3: Usar fillna en DataFrame
df['down_reason'] = df['down_reason'].fillna('N/A')
```

### 5. Timestamps no se convierten correctamente

**Solución:** APs usan segundos, Clientes usan milisegundos
```python
# APs (segundos)
df_aps['datetime'] = pd.to_datetime(df_aps['last_modified'], unit='s')

# Clientes (milisegundos)
df_clients['datetime'] = pd.to_datetime(df_clients['last_connection_time'], unit='ms')
```

---

## Recursos

### Documentación del Proyecto

- `README.md` - Visión general y niveles del challenge
- `DATA_DICTIONARY.md` - Referencia completa de campos
- `ANONYMIZATION_STRATEGY.md` - Detalles de privacidad

### Bibliotecas Recomendadas

**Análisis Básico:**
- pandas: https://pandas.pydata.org/docs/
- matplotlib: https://matplotlib.org/stable/contents.html
- seaborn: https://seaborn.pydata.org/

**Análisis Avanzado:**
- NetworkX: https://networkx.org/ (grafos de movilidad)
- Plotly: https://plotly.com/python/ (dashboards interactivos)
- Folium: https://python-visualization.github.io/folium/ (mapas)

**Machine Learning:**
- scikit-learn: https://scikit-learn.org/stable/
- PyTorch: https://pytorch.org/docs/stable/index.html
- TensorFlow: https://www.tensorflow.org/api_docs

**LLMs y RAG:**
- LangChain: https://python.langchain.com/
- Anthropic Claude API: https://docs.anthropic.com/
- OpenAI API: https://platform.openai.com/docs/

### Tutoriales Externos

- **WiFi Signal Strength**: https://www.metageek.com/training/resources/wifi-signal-strength-basics/
- **NetworkX Tutorial**: https://networkx.org/documentation/stable/tutorial.html
- **Time Series con Pandas**: https://pandas.pydata.org/docs/user_guide/timeseries.html
- **RAG con LangChain**: https://python.langchain.com/docs/use_cases/question_answering/

### Contacto

**Durante el hackathon:**
- Busca a los mentores de DTIC en el stand
- Preguntas técnicas: albert.gil.lopez@uab.cat

**Responsable técnico:**
- Gonçal Badenes Guia (goncal.badenes@uab.cat)

---

## Checklist para Empezar

- [ ] Instalar Python >= 3.8
- [ ] Instalar dependencias (`pip install -r requirements.txt`)
- [ ] Descargar dataset completo desde Google Drive
- [ ] Verificar que puedes cargar un archivo de muestra
- [ ] Leer `DATA_DICTIONARY.md` completo
- [ ] Explorar un notebook de `starter_kits/`
- [ ] Elegir tu nivel (Rookie / Intermedio / Avanzado)
- [ ] Formar equipo y definir objetivo
- [ ] ¡Hackear y divertirse!

---

**¡Buena suerte en el hackathon! 🚀**

**Última actualización:** 6 de noviembre de 2025
**Versión:** 1.0
