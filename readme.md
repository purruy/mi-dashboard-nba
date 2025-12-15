# Proyecto: Pronóstico de Estadísticas Deportivas

## Descripción
Sistema de pronóstico de estadísticas esperadas (xRebound, xAssists, xPoints) para jugadores de baloncesto utilizando múltiples metodologías estadísticas avanzadas.

## Características
- **4 Metodologías de Pronóstico:**
  1. Estimación de Calidad
  2. Regresión Lineal
  3. Machine Learning
  4. EV+ (Expected Value Plus)

- **Factores Considerados:**
  - Minutos de juego estimados
  - Ventaja de local/visitante
  - Estadísticas ofensivas/defensivas de equipos
  - Ritmo de juego (PACE)
  - Porcentajes de tiro del jugador
  - Factores de ajuste contextual

## Archivos del Proyecto

### 1. `index.html`
Interfaz web interactiva que muestra:
- Pronósticos de xRebound, xAssists y xPoints
- Comparativa de estadísticas entre equipos
- Desglose por metodología
- Visualizaciones atractivas con animaciones

### 2. `pronostico_estadisticas.py`
Script Python que implementa:
- Clase `PronosticoEstadisticas` con los 4 métodos
- Generación de datos históricos simulados
- Cálculo de promedios ponderados
- Exportación de resultados a CSV

### 3. `pronostico_giddey.csv` (generado)
Resultados del pronóstico en formato CSV

## Cómo Usar

### Para la interfaz web:
1. Abrir `index.html` en cualquier navegador web
2. Ver los pronósticos con animación incluida

### Para el análisis Python:
```python
# Ejecutar el script
python pronostico_estadisticas.py

# O importar la clase
from pronostico_estadisticas import PronosticoEstadisticas

pronostico = PronosticoEstadisticas()
resultados = pronostico.calcular_promedio_ponderado()
