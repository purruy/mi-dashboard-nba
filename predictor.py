import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PronosticoEstadisticas:
    """
    Clase para pronosticar estadísticas esperadas (xRebound, xAssists, xPoints)
    utilizando múltiples metodologías estadísticas.
    """
    
    def __init__(self):
        """Inicializa los datos del jugador Coby White y equipos."""
        # Datos del jugador Coby White
        self.jugador = {
            'nombre': 'Coby White',
            'equipo': 'CHI',
            'oponente': 'NO',
            'minutos_estimados': 29,  # Tiempo promedio de Coby White
            'ventaja_local': False,  # NO es local
            
            # Estadísticas base por 36 minutos (estimadas de los datos)
            'rebotes_36min': 4.2,  # Estimado basado en Av. Def Reb
            'asistencias_36min': 6.8,  # Estimado (promedio de base)
            'puntos_36min': 18.5,  # Estimado de FGA+FTA y porcentajes
            
            # Porcentajes de tiro de los datos proporcionados
            'fg2_percent': 0.537,  # 53.7%
            'fg3_percent': 0.238,  # 23.8%
            'ft_percent': 0.800,  # 80.00%
            
            # Distribución de posesiones de los datos
            'poss_2p': 0.356,  # 35.6%
            'poss_3p': 0.301,  # 30.1%
            'poss_ft': 0.343,  # 34.3%
            
            # Factores de ajuste de los datos
            'pace_adj': 1.02,
            'off_reb_adj': 1.08,
            'def_reb_adj': 0.99,
            'points_adj': 1.05,
            'home_adj': 1.00  # No tiene ventaja local
        }
        
        # Estadísticas de equipos (actualizadas)
        self.estadisticas_equipos = {
            'NO': {
                'off_pts_game': 114.2,
                'def_pts_game': 123.5,
                'off_efg': 0.522,
                'def_efg': 0.582,
                'off_reb_pct': 0.275,
                'def_reb_pct': 1 - 0.237,  # NO defiende vs CHI off reb 23.7%
                'assists_game': 24.8,
                'opp_assists_game': 28.5,  # Asistencias permitidas
                'pace': 98.5,  # Estimado
                'def_rating': 115.0  # Estimado
            },
            'CHI': {
                'off_pts_game': 117.8,
                'def_pts_game': 122.7,
                'off_efg': 0.546,
                'def_efg': 0.551,
                'off_reb_pct': 0.237,
                'def_reb_pct': 1 - 0.275,  # CHI defiende vs NO off reb 27.5%
                'assists_game': 28.5,
                'opp_assists_game': 24.8,  # Asistencias permitidas
                'pace': 100.0,  # Estimado
                'def_rating': 114.0  # Estimado
            }
        }
        
        # Calcular estadísticas más precisas basadas en los datos
        self._calcular_stats_precisas()
        
        # Datos históricos simulados para Machine Learning
        self._generar_datos_historicos()
    
    def _calcular_stats_precisas(self):
        """Calcula estadísticas más precisas basadas en los datos de Coby White."""
        # Total de posesiones de tiros por 36 minutos
        fga_fta_per_36 = 23.6  # Av. FGA+FTA del dato
        
        # Calcular puntos por 36 minutos basados en distribución
        poss_total = fga_fta_per_36
        poss_2p = poss_total * self.jugador['poss_2p']
        poss_3p = poss_total * self.jugador['poss_3p']
        poss_ft = poss_total * self.jugador['poss_ft'] / 2  # ~2 FTA por posesión
        
        pts_2p = poss_2p * 2 * self.jugador['fg2_percent']
        pts_3p = poss_3p * 3 * self.jugador['fg3_percent']
        pts_ft = poss_ft * 1 * self.jugador['ft_percent']
        
        puntos_36 = pts_2p + pts_3p + pts_ft
        self.jugador['puntos_36min'] = round(puntos_36, 1)
        
        # Estimación de rebotes basada en Av. Def Reb
        # Asumiendo que Av. Def Reb = 28 es por 100 posesiones
        # Convertimos a por 36 minutos
        self.jugador['rebotes_36min'] = round(28 * 0.36 * 0.5, 1)  # Factor estimado
        
        # Asistencias estimadas basadas en perfil de base
        self.jugador['asistencias_36min'] = 6.8
    
    def _generar_datos_historicos(self):
        """Genera datos históricos simulados para entrenamiento."""
        np.random.seed(42)
        n_muestras = 100
        
        # Variables: minutos, pace_opp, def_rating_opp, home, reb_skill, ast_skill, pts_skill
        self.X_hist = np.random.randn(n_muestras, 7)
        self.X_hist[:, 0] = np.random.uniform(20, 35, n_muestras)  # Minutos (ajustado para base)
        self.X_hist[:, 1] = np.random.uniform(95, 105, n_muestras)  # Ritmo oponente
        self.X_hist[:, 2] = np.random.uniform(110, 120, n_muestras)  # Rating defensivo oponente
        self.X_hist[:, 3] = np.random.randint(0, 2, n_muestras)  # Local/Visitante
        self.X_hist[:, 4] = np.random.uniform(0.6, 1.0, n_muestras)  # Habilidad rebotes (base)
        self.X_hist[:, 5] = np.random.uniform(0.9, 1.3, n_muestras)  # Habilidad asistencias (base)
        self.X_hist[:, 6] = np.random.uniform(0.8, 1.2, n_muestras)  # Habilidad puntos (anotador)
        
        # Valores objetivo (simulados para perfil de base)
        self.y_reb_hist = (
            0.15 * self.X_hist[:, 0] +  # Minutos
            0.04 * self.X_hist[:, 1] -  # Ritmo (positivo)
            0.02 * self.X_hist[:, 2] +  # Def rating (negativo)
            0.3 * self.X_hist[:, 3] +  # Ventaja local
            2.5 * self.X_hist[:, 4] +  # Habilidad
            np.random.randn(n_muestras) * 1.2  # Ruido
        )
        
        self.y_ast_hist = (
            0.20 * self.X_hist[:, 0] +  # Minutos
            0.07 * self.X_hist[:, 1] -  # Ritmo (positivo)
            0.03 * self.X_hist[:, 2] +  # Def rating (ligeramente negativo)
            0.4 * self.X_hist[:, 3] +  # Ventaja local
            3.0 * self.X_hist[:, 5] +  # Habilidad
            np.random.randn(n_muestras) * 1.5  # Ruido
        )
        
        self.y_pts_hist = (
            0.55 * self.X_hist[:, 0] +  # Minutos
            0.09 * self.X_hist[:, 1] -  # Ritmo (positivo)
            0.12 * self.X_hist[:, 2] +  # Def rating (negativo)
            1.2 * self.X_hist[:, 3] +  # Ventaja local
            5.0 * self.X_hist[:, 6] +  # Habilidad
            np.random.randn(n_muestras) * 2.5  # Ruido
        )
    
    def estimacion_calidad(self):
        """Método 1: Estimación basada en calidad de oponente y ajustes."""
        minutos = self.jugador['minutos_estimados']
        
        # Factores de calidad del oponente
        oponente = self.estadisticas_equipos[self.jugador['oponente']]
        propio = self.estadisticas_equipos[self.jugador['equipo']]
        
        # Factor defensivo del oponente (1.0 = promedio, <1.0 = mejor defensa)
        factor_def_oponente = (oponente['def_rating'] / 115)  # 115 como promedio
        
        # Ajuste por ritmo
        factor_ritmo = oponente['pace'] / 100
        
        # Rebotes esperados
        reb_base = self.jugador['rebotes_36min'] * (minutos / 36)
        xReb = (
            reb_base *
            self.jugador['off_reb_adj'] *
            self.jugador['def_reb_adj'] *
            (1 / factor_def_oponente) *  # Mejor defensa = menos rebotes
            factor_ritmo *
            self.jugador['home_adj']
        )
        
        # Asistencias esperadas
        ast_base = self.jugador['asistencias_36min'] * (minutos / 36)
        xAssists = (
            ast_base *
            (propio['off_efg'] / 0.54) *  # Mejor porcentaje = más asistencias
            (1 / factor_def_oponente) *  # Mejor defensa = menos asistencias
            factor_ritmo *
            self.jugador['home_adj']
        )
        
        # Puntos esperados - cálculo detallado
        # FGA+FTA por minuto del dato
        fga_fta_per_min = 23.6 / 36  # Por minuto
        
        # Posesiones estimadas para los minutos dados
        poss_totales = fga_fta_per_min * minutos
        
        # Distribución por tipo de tiro
        poss_2p = poss_totales * self.jugador['poss_2p']
        poss_3p = poss_totales * self.jugador['poss_3p']
        poss_ft = poss_totales * self.jugador['poss_ft'] / 2  # Aprox 2 FTA por posesión
        
        # Puntos por tipo de tiro
        pts_2p = poss_2p * 2 * self.jugador['fg2_percent']
        pts_3p = poss_3p * 3 * self.jugador['fg3_percent']
        pts_ft = poss_ft * 1 * self.jugador['ft_percent']
        
        xPoints = (
            (pts_2p + pts_3p + pts_ft) *
            self.jugador['points_adj'] *
            (1 / factor_def_oponente) *  # Mejor defensa = menos puntos
            self.jugador['home_adj']
        )
        
        return {
            'xRebound': round(xReb, 1),
            'xAssists': round(xAssists, 1),
            'xPoints': round(xPoints, 1)
        }
    
    def regresion_lineal(self):
        """Método 2: Regresión lineal basada en minutos y factores."""
        minutos = self.jugador['minutos_estimados']
        
        # Modelos lineales ajustados para Coby White (base)
        xReb = 0.12 * minutos + 1.2 * self.jugador['off_reb_adj'] + 0.5
        xAssists = 0.22 * minutos + 1.5 * (self.jugador['pace_adj'] - 1) * 10 + 1.2
        xPoints = 0.65 * minutos + 3.0 * (self.jugador['points_adj'] - 1) * 10 + 3.5
        
        # Ajuste por oponente
        oponente = self.estadisticas_equipos[self.jugador['oponente']]
        factor_def = oponente['def_rating'] / 114
        
        xReb *= (1 / factor_def) * 0.95
        xAssists *= (1 / factor_def) * 0.97
        xPoints *= (1 / factor_def) * 0.92  # Más ajuste para puntos
        
        # Ajuste por NO ser local
        if not self.jugador['ventaja_local']:
            xReb *= 0.98
            xAssists *= 0.97
            xPoints *= 0.96
        
        return {
            'xRebound': round(xReb, 1),
            'xAssists': round(xAssists, 1),
            'xPoints': round(xPoints, 1)
        }
    
    def machine_learning(self):
        """Método 3: Predicción usando modelo de machine learning (simulado)."""
        # Preparar datos de entrada
        oponente = self.estadisticas_equipos[self.jugador['oponente']]
        
        X_input = np.array([[
            self.jugador['minutos_estimados'],  # Minutos
            oponente['pace'],  # Ritmo del oponente
            oponente['def_rating'],  # Rating defensivo del oponente
            1 if self.jugador['ventaja_local'] else 0,  # Local/Visitante
            0.8,  # Habilidad rebotes (base, no especialista)
            1.2,  # Habilidad asistencias (buen pasador)
            1.1  # Habilidad puntos (buen anotador)
        ]])
        
        # Escalar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_hist)
        X_input_scaled = scaler.transform(X_input)
        
        # Modelos de regresión lineal (simplificado)
        model_reb = LinearRegression()
        model_ast = LinearRegression()
        model_pts = LinearRegression()
        
        model_reb.fit(X_scaled, self.y_reb_hist)
        model_ast.fit(X_scaled, self.y_ast_hist)
        model_pts.fit(X_scaled, self.y_pts_hist)
        
        # Predicciones
        xReb = model_reb.predict(X_input_scaled)[0]
        xAssists = model_ast.predict(X_input_scaled)[0]
        xPoints = model_pts.predict(X_input_scaled)[0]
        
        # Ajustar por factores específicos
        xReb *= self.jugador['off_reb_adj'] * self.jugador['def_reb_adj']
        xAssists *= self.jugador['pace_adj']
        xPoints *= self.jugador['points_adj']
        
        # Ajuste adicional para visitante
        if not self.jugador['ventaja_local']:
            xReb *= 0.97
            xAssists *= 0.96
            xPoints *= 0.95
        
        return {
            'xRebound': round(max(0, xReb), 1),
            'xAssists': round(max(0, xAssists), 1),
            'xPoints': round(max(0, xPoints), 1)
        }
    
    def expected_value_plus(self):
        """Método 4: Expected Value Plus con factores contextuales."""
        # Obtener estimaciones base de otros métodos
        calidad = self.estimacion_calidad()
        lineal = self.regresion_lineal()
        ml = self.machine_learning()
        
        # Factores contextuales adicionales
        oponente = self.estadisticas_equipos[self.jugador['oponente']]
        propio = self.estadisticas_equipos[self.jugador['equipo']]
        
        # 1. Factor de ritmo del juego
        factor_ritmo = self.jugador['pace_adj']
        
        # 2. Factor de eficiencia defensiva del oponente
        # NO tiene peor defensa (58.2% eFG permitido) vs CHI (55.1%)
        factor_def_efg = (oponente['def_efg'] - 0.54) * 8  # Desviación del promedio
        
        # 3. Factor de rebotes
        # CHI tiene 23.7% de rebotes ofensivos (bajo)
        factor_reb_off_team = (propio['off_reb_pct'] - 0.25) * 4
        
        # 4. Factor de asistencias para base
        factor_ast_base = 1.15  # Bases tienden a dar más asistencias
        
        # 5. Factor de visitante (negativo)
        factor_visitante = 0.94 if not self.jugador['ventaja_local'] else 1.0
        
        # Calcular EV+ con pesos ajustados
        peso_calidad = 0.30  # Más peso a calidad por datos concretos
        peso_lineal = 0.25
        peso_ml = 0.45  # Mayor peso a ML
        
        xRebound = (
            peso_calidad * calidad['xRebound'] +
            peso_lineal * lineal['xRebound'] +
            peso_ml * ml['xRebound']
        ) * (1 + 0.08 * factor_reb_off_team) * factor_ritmo * factor_visitante
        
        xAssists = (
            peso_calidad * calidad['xAssists'] +
            peso_lineal * lineal['xAssists'] +
            peso_ml * ml['xAssists']
        ) * factor_ast_base * factor_ritmo * factor_visitante
        
        xPoints = (
            peso_calidad * calidad['xPoints'] +
            peso_lineal * lineal['xPoints'] +
            peso_ml * ml['xPoints']
        ) * (1 - 0.06 * factor_def_efg) * factor_ritmo * factor_visitante
        
        return {
            'xRebound': round(xRebound, 1),
            'xAssists': round(xAssists, 1),
            'xPoints': round(xPoints, 1)
        }
    
    def calcular_promedio_ponderado(self):
        """Calcula el promedio ponderado de todas las metodologías."""
        # Obtener todas las predicciones
        resultados = {
            'Calidad': self.estimacion_calidad(),
            'Lineal': self.regresion_lineal(),
            'ML': self.machine_learning(),
            'EV+': self.expected_value_plus()
        }
        
        # Pesos para cada metodología (ajustados para Coby)
        pesos = {'Calidad': 0.25, 'Lineal': 0.25, 'ML': 0.30, 'EV+': 0.20}
        
        # Calcular promedios ponderados
        xRebound_total = 0
        xAssists_total = 0
        xPoints_total = 0
        
        for metodo, peso in pesos.items():
            xRebound_total += resultados[metodo]['xRebound'] * peso
            xAssists_total += resultados[metodo]['xAssists'] * peso
            xPoints_total += resultados[metodo]['xPoints'] * peso
        
        return {
            'xRebound': round(xRebound_total, 1),
            'xAssists': round(xAssists_total, 1),
            'xPoints': round(xPoints_total, 1),
            'desglose': resultados
        }
    
    def imprimir_resultados(self):
        """Imprime los resultados del pronóstico."""
        print("=" * 60)
        print(f"PRONÓSTICO DE ESTADÍSTICAS - {self.jugador['nombre']}")
        print(f"VS {self.jugador['oponente']} | {self.jugador['minutos_estimados']} MIN ESTIMADOS")
        print(f"LOCAL: {'Sí' if self.jugador['ventaja_local'] else 'No'}")
        print("=" * 60)
        
        resultados = self.calcular_promedio_ponderado()
        
        print(f"\nRESULTADOS FINALES (Promedio Ponderado):")
        print(f"  xRebound (Rebotes esperados): {resultados['xRebound']}")
        print(f"  xAssists (Asistencias esperadas): {resultados['xAssists']}")
        print(f"  xPoints (Puntos esperados): {resultados['xPoints']}")
        
        print(f"\nDESGLOSE POR METODOLOGÍA:")
        for metodo, stats in resultados['desglose'].items():
            print(f"\n  {metodo}:")
            print(f"    Rebotes: {stats['xRebound']}")
            print(f"    Asistencias: {stats['xAssists']}")
            print(f"    Puntos: {stats['xPoints']}")
        
        print(f"\nFACTORES DE AJUSTE APLICADOS:")
        print(f"  Ventaja local: {'Sí' if self.jugador['home_adj'] > 1 else 'No'}")
        print(f"  Ajuste de ritmo (PACE): {self.jugador['pace_adj']}")
        print(f"  Ajuste rebotes ofensivos: {self.jugador['off_reb_adj']}")
        print(f"  Ajuste rebotes defensivos: {self.jugador['def_reb_adj']}")
        print(f"  Ajuste de puntos: {self.jugador['points_adj']}")
        print(f"  % Tiros de 2: {self.jugador['fg2_percent']*100:.1f}%")
        print(f"  % Tiros de 3: {self.jugador['fg3_percent']*100:.1f}%")
        print(f"  % Tiros libres: {self.jugador['ft_percent']*100:.1f}%")
        
        print("\n" + "=" * 60)

# Ejecutar el pronóstico
if __name__ == "__main__":
    print("Calculando pronóstico para Coby White...\n")
    pronostico = PronosticoEstadisticas()
    pronostico.imprimir_resultados()
    
    # Generar también un archivo CSV con los resultados
    resultados = pronostico.calcular_promedio_ponderado()
    
    df_resultados = pd.DataFrame(resultados['desglose']).T
    df_resultados.loc['PROMEDIO'] = [
        resultados['xRebound'],
        resultados['xAssists'],
        resultados['xPoints']
    ]
    
    df_resultados.columns = ['xRebound', 'xAssists', 'xPoints']
    df_resultados.to_csv('pronostico_coby_white.csv')
    print("\nResultados guardados en 'pronostico_coby_white.csv'")
    
    # Mostrar resultados finales para el HTML
    print("\n" + "=" * 60)
    print("VALORES PARA HTML:")
    print(f"xRebound: {resultados['xRebound']}")
    print(f"xAssists: {resultados['xAssists']}")
    print(f"xPoints: {resultados['xPoints']}")
    
    print("\nDesglose para HTML:")
    for metodo, stats in resultados['desglose'].items():
        print(f"{metodo}: R{stats['xRebound']} A{stats['xAssists']} P{stats['xPoints']}")
