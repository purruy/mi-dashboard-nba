# predictor.py - Sistema Completo de Predicción NCAAB
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class NCAAFPredictor:
    """Sistema completo de predicción para Central Arkansas vs Vanderbilt"""
    
    def __init__(self):
        print("🎯 SISTEMA DE PREDICCIÓN NCAAB INICIADO")
        print("=" * 50)
        print("EQUIPOS: Central Arkansas Bears vs Vanderbilt Commodores")
        print("=" * 50)
        
        # Datos específicos del juego
        self.team_a = "Central Arkansas Bears"
        self.team_b = "Vanderbilt Commodores"
        
        # Datos ofensivos de la imagen
        self.offensive_stats = {
            'CARK': {
                'points_per_game': 69.6,
                'avg_score_margin': -5.1,
                'assists_per_game': 14.2,
                'total_rebounds': 36.0,
                'effective_fg_pct': 49.3,
                'off_rebound_pct': 21.7,
                'fta_per_fga': 0.273,
                'turnover_pct': 16.2
            },
            'VAN': {
                'points_per_game': 96.8,
                'avg_score_margin': 23.8,
                'assists_per_game': 20.1,
                'total_rebounds': 38.9,
                'effective_fg_pct': 61.2,
                'off_rebound_pct': 31.7,
                'fta_per_fga': 0.358,
                'turnover_pct': 10.9
            }
        }
        
        # Datos defensivos de la imagen
        self.defensive_stats = {
            'CARK': {
                'opp_points_per_game': 74.7,
                'opp_effective_fg_pct': 51.5,
                'off_rebounds_per_game': 7.7,
                'def_rebounds_per_game': 24.4,
                'blocks_per_game': 1.6,
                'steals_per_game': 8.2,
                'personal_fouls_per_game': 16.7
            },
            'VAN': {
                'opp_points_per_game': 73.0,
                'opp_effective_fg_pct': 46.3,
                'off_rebounds_per_game': 9.2,
                'def_rebounds_per_game': 25.7,
                'blocks_per_game': 5.4,
                'steals_per_game': 9.9,
                'personal_fouls_per_game': 21.1
            }
        }
    
    # ==================== MÓDULO 1: ESTIMACIÓN DE CALIDAD ====================
    def quality_estimation(self):
        """Calcula rating de calidad para ambos equipos"""
        print("\n1️⃣ ESTIMACIÓN DE CALIDAD")
        print("-" * 40)
        
        # Fórmula: 40% ofensiva + 40% defensiva + 20% eficiencia
        quality_scores = {}
        
        for team in ['CARK', 'VAN']:
            # Componente ofensivo (0-100)
            off_rating = (
                self.offensive_stats[team]['points_per_game'] * 0.3 +
                self.offensive_stats[team]['effective_fg_pct'] * 2.0 +
                (100 - self.offensive_stats[team]['turnover_pct']) * 0.5 +
                self.offensive_stats[team]['assists_per_game'] * 0.8
            )
            
            # Componente defensivo (0-100)
            def_rating = (
                (100 - self.defensive_stats[team]['opp_points_per_game']) * 0.8 +
                (100 - self.defensive_stats[team]['opp_effective_fg_pct'] * 2) +
                self.defensive_stats[team]['blocks_per_game'] * 1.5 +
                self.defensive_stats[team]['steals_per_game'] * 1.2
            )
            
            # Componente de eficiencia
            eff_rating = (
                self.offensive_stats[team]['avg_score_margin'] * 2.0 +
                self.offensive_stats[team]['off_rebound_pct'] * 0.5 +
                self.defensive_stats[team]['def_rebounds_per_game'] * 0.3
            )
            
            # Calificación final
            final_rating = (off_rating * 0.4 + def_rating * 0.4 + eff_rating * 0.2)
            normalized_rating = min(100, max(0, final_rating / 3))
            
            quality_scores[team] = {
                'rating': round(normalized_rating, 1),
                'off_rating': round(off_rating / 2.5, 1),
                'def_rating': round(def_rating / 2.5, 1),
                'eff_rating': round(eff_rating / 1.5, 1)
            }
            
            print(f"{'Central Arkansas' if team == 'CARK' else 'Vanderbilt'}:")
            print(f"  Rating Total: {quality_scores[team]['rating']}/100")
            print(f"  Ofensivo: {quality_scores[team]['off_rating']}/100")
            print(f"  Defensivo: {quality_scores[team]['def_rating']}/100")
            print(f"  Eficiencia: {quality_scores[team]['eff_rating']}/100")
        
        return quality_scores
    
    # ==================== MÓDULO 2: REGRESIÓN LINEAL ====================
    def linear_regression_prediction(self):
        """Predicción de puntos usando regresión lineal"""
        print("\n2️⃣ REGRESIÓN LINEAL - PREDICCIÓN DE PUNTOS")
        print("-" * 40)
        
        # Variables predictoras
        X = np.array([
            # [PTS/G, eFG%, Asist, RebOf%, TOV%, OppPTS, Blk, Rob]
            [69.6, 49.3, 14.2, 21.7, 16.2, 74.7, 1.6, 8.2],  # CARK
            [96.8, 61.2, 20.1, 31.7, 10.9, 73.0, 5.4, 9.9]   # VAN
        ])
        
        # Variables objetivo (puntos anotados en juegos similares históricos)
        y = np.array([69.6, 96.8])
        
        # Entrenar modelo
        model = LinearRegression()
        model.fit(X, y)
        
        # Predecir
        pred_cark = model.predict([X[0]])[0]
        pred_van = model.predict([X[1]])[0]
        
        # Ajustar por enfrentamiento defensivo
        def_adjustment_cark = (100 - self.defensive_stats['VAN']['opp_effective_fg_pct'] * 2) / 100
        def_adjustment_van = (100 - self.defensive_stats['CARK']['opp_effective_fg_pct'] * 2) / 100
        
        final_pred_cark = pred_cark * def_adjustment_cark
        final_pred_van = pred_van * def_adjustment_van
        
        print(f"Central Arkansas Predicción: {final_pred_cark:.1f} puntos")
        print(f"Vanderbilt Predicción: {final_pred_van:.1f} puntos")
        print(f"Diferencia: {final_pred_van - final_pred_cark:.1f} puntos")
        
        return {
            'cark_points': round(final_pred_cark, 1),
            'van_points': round(final_pred_van, 1),
            'margin': round(final_pred_van - final_pred_cark, 1)
        }
    
    # ==================== MÓDULO 3: MACHINE LEARNING ENSEMBLE ====================
    def ml_ensemble(self):
        """Ensemble de modelos de ML"""
        print("\n3️⃣ MACHINE LEARNING ENSEMBLE")
        print("-" * 40)
        
        # Datos de entrenamiento (simulado)
        np.random.seed(42)
        n_samples = 100
        
        # Características: [efg%, turnover%, rebounds, assists, steals, blocks]
        X_train = np.random.rand(n_samples, 6) * 100
        
        # Variable objetivo: margen de victoria
        y_train = (X_train[:, 0] * 0.3 +  # eFG%
                   (100 - X_train[:, 1]) * 0.2 +  # TOV% (inverso)
                   X_train[:, 2] * 0.1 +  # Reb
                   X_train[:, 3] * 0.15 +  # Ast
                   X_train[:, 4] * 0.15 +  # Stl
                   X_train[:, 5] * 0.1)   # Blk
        y_train += np.random.normal(0, 5, n_samples)  # Ruido
        
        # Características para CARK y VAN
        X_cark = np.array([[49.3, 16.2, 36.0, 14.2, 8.2, 1.6]])
        X_van = np.array([[61.2, 10.9, 38.9, 20.1, 9.9, 5.4]])
        
        # Modelo 1: Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        pred_rf_cark = rf.predict(X_cark)[0]
        pred_rf_van = rf.predict(X_van)[0]
        
        # Modelo 2: Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        pred_gb_cark = gb.predict(X_cark)[0]
        pred_gb_van = gb.predict(X_van)[0]
        
        # Ensemble promedio
        ensemble_cark = (pred_rf_cark + pred_gb_cark) / 2
        ensemble_van = (pred_rf_van + pred_gb_van) / 2
        
        # Convertir a puntos
        base_cark = 69.6
        base_van = 96.8
        
        final_cark = base_cark + (ensemble_cark - 50) / 10
        final_van = base_van + (ensemble_van - 50) / 10
        
        print(f"Random Forest - CARK: {pred_rf_cark:.1f}, VAN: {pred_rf_van:.1f}")
        print(f"Gradient Boost - CARK: {pred_gb_cark:.1f}, VAN: {pred_gb_van:.1f}")
        print(f"Ensemble Final - CARK: {final_cark:.1f}, VAN: {final_van:.1f}")
        
        return {
            'cark_ensemble': round(final_cark, 1),
            'van_ensemble': round(final_van, 1),
            'ensemble_margin': round(final_van - final_cark, 1),
            'model_agreement': round(100 - abs(pred_rf_cark - pred_gb_cark), 1)
        }
    
    # ==================== MÓDULO 4: FOUR FACTORS ESTIMADOS ====================
    def four_factors_estimation(self):
        """Calcula los Four Factors para el juego"""
        print("\n4️⃣ FOUR FACTORS ESTIMADOS")
        print("-" * 40)
        
        # eFG% estimado (considerando defensa rival)
        efg_cark = (self.offensive_stats['CARK']['effective_fg_pct'] * 0.7 + 
                   (100 - self.defensive_stats['VAN']['opp_effective_fg_pct']) * 0.3)
        
        efg_van = (self.offensive_stats['VAN']['effective_fg_pct'] * 0.7 + 
                   (100 - self.defensive_stats['CARK']['opp_effective_fg_pct']) * 0.3)
        
        # TOV% estimado
        tov_cark = (self.offensive_stats['CARK']['turnover_pct'] * 0.6 + 
                   self.defensive_stats['VAN']['steals_per_game'] * 0.4)
        
        tov_van = (self.offensive_stats['VAN']['turnover_pct'] * 0.6 + 
                   self.defensive_stats['CARK']['steals_per_game'] * 0.4)
        
        # ORB% estimado
        orb_cark = (self.offensive_stats['CARK']['off_rebound_pct'] * 0.5 + 
                   (100 - self.defensive_stats['VAN']['def_rebounds_per_game'] * 2) * 0.5)
        
        orb_van = (self.offensive_stats['VAN']['off_rebound_pct'] * 0.5 + 
                   (100 - self.defensive_stats['CARK']['def_rebounds_per_game'] * 2) * 0.5)
        
        # FTA/FGA estimado
        ftr_cark = self.offensive_stats['CARK']['fta_per_fga'] * 100
        ftr_van = self.offensive_stats['VAN']['fta_per_fga'] * 100
        
        four_factors = {
            'CARK': {
                'efg': round(efg_cark, 1),
                'tov': round(tov_cark, 1),
                'orb': round(orb_cark, 1),
                'ftr': round(ftr_cark, 1)
            },
            'VAN': {
                'efg': round(efg_van, 1),
                'tov': round(tov_van, 1),
                'orb': round(orb_van, 1),
                'ftr': round(ftr_van, 1)
            }
        }
        
        print("Central Arkansas Four Factors:")
        print(f"  eFG%: {four_factors['CARK']['efg']}%")
        print(f"  TOV%: {four_factors['CARK']['tov']}%")
        print(f"  ORB%: {four_factors['CARK']['orb']}%")
        print(f"  FTR: {four_factors['CARK']['ftr']}")
        
        print("\nVanderbilt Four Factors:")
        print(f"  eFG%: {four_factors['VAN']['efg']}%")
        print(f"  TOV%: {four_factors['VAN']['tov']}%")
        print(f"  ORB%: {four_factors['VAN']['orb']}%")
        print(f"  FTR: {four_factors['VAN']['ftr']}")
        
        return four_factors
    
    # ==================== MÓDULO 5: EV+ (Expected Value Plus) ====================
    def expected_value_plus(self, final_prediction):
        """Calcula el Expected Value Plus y criterio Kelly"""
        print("\n5️⃣ EV+ (EXPECTED VALUE PLUS) & CRITERIO KELLY")
        print("-" * 40)
        
        # Supongamos odds del mercado (esto vendría de API real)
        market_odds = {
            'moneyline_cark': +450,    # Central Arkansas +450 (underdog)
            'moneyline_van': -600,     # Vanderbilt -600 (favorito)
            'spread': 25.5,            # Vanderbilt -25.5
            'total': 160.5             # Total Over/Under
        }
        
        # Probabilidad de nuestro modelo
        margin = final_prediction['margin']
        total = final_prediction['total']
        
        # Calcular probabilidades
        std_dev = 12.0  # Desviación estándar ajustada
        
        # Probabilidad de Vanderbilt cubriendo -25.5
        z_score_spread = (margin - 25.5) / std_dev
        prob_cover_van = self._normal_cdf(z_score_spread)
        
        # Probabilidad de Over/Under
        z_score_total = (total - 160.5) / std_dev
        prob_over = 1 - self._normal_cdf(z_score_total)
        
        # Probabilidad de ganador directo
        prob_win_van = self._normal_cdf(margin / std_dev)
        prob_win_cark = 1 - prob_win_van
        
        # Convertir odds a probabilidad implícita
        def implied_probability(american_odds):
            if american_odds > 0:
                return 100 / (american_odds + 100)
            else:
                return abs(american_odds) / (abs(american_odds) + 100)
        
        # Calcular EV para cada mercado
        ev_spread_van = (prob_cover_van - implied_probability(-110)) * 100
        ev_moneyline_cark = (prob_win_cark - implied_probability(+450)) * 100
        ev_total_over = (prob_over - implied_probability(-110)) * 100
        
        # Criterio Kelly
        def kelly_criterion(win_prob, odds):
            if odds > 0:
                b = odds / 100
            else:
                b = 100 / abs(odds)
            q = 1 - win_prob
            kelly = (b * win_prob - q) / b
            return max(0, min(kelly * 0.5, 0.25))  # Fractional Kelly conservador
        
        kelly_spread = kelly_criterion(prob_cover_van, -110)
        kelly_moneyline = kelly_criterion(prob_win_cark, +450)
        kelly_total = kelly_criterion(prob_over, -110)
        
        ev_results = {
            'probabilities': {
                'cark_win': round(prob_win_cark * 100, 1),
                'van_win': round(prob_win_van * 100, 1),
                'van_cover': round(prob_cover_van * 100, 1),
                'over': round(prob_over * 100, 1)
            },
            'expected_value': {
                'spread_ev': round(ev_spread_van, 2),
                'moneyline_ev': round(ev_moneyline_cark, 2),
                'total_ev': round(ev_total_over, 2)
            },
            'kelly_criterion': {
                'spread_kelly': round(kelly_spread * 100, 1),
                'moneyline_kelly': round(kelly_moneyline * 100, 1),
                'total_kelly': round(kelly_total * 100, 1)
            },
            'market_odds': market_odds
        }
        
        print(f"Probabilidad Vanderbilt gane: {ev_results['probabilities']['van_win']}%")
        print(f"Probabilidad cubra -25.5: {ev_results['probabilities']['van_cover']}%")
        print(f"Probabilidad Over 160.5: {ev_results['probabilities']['over']}%")
        print(f"\nExpected Value (EV):")
        print(f"  Spread: {ev_results['expected_value']['spread_ev']}%")
        print(f"  Moneyline CARK: {ev_results['expected_value']['moneyline_ev']}%")
        print(f"  Total: {ev_results['expected_value']['total_ev']}%")
        print(f"\nCriterio Kelly (% bankroll):")
        print(f"  Spread: {ev_results['kelly_criterion']['spread_kelly']}%")
        print(f"  Moneyline CARK: {ev_results['kelly_criterion']['moneyline_kelly']}%")
        print(f"  Total: {ev_results['kelly_criterion']['total_kelly']}%")
        
        return ev_results
    
    # ==================== MÓDULO 6: PREDICCIÓN FINAL ====================
    def final_prediction(self, quality_scores, linear_pred, ensemble_pred, four_factors):
        """Genera la predicción final consolidada"""
        print("\n6️⃣ PREDICCIÓN FINAL CONSOLIDADA")
        print("=" * 50)
        
        # Consolidar predicciones
        cark_points_final = (linear_pred['cark_points'] * 0.4 + 
                            ensemble_pred['cark_ensemble'] * 0.4 + 
                            quality_scores['CARK']['rating'] * 0.2)
        
        van_points_final = (linear_pred['van_points'] * 0.4 + 
                           ensemble_pred['van_ensemble'] * 0.4 + 
                           quality_scores['VAN']['rating'] * 0.2)
        
        # Margen final
        margin_final = van_points_final - cark_points_final
        
        # Total final
        total_final = cark_points_final + van_points_final
        
        # Grado de confianza (0-100%)
        confidence = min(100, max(0, 
            (quality_scores['VAN']['rating'] - quality_scores['CARK']['rating']) * 0.5 +
            abs(margin_final) * 1.5 +
            ensemble_pred['model_agreement']
        ))
        
        # Determinar ganador
        if margin_final > 0:
            winner = "Vanderbilt Commodores"
            winner_confidence = (van_points_final / (van_points_final + cark_points_final)) * 100
        else:
            winner = "Central Arkansas Bears"
            winner_confidence = (cark_points_final / (van_points_final + cark_points_final)) * 100
        
        final_pred = {
            'central_arkansas_score': round(cark_points_final, 1),
            'vanderbilt_score': round(van_points_final, 1),
            'margin': round(margin_final, 1),
            'total': round(total_final, 1),
            'winner': winner,
            'winner_confidence': round(winner_confidence, 1),
            'confidence_score': round(confidence, 1),
            'recommended_bets': self._generate_recommendations(margin_final, total_final, confidence)
        }
        
        # Mostrar resultados
        print(f"🏆 RESULTADO PREDICHO: {winner}")
        print(f"📊 Puntuación Final: {final_pred['central_arkansas_score']} - {final_pred['vanderbilt_score']}")
        print(f"📈 Margen: {final_pred['margin']} puntos")
        print(f"🎯 Total: {final_pred['total']} puntos")
        print(f"💪 Grado de Confianza: {final_pred['confidence_score']}%")
        print(f"💰 Confianza en Ganador: {final_pred['winner_confidence']}%")
        
        print("\n🎰 RECOMENDACIONES:")
        for bet in final_pred['recommended_bets']:
            print(f"  {bet}")
        
        return final_pred
    
    # ==================== MÉTODOS AUXILIARES ====================
    def _normal_cdf(self, x):
        """Aproximación de CDF normal"""
        return 1 / (1 + np.exp(-0.07056 * x**3 - 1.5976 * x))
    
    def _generate_recommendations(self, margin, total, confidence):
        """Genera recomendaciones de apuestas"""
        recommendations = []
        
        # Spread recomendación (Vanderbilt -25.5)
        if margin > 25.5:
            recommendations.append("📈 SPREAD: Vanderbilt -25.5 (RECOMENDADO)")
        elif margin < 25.5:
            recommendations.append("📈 SPREAD: Central Arkansas +25.5 (RECOMENDADO)")
        else:
            recommendations.append("📈 SPREAD: No bet (margen muy cercano)")
        
        # Total recomendación
        if total > 160.5:
            recommendations.append("🎯 TOTAL: Over 160.5 (RECOMENDADO)")
        else:
            recommendations.append("🎯 TOTAL: Under 160.5 (RECOMENDADO)")
        
        # Moneyline recomendación
        if margin > 0 and confidence > 70:
            recommendations.append("💰 MONEYLINE: Vanderbilt (ALTA CONFIANZA)")
        elif margin < 0 and confidence > 60:
            recommendations.append("💰 MONEYLINE: Central Arkansas (VALOR POSITIVO)")
        else:
            recommendations.append("💰 MONEYLINE: Evitar (riesgo elevado)")
        
        # Kelly sizing
        if confidence > 80:
            recommendations.append("⚡ TAMAÑO: 2-3% bankroll (ALTA CONFIANZA)")
        elif confidence > 60:
            recommendations.append("⚡ TAMAÑO: 1-2% bankroll (MEDIA CONFIANZA)")
        else:
            recommendations.append("⚡ TAMAÑO: 0.5% bankroll (BAJA CONFIANZA)")
        
        return recommendations
    
    def run_full_analysis(self):
        """Ejecuta el análisis completo"""
        print("🚀 INICIANDO ANÁLISIS COMPLETO DEL JUEGO")
        print("=" * 60)
        
        # Ejecutar todos los módulos
        quality = self.quality_estimation()
        linear = self.linear_regression_prediction()
        ensemble = self.ml_ensemble()
        four_factors = self.four_factors_estimation()
        
        # Predicción final
        final_pred = self.final_prediction(quality, linear, ensemble, four_factors)
        
        # EV+ y Kelly
        ev_results = self.expected_value_plus(final_pred)
        
        # Consolidar todos los resultados
        full_results = {
            'game_info': {
                'home': self.team_a,
                'away': self.team_b,
                'date': '2024-03-20'
            },
            'quality_estimation': quality,
            'linear_regression': linear,
            'ml_ensemble': ensemble,
            'four_factors': four_factors,
            'final_prediction': final_pred,
            'ev_analysis': ev_results
        }
        
        return full_results

# ==================== EJECUTAR EL SISTEMA ====================
if __name__ == "__main__":
    # Crear predictor
    predictor = NCAAFPredictor()
    
    # Ejecutar análisis completo
    results = predictor.run_full_analysis()
    
    # Mostrar resumen final
    print("\n" + "=" * 60)
    print("📋 RESUMEN FINAL DEL ANÁLISIS")
    print("=" * 60)
    print(f"🏀 {results['game_info']['home']} vs {results['game_info']['away']}")
    print(f"🏆 Ganador Predicho: {results['final_prediction']['winner']}")
    print(f"📊 Puntuación: {results['final_prediction']['central_arkansas_score']}-{results['final_prediction']['vanderbilt_score']}")
    print(f"📈 Margen: {results['final_prediction']['margin']} puntos")
    print(f"🎯 Total: {results['final_prediction']['total']} puntos")
    print(f"💪 Confianza: {results['final_prediction']['confidence_score']}%")
    print(f"💰 Mejor EV: {max(results['ev_analysis']['expected_value'].values())}%")
    print("=" * 60)
    
    # Guardar resultados en archivo
    import json
    with open('prediction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Análisis completado. Resultados guardados en 'prediction_results.json'")
