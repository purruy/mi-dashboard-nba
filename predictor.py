# predictor.py - Sistema Completo de Predicción NCAAB
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class NCAAFPredictor:
    """Sistema completo de predicción para Arkansas vs Texas Tech"""
    
    def __init__(self):
        print("🎯 SISTEMA DE PREDICCIÓN NCAAB INICIADO")
        print("=" * 50)
        print("EQUIPOS: Arkansas Razorbacks vs Texas Tech Red Raiders")
        print("=" * 50)
        
        # Datos específicos del juego
        self.team_a = "Arkansas Razorbacks"
        self.team_b = "Texas Tech Red Raiders"
        
        # Datos ofensivos de la imagen
        self.offensive_stats = {
            'ARK': {
                'points_per_game': 87.6,
                'avg_score_margin': 16.6,
                'assists_per_game': 17.1,
                'total_rebounds': 38.9,
                'effective_fg_pct': 54.2,
                'off_rebound_pct': 29.6,
                'fta_per_fga': 0.381,
                'turnover_pct': 11.2
            },
            'TTU': {
                'points_per_game': 81.4,
                'avg_score_margin': 11.4,
                'assists_per_game': 15.8,
                'total_rebounds': 40.3,
                'effective_fg_pct': 52.6,
                'off_rebound_pct': 35.8,
                'fta_per_fga': 0.334,
                'turnover_pct': 12.9
            }
        }
        
        # Datos defensivos de la imagen
        self.defensive_stats = {
            'ARK': {
                'opp_points_per_game': 71.0,
                'opp_effective_fg_pct': 45.7,
                'off_rebounds_per_game': 9.6,
                'def_rebounds_per_game': 25.3,
                'blocks_per_game': 4.7,
                'steals_per_game': 8.2,
                'personal_fouls_per_game': 16.2
            },
            'TTU': {
                'opp_points_per_game': 70.0,
                'opp_effective_fg_pct': 47.9,
                'off_rebounds_per_game': 12.9,
                'def_rebounds_per_game': 24.0,
                'blocks_per_game': 3.6,
                'steals_per_game': 7.4,
                'personal_fouls_per_game': 16.2
            }
        }
    
    # ==================== MÓDULO 1: ESTIMACIÓN DE CALIDAD ====================
    def quality_estimation(self):
        """Calcula rating de calidad para ambos equipos"""
        print("\n1️⃣ ESTIMACIÓN DE CALIDAD")
        print("-" * 40)
        
        # Fórmula: 40% ofensiva + 40% defensiva + 20% eficiencia
        quality_scores = {}
        
        for team in ['ARK', 'TTU']:
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
            
            print(f"{'Arkansas' if team == 'ARK' else 'Texas Tech'}:")
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
            [87.6, 54.2, 17.1, 29.6, 11.2, 71.0, 4.7, 8.2],  # Arkansas
            [81.4, 52.6, 15.8, 35.8, 12.9, 70.0, 3.6, 7.4]   # Texas Tech
        ])
        
        # Variables objetivo (puntos anotados en juegos similares históricos)
        y = np.array([87.6, 81.4])
        
        # Entrenar modelo
        model = LinearRegression()
        model.fit(X, y)
        
        # Predecir
        pred_ark = model.predict([X[0]])[0]
        pred_ttu = model.predict([X[1]])[0]
        
        # Ajustar por enfrentamiento defensivo
        def_adjustment_ark = (100 - self.defensive_stats['TTU']['opp_effective_fg_pct'] * 2) / 100
        def_adjustment_ttu = (100 - self.defensive_stats['ARK']['opp_effective_fg_pct'] * 2) / 100
        
        final_pred_ark = pred_ark * def_adjustment_ark
        final_pred_ttu = pred_ttu * def_adjustment_ttu
        
        print(f"Arkansas Predicción: {final_pred_ark:.1f} puntos")
        print(f"Texas Tech Predicción: {final_pred_ttu:.1f} puntos")
        print(f"Diferencia: {final_pred_ark - final_pred_ttu:.1f} puntos")
        
        return {
            'ark_points': round(final_pred_ark, 1),
            'ttu_points': round(final_pred_ttu, 1),
            'margin': round(final_pred_ark - final_pred_ttu, 1)
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
        
        # Características para Arkansas y Texas Tech
        X_ark = np.array([[54.2, 11.2, 38.9, 17.1, 8.2, 4.7]])
        X_ttu = np.array([[52.6, 12.9, 40.3, 15.8, 7.4, 3.6]])
        
        # Modelo 1: Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        pred_rf_ark = rf.predict(X_ark)[0]
        pred_rf_ttu = rf.predict(X_ttu)[0]
        
        # Modelo 2: Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        pred_gb_ark = gb.predict(X_ark)[0]
        pred_gb_ttu = gb.predict(X_ttu)[0]
        
        # Ensemble promedio
        ensemble_ark = (pred_rf_ark + pred_gb_ark) / 2
        ensemble_ttu = (pred_rf_ttu + pred_gb_ttu) / 2
        
        # Convertir a puntos
        base_ark = 87.6
        base_ttu = 81.4
        
        final_ark = base_ark + (ensemble_ark - 50) / 10
        final_ttu = base_ttu + (ensemble_ttu - 50) / 10
        
        print(f"Random Forest - ARK: {pred_rf_ark:.1f}, TTU: {pred_rf_ttu:.1f}")
        print(f"Gradient Boost - ARK: {pred_gb_ark:.1f}, TTU: {pred_gb_ttu:.1f}")
        print(f"Ensemble Final - ARK: {final_ark:.1f}, TTU: {final_ttu:.1f}")
        
        return {
            'ark_ensemble': round(final_ark, 1),
            'ttu_ensemble': round(final_ttu, 1),
            'ensemble_margin': round(final_ark - final_ttu, 1),
            'model_agreement': round(100 - abs(pred_rf_ark - pred_gb_ark), 1)
        }
    
    # ==================== MÓDULO 4: FOUR FACTORS ESTIMADOS ====================
    def four_factors_estimation(self):
        """Calcula los Four Factors para el juego"""
        print("\n4️⃣ FOUR FACTORS ESTIMADOS")
        print("-" * 40)
        
        # eFG% estimado (considerando defensa rival)
        efg_ark = (self.offensive_stats['ARK']['effective_fg_pct'] * 0.7 + 
                   (100 - self.defensive_stats['TTU']['opp_effective_fg_pct']) * 0.3)
        
        efg_ttu = (self.offensive_stats['TTU']['effective_fg_pct'] * 0.7 + 
                   (100 - self.defensive_stats['ARK']['opp_effective_fg_pct']) * 0.3)
        
        # TOV% estimado
        tov_ark = (self.offensive_stats['ARK']['turnover_pct'] * 0.6 + 
                   self.defensive_stats['TTU']['steals_per_game'] * 0.4)
        
        tov_ttu = (self.offensive_stats['TTU']['turnover_pct'] * 0.6 + 
                   self.defensive_stats['ARK']['steals_per_game'] * 0.4)
        
        # ORB% estimado
        orb_ark = (self.offensive_stats['ARK']['off_rebound_pct'] * 0.5 + 
                   (100 - self.defensive_stats['TTU']['def_rebounds_per_game'] * 2) * 0.5)
        
        orb_ttu = (self.offensive_stats['TTU']['off_rebound_pct'] * 0.5 + 
                   (100 - self.defensive_stats['ARK']['def_rebounds_per_game'] * 2) * 0.5)
        
        # FTA/FGA estimado
        ftr_ark = self.offensive_stats['ARK']['fta_per_fga'] * 100
        ftr_ttu = self.offensive_stats['TTU']['fta_per_fga'] * 100
        
        four_factors = {
            'ARK': {
                'efg': round(efg_ark, 1),
                'tov': round(tov_ark, 1),
                'orb': round(orb_ark, 1),
                'ftr': round(ftr_ark, 1)
            },
            'TTU': {
                'efg': round(efg_ttu, 1),
                'tov': round(tov_ttu, 1),
                'orb': round(orb_ttu, 1),
                'ftr': round(ftr_ttu, 1)
            }
        }
        
        print("Arkansas Four Factors:")
        print(f"  eFG%: {four_factors['ARK']['efg']}%")
        print(f"  TOV%: {four_factors['ARK']['tov']}%")
        print(f"  ORB%: {four_factors['ARK']['orb']}%")
        print(f"  FTR: {four_factors['ARK']['ftr']}")
        
        print("\nTexas Tech Four Factors:")
        print(f"  eFG%: {four_factors['TTU']['efg']}%")
        print(f"  TOV%: {four_factors['TTU']['tov']}%")
        print(f"  ORB%: {four_factors['TTU']['orb']}%")
        print(f"  FTR: {four_factors['TTU']['ftr']}")
        
        return four_factors
    
    # ==================== MÓDULO 5: EV+ (Expected Value Plus) ====================
    def expected_value_plus(self, final_prediction):
        """Calcula el Expected Value Plus y criterio Kelly"""
        print("\n5️⃣ EV+ (EXPECTED VALUE PLUS) & CRITERIO KELLY")
        print("-" * 40)
        
        # Supongamos odds del mercado (esto vendría de API real)
        market_odds = {
            'moneyline_ark': -150,  # Arkansas -150
            'moneyline_ttu': +130,  # Texas Tech +130
            'spread': -4.5,         # Arkansas -4.5
            'total': 158.5          # Total Over/Under
        }
        
        # Probabilidad de nuestro modelo
        margin = final_prediction['margin']
        total = final_prediction['total']
        
        # Calcular probabilidades
        std_dev = 10.0  # Desviación estándar típica NCAAB
        
        # Probabilidad de Arkansas cubriendo -4.5
        z_score_spread = (margin - (-4.5)) / std_dev
        prob_cover_ark = self._normal_cdf(z_score_spread)
        
        # Probabilidad de Over/Under
        z_score_total = (total - 158.5) / std_dev
        prob_over = 1 - self._normal_cdf(z_score_total)
        
        # Probabilidad de ganador directo
        prob_win_ark = self._normal_cdf(margin / std_dev)
        prob_win_ttu = 1 - prob_win_ark
        
        # Convertir odds a probabilidad implícita
        def implied_probability(american_odds):
            if american_odds > 0:
                return 100 / (american_odds + 100)
            else:
                return abs(american_odds) / (abs(american_odds) + 100)
        
        # Calcular EV para cada mercado
        ev_spread_ark = (prob_cover_ark - implied_probability(-110)) * 100  # Spread usa -110 típico
        ev_moneyline_ark = (prob_win_ark - implied_probability(-150)) * 100
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
        
        kelly_spread = kelly_criterion(prob_cover_ark, -110)
        kelly_moneyline = kelly_criterion(prob_win_ark, -150)
        kelly_total = kelly_criterion(prob_over, -110)
        
        ev_results = {
            'probabilities': {
                'ark_win': round(prob_win_ark * 100, 1),
                'ttu_win': round(prob_win_ttu * 100, 1),
                'ark_cover': round(prob_cover_ark * 100, 1),
                'over': round(prob_over * 100, 1)
            },
            'expected_value': {
                'spread_ev': round(ev_spread_ark, 2),
                'moneyline_ev': round(ev_moneyline_ark, 2),
                'total_ev': round(ev_total_over, 2)
            },
            'kelly_criterion': {
                'spread_kelly': round(kelly_spread * 100, 1),
                'moneyline_kelly': round(kelly_moneyline * 100, 1),
                'total_kelly': round(kelly_total * 100, 1)
            },
            'market_odds': market_odds
        }
        
        print(f"Probabilidad Arkansas gane: {ev_results['probabilities']['ark_win']}%")
        print(f"Probabilidad cubra -4.5: {ev_results['probabilities']['ark_cover']}%")
        print(f"Probabilidad Over 158.5: {ev_results['probabilities']['over']}%")
        print(f"\nExpected Value (EV):")
        print(f"  Spread: {ev_results['expected_value']['spread_ev']}%")
        print(f"  Moneyline: {ev_results['expected_value']['moneyline_ev']}%")
        print(f"  Total: {ev_results['expected_value']['total_ev']}%")
        print(f"\nCriterio Kelly (% bankroll):")
        print(f"  Spread: {ev_results['kelly_criterion']['spread_kelly']}%")
        print(f"  Moneyline: {ev_results['kelly_criterion']['moneyline_kelly']}%")
        print(f"  Total: {ev_results['kelly_criterion']['total_kelly']}%")
        
        return ev_results
    
    # ==================== MÓDULO 6: PREDICCIÓN FINAL ====================
    def final_prediction(self, quality_scores, linear_pred, ensemble_pred, four_factors):
        """Genera la predicción final consolidada"""
        print("\n6️⃣ PREDICCIÓN FINAL CONSOLIDADA")
        print("=" * 50)
        
        # Consolidar predicciones
        ark_points_final = (linear_pred['ark_points'] * 0.4 + 
                           ensemble_pred['ark_ensemble'] * 0.4 + 
                           quality_scores['ARK']['rating'] * 0.2)
        
        ttu_points_final = (linear_pred['ttu_points'] * 0.4 + 
                           ensemble_pred['ttu_ensemble'] * 0.4 + 
                           quality_scores['TTU']['rating'] * 0.2)
        
        # Margen final
        margin_final = ark_points_final - ttu_points_final
        
        # Total final
        total_final = ark_points_final + ttu_points_final
        
        # Grado de confianza (0-100%)
        confidence = min(100, max(0, 
            (quality_scores['ARK']['rating'] - quality_scores['TTU']['rating']) * 0.5 +
            abs(margin_final) * 2 +
            ensemble_pred['model_agreement']
        ))
        
        # Determinar ganador
        if margin_final > 0:
            winner = "Arkansas Razorbacks"
            winner_confidence = (ark_points_final / (ark_points_final + ttu_points_final)) * 100
        else:
            winner = "Texas Tech Red Raiders"
            winner_confidence = (ttu_points_final / (ark_points_final + ttu_points_final)) * 100
        
        final_pred = {
            'arkansas_score': round(ark_points_final, 1),
            'texas_tech_score': round(ttu_points_final, 1),
            'margin': round(margin_final, 1),
            'total': round(total_final, 1),
            'winner': winner,
            'winner_confidence': round(winner_confidence, 1),
            'confidence_score': round(confidence, 1),
            'recommended_bets': self._generate_recommendations(margin_final, total_final, confidence)
        }
        
        # Mostrar resultados
        print(f"🏆 RESULTADO PREDICHO: {winner}")
        print(f"📊 Puntuación Final: {final_pred['arkansas_score']} - {final_pred['texas_tech_score']}")
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
        
        # Spread recomendación
        if margin > 4.5:
            recommendations.append("📈 SPREAD: Arkansas -4.5 (RECOMENDADO)")
        elif margin < -4.5:
            recommendations.append("📈 SPREAD: Texas Tech +4.5 (RECOMENDADO)")
        else:
            recommendations.append("📈 SPREAD: No bet (margen muy cercano)")
        
        # Total recomendación
        if total > 158.5:
            recommendations.append("🎯 TOTAL: Over 158.5 (RECOMENDADO)")
        else:
            recommendations.append("🎯 TOTAL: Under 158.5 (RECOMENDADO)")
        
        # Moneyline recomendación
        if margin > 0 and confidence > 60:
            recommendations.append("💰 MONEYLINE: Arkansas (VALOR POSITIVO)")
        elif margin < 0 and confidence > 60:
            recommendations.append("💰 MONEYLINE: Texas Tech (VALOR POSITIVO)")
        else:
            recommendations.append("💰 MONEYLINE: Evitar (poco valor)")
        
        # Kelly sizing
        if confidence > 70:
            recommendations.append("⚡ TAMAÑO: 3-5% bankroll (ALTA CONFIANZA)")
        elif confidence > 50:
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
    print(f"📊 Puntuación: {results['final_prediction']['arkansas_score']}-{results['final_prediction']['texas_tech_score']}")
    print(f"📈 Margen: {results['final_prediction']['margin']} puntos")
    print(f"🎯 Total: {results['final_prediction']['total']} puntos")
    print(f"💪 Confianza: {results['final_prediction']['confidence_score']}%")
    print(f"💰 Mejor EV: {results['ev_analysis']['expected_value']['spread_ev']}% (Spread)")
    print("=" * 60)
    
    # Guardar resultados en archivo
    import json
    with open('prediction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Análisis completado. Resultados guardados en 'prediction_results.json'")
