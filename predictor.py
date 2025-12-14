# predictor.py - Sistema Completo de Predicción NCAAB
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class NCAAFPredictor:
    """Sistema completo de predicción para Alabama vs Arizona"""
    
    def __init__(self):
        print("🎯 SISTEMA DE PREDICCIÓN NCAAB INICIADO")
        print("=" * 50)
        print("EQUIPOS: Alabama Crimson Tide vs Arizona Wildcats")
        print("=" * 50)
        
        # Datos específicos del juego
        self.team_a = "Alabama Crimson Tide"
        self.team_b = "Arizona Wildcats"
        
        # Datos ofensivos actualizados
        self.offensive_stats = {
            'ALA': {
                'points_per_game': 95.1,
                'avg_score_margin': 15.9,
                'assists_per_game': 18.2,
                'total_rebounds': 42.7,
                'effective_fg_pct': 57.2,
                'off_rebound_pct': 31.0,
                'fta_per_fga': 0.342,
                'turnover_pct': 11.1
            },
            'ARIZ': {
                'points_per_game': 88.5,
                'avg_score_margin': 21.4,
                'assists_per_game': 19.5,
                'total_rebounds': 42.3,
                'effective_fg_pct': 58.0,
                'off_rebound_pct': 42.3,
                'fta_per_fga': 0.404,
                'turnover_pct': 14.9
            }
        }
        
        # Datos defensivos actualizados
        self.defensive_stats = {
            'ALA': {
                'opp_points_per_game': 79.2,
                'opp_effective_fg_pct': 47.6,
                'off_rebounds_per_game': 10.4,
                'def_rebounds_per_game': 28.7,
                'blocks_per_game': 6.2,
                'steals_per_game': 7.8,
                'personal_fouls_per_game': 18.6
            },
            'ARIZ': {
                'opp_points_per_game': 67.1,
                'opp_effective_fg_pct': 45.1,
                'off_rebounds_per_game': 12.4,
                'def_rebounds_per_game': 26.8,
                'blocks_per_game': 4.4,
                'steals_per_game': 8.4,
                'personal_fouls_per_game': 17.0
            }
        }
    
    # ==================== MÓDULO 1: ESTIMACIÓN DE CALIDAD ====================
    def quality_estimation(self):
        """Calcula rating de calidad para ambos equipos"""
        print("\n1️⃣ ESTIMACIÓN DE CALIDAD")
        print("-" * 40)
        
        # Fórmula: 40% ofensiva + 40% defensiva + 20% eficiencia
        quality_scores = {}
        
        for team in ['ALA', 'ARIZ']:
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
            
            print(f"{'Alabama' if team == 'ALA' else 'Arizona'}:")
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
        
        # Variables predictoras con nuevos datos
        X = np.array([
            # [PTS/G, eFG%, Asist, RebOf%, TOV%, OppPTS, Blk, Rob]
            [95.1, 57.2, 18.2, 31.0, 11.1, 79.2, 6.2, 7.8],   # Alabama
            [88.5, 58.0, 19.5, 42.3, 14.9, 67.1, 4.4, 8.4]    # Arizona
        ])
        
        # Variables objetivo
        y = np.array([95.1, 88.5])
        
        # Entrenar modelo
        model = LinearRegression()
        model.fit(X, y)
        
        # Predecir
        pred_ala = model.predict([X[0]])[0]
        pred_ariz = model.predict([X[1]])[0]
        
        # Ajustar por enfrentamiento defensivo
        def_adjustment_ala = (100 - self.defensive_stats['ARIZ']['opp_effective_fg_pct'] * 2) / 100
        def_adjustment_ariz = (100 - self.defensive_stats['ALA']['opp_effective_fg_pct'] * 2) / 100
        
        final_pred_ala = pred_ala * def_adjustment_ala
        final_pred_ariz = pred_ariz * def_adjustment_ariz
        
        print(f"Alabama Predicción: {final_pred_ala:.1f} puntos")
        print(f"Arizona Predicción: {final_pred_ariz:.1f} puntos")
        print(f"Diferencia: {final_pred_ala - final_pred_ariz:.1f} puntos")
        
        return {
            'ala_points': round(final_pred_ala, 1),
            'ariz_points': round(final_pred_ariz, 1),
            'margin': round(final_pred_ala - final_pred_ariz, 1)
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
        
        # Características para Alabama y Arizona
        X_ala = np.array([[57.2, 11.1, 42.7, 18.2, 7.8, 6.2]])
        X_ariz = np.array([[58.0, 14.9, 42.3, 19.5, 8.4, 4.4]])
        
        # Modelo 1: Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        pred_rf_ala = rf.predict(X_ala)[0]
        pred_rf_ariz = rf.predict(X_ariz)[0]
        
        # Modelo 2: Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        pred_gb_ala = gb.predict(X_ala)[0]
        pred_gb_ariz = gb.predict(X_ariz)[0]
        
        # Ensemble promedio
        ensemble_ala = (pred_rf_ala + pred_gb_ala) / 2
        ensemble_ariz = (pred_rf_ariz + pred_gb_ariz) / 2
        
        # Convertir a puntos
        base_ala = 95.1
        base_ariz = 88.5
        
        final_ala = base_ala + (ensemble_ala - 50) / 10
        final_ariz = base_ariz + (ensemble_ariz - 50) / 10
        
        print(f"Random Forest - ALA: {pred_rf_ala:.1f}, ARIZ: {pred_rf_ariz:.1f}")
        print(f"Gradient Boost - ALA: {pred_gb_ala:.1f}, ARIZ: {pred_gb_ariz:.1f}")
        print(f"Ensemble Final - ALA: {final_ala:.1f}, ARIZ: {final_ariz:.1f}")
        
        return {
            'ala_ensemble': round(final_ala, 1),
            'ariz_ensemble': round(final_ariz, 1),
            'ensemble_margin': round(final_ala - final_ariz, 1),
            'model_agreement': round(100 - abs(pred_rf_ala - pred_gb_ala), 1)
        }
    
    # ==================== MÓDULO 4: FOUR FACTORS ESTIMADOS ====================
    def four_factors_estimation(self):
        """Calcula los Four Factors para el juego"""
        print("\n4️⃣ FOUR FACTORS ESTIMADOS")
        print("-" * 40)
        
        # eFG% estimado (considerando defensa rival)
        efg_ala = (self.offensive_stats['ALA']['effective_fg_pct'] * 0.7 + 
                   (100 - self.defensive_stats['ARIZ']['opp_effective_fg_pct']) * 0.3)
        
        efg_ariz = (self.offensive_stats['ARIZ']['effective_fg_pct'] * 0.7 + 
                    (100 - self.defensive_stats['ALA']['opp_effective_fg_pct']) * 0.3)
        
        # TOV% estimado
        tov_ala = (self.offensive_stats['ALA']['turnover_pct'] * 0.6 + 
                   self.defensive_stats['ARIZ']['steals_per_game'] * 0.4)
        
        tov_ariz = (self.offensive_stats['ARIZ']['turnover_pct'] * 0.6 + 
                    self.defensive_stats['ALA']['steals_per_game'] * 0.4)
        
        # ORB% estimado
        orb_ala = (self.offensive_stats['ALA']['off_rebound_pct'] * 0.5 + 
                   (100 - self.defensive_stats['ARIZ']['def_rebounds_per_game'] * 2) * 0.5)
        
        orb_ariz = (self.offensive_stats['ARIZ']['off_rebound_pct'] * 0.5 + 
                    (100 - self.defensive_stats['ALA']['def_rebounds_per_game'] * 2) * 0.5)
        
        # FTA/FGA estimado
        ftr_ala = self.offensive_stats['ALA']['fta_per_fga'] * 100
        ftr_ariz = self.offensive_stats['ARIZ']['fta_per_fga'] * 100
        
        four_factors = {
            'ALA': {
                'efg': round(efg_ala, 1),
                'tov': round(tov_ala, 1),
                'orb': round(orb_ala, 1),
                'ftr': round(ftr_ala, 1)
            },
            'ARIZ': {
                'efg': round(efg_ariz, 1),
                'tov': round(tov_ariz, 1),
                'orb': round(orb_ariz, 1),
                'ftr': round(ftr_ariz, 1)
            }
        }
        
        print("Alabama Four Factors:")
        print(f"  eFG%: {four_factors['ALA']['efg']}%")
        print(f"  TOV%: {four_factors['ALA']['tov']}%")
        print(f"  ORB%: {four_factors['ALA']['orb']}%")
        print(f"  FTR: {four_factors['ALA']['ftr']}")
        
        print("\nArizona Four Factors:")
        print(f"  eFG%: {four_factors['ARIZ']['efg']}%")
        print(f"  TOV%: {four_factors['ARIZ']['tov']}%")
        print(f"  ORB%: {four_factors['ARIZ']['orb']}%")
        print(f"  FTR: {four_factors['ARIZ']['ftr']}")
        
        return four_factors
    
    # ==================== MÓDULO 5: EV+ (Expected Value Plus) ====================
    def expected_value_plus(self, final_prediction):
        """Calcula el Expected Value Plus y criterio Kelly"""
        print("\n5️⃣ EV+ (EXPECTED VALUE PLUS) & CRITERIO KELLY")
        print("-" * 40)
        
        # Odds del mercado para Alabama vs Arizona
        market_odds = {
            'moneyline_ala': -120,    # Alabama -120
            'moneyline_ariz': +100,   # Arizona +100
            'spread': -2.5,           # Alabama -2.5
            'total': 175.5            # Total Over/Under
        }
        
        # Probabilidad de nuestro modelo
        margin = final_prediction['margin']
        total = final_prediction['total']
        
        # Calcular probabilidades
        std_dev = 8.5  # Desviación estándar ajustada para equipos de élite
        
        # Probabilidad de Alabama cubriendo -2.5
        z_score_spread = (margin - (-2.5)) / std_dev
        prob_cover_ala = self._normal_cdf(z_score_spread)
        
        # Probabilidad de Over/Under
        z_score_total = (total - 175.5) / std_dev
        prob_over = 1 - self._normal_cdf(z_score_total)
        
        # Probabilidad de ganador directo
        prob_win_ala = self._normal_cdf(margin / std_dev)
        prob_win_ariz = 1 - prob_win_ala
        
        # Convertir odds a probabilidad implícita
        def implied_probability(american_odds):
            if american_odds > 0:
                return 100 / (american_odds + 100)
            else:
                return abs(american_odds) / (abs(american_odds) + 100)
        
        # Calcular EV para cada mercado
        ev_spread_ala = (prob_cover_ala - implied_probability(-110)) * 100
        ev_moneyline_ala = (prob_win_ala - implied_probability(-120)) * 100
        ev_moneyline_ariz = (prob_win_ariz - implied_probability(+100)) * 100
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
        
        kelly_spread = kelly_criterion(prob_cover_ala, -110)
        kelly_moneyline_ala = kelly_criterion(prob_win_ala, -120)
        kelly_moneyline_ariz = kelly_criterion(prob_win_ariz, +100)
        kelly_total = kelly_criterion(prob_over, -110)
        
        ev_results = {
            'probabilities': {
                'ala_win': round(prob_win_ala * 100, 1),
                'ariz_win': round(prob_win_ariz * 100, 1),
                'ala_cover': round(prob_cover_ala * 100, 1),
                'over': round(prob_over * 100, 1)
            },
            'expected_value': {
                'spread_ev': round(ev_spread_ala, 2),
                'moneyline_ala_ev': round(ev_moneyline_ala, 2),
                'moneyline_ariz_ev': round(ev_moneyline_ariz, 2),
                'total_ev': round(ev_total_over, 2)
            },
            'kelly_criterion': {
                'spread_kelly': round(kelly_spread * 100, 1),
                'moneyline_ala_kelly': round(kelly_moneyline_ala * 100, 1),
                'moneyline_ariz_kelly': round(kelly_moneyline_ariz * 100, 1),
                'total_kelly': round(kelly_total * 100, 1)
            },
            'market_odds': market_odds
        }
        
        print(f"Probabilidad Alabama gane: {ev_results['probabilities']['ala_win']}%")
        print(f"Probabilidad cubra -2.5: {ev_results['probabilities']['ala_cover']}%")
        print(f"Probabilidad Over 175.5: {ev_results['probabilities']['over']}%")
        print(f"\nExpected Value (EV):")
        print(f"  Spread: {ev_results['expected_value']['spread_ev']}%")
        print(f"  Moneyline ALA: {ev_results['expected_value']['moneyline_ala_ev']}%")
        print(f"  Moneyline ARIZ: {ev_results['expected_value']['moneyline_ariz_ev']}%")
        print(f"  Total: {ev_results['expected_value']['total_ev']}%")
        print(f"\nCriterio Kelly (% bankroll):")
        print(f"  Spread: {ev_results['kelly_criterion']['spread_kelly']}%")
        print(f"  Moneyline ALA: {ev_results['kelly_criterion']['moneyline_ala_kelly']}%")
        print(f"  Moneyline ARIZ: {ev_results['kelly_criterion']['moneyline_ariz_kelly']}%")
        print(f"  Total: {ev_results['kelly_criterion']['total_kelly']}%")
        
        return ev_results
    
    # ==================== MÓDULO 6: PREDICCIÓN FINAL ====================
    def final_prediction(self, quality_scores, linear_pred, ensemble_pred, four_factors):
        """Genera la predicción final consolidada"""
        print("\n6️⃣ PREDICCIÓN FINAL CONSOLIDADA")
        print("=" * 50)
        
        # Consolidar predicciones
        ala_points_final = (linear_pred['ala_points'] * 0.4 + 
                           ensemble_pred['ala_ensemble'] * 0.4 + 
                           quality_scores['ALA']['rating'] * 0.2)
        
        ariz_points_final = (linear_pred['ariz_points'] * 0.4 + 
                            ensemble_pred['ariz_ensemble'] * 0.4 + 
                            quality_scores['ARIZ']['rating'] * 0.2)
        
        # Margen final
        margin_final = ala_points_final - ariz_points_final
        
        # Total final
        total_final = ala_points_final + ariz_points_final
        
        # Grado de confianza (0-100%)
        confidence = min(100, max(0, 
            (quality_scores['ALA']['rating'] - quality_scores['ARIZ']['rating']) * 0.5 +
            abs(margin_final) * 2.5 +
            ensemble_pred['model_agreement']
        ))
        
        # Determinar ganador
        if margin_final > 0:
            winner = "Alabama Crimson Tide"
            winner_confidence = (ala_points_final / (ala_points_final + ariz_points_final)) * 100
        else:
            winner = "Arizona Wildcats"
            winner_confidence = (ariz_points_final / (ala_points_final + ariz_points_final)) * 100
        
        final_pred = {
            'alabama_score': round(ala_points_final, 1),
            'arizona_score': round(ariz_points_final, 1),
            'margin': round(margin_final, 1),
            'total': round(total_final, 1),
            'winner': winner,
            'winner_confidence': round(winner_confidence, 1),
            'confidence_score': round(confidence, 1),
            'recommended_bets': self._generate_recommendations(margin_final, total_final, confidence)
        }
        
        # Mostrar resultados
        print(f"🏆 RESULTADO PREDICHO: {winner}")
        print(f"📊 Puntuación Final: {final_pred['alabama_score']} - {final_pred['arizona_score']}")
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
        
        # Spread recomendación (Alabama -2.5)
        if margin > 2.5 and confidence > 60:
            recommendations.append("📈 SPREAD: Alabama -2.5 (RECOMENDADO)")
        elif margin < -2.5 and confidence > 60:
            recommendations.append("📈 SPREAD: Arizona +2.5 (RECOMENDADO)")
        else:
            recommendations.append("📈 SPREAD: No bet (margen muy cercano)")
        
        # Total recomendación
        if total > 175.5 and confidence > 55:
            recommendations.append("🎯 TOTAL: Over 175.5 (RECOMENDADO)")
        elif total < 175.5 and confidence > 55:
            recommendations.append("🎯 TOTAL: Under 175.5 (RECOMENDADO)")
        else:
            recommendations.append("🎯 TOTAL: Cautela (total muy ajustado)")
        
        # Moneyline recomendación
        if margin > 0 and confidence > 65:
            recommendations.append("💰 MONEYLINE: Alabama (VALOR MODERADO)")
        elif margin < 0 and confidence > 65:
            recommendations.append("💰 MONEYLINE: Arizona (VALOR POSITIVO)")
        else:
            recommendations.append("💰 MONEYLINE: Evitar (poco valor)")
        
        # Kelly sizing basado en confianza
        if confidence > 75:
            recommendations.append("⚡ TAMAÑO: 3-4% bankroll (ALTA CONFIANZA)")
        elif confidence > 60:
            recommendations.append("⚡ TAMAÑO: 1-2% bankroll (MEDIA CONFIANZA)")
        else:
            recommendations.append("⚡ TAMAÑO: 0.5% bankroll (BAJA CONFIANZA)")
        
        # Hot take basado en datos
        if margin > 5:
            recommendations.append("🔥 HOT TAKE: Alabama gana por más de 10")
        elif margin < -5:
            recommendations.append("🔥 HOT TAKE: Arizona sorprende por +7")
        
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
                'date': '2024-03-25'
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
    print(f"📊 Puntuación: {results['final_prediction']['alabama_score']}-{results['final_prediction']['arizona_score']}")
    print(f"📈 Margen: {results['final_prediction']['margin']} puntos")
    print(f"🎯 Total: {results['final_prediction']['total']} puntos")
    print(f"💪 Confianza: {results['final_prediction']['confidence_score']}%")
    
    # Encontrar mejor EV
    ev_values = results['ev_analysis']['expected_value']
    best_ev_market = max(ev_values, key=ev_values.get)
    best_ev_value = ev_values[best_ev_market]
    
    print(f"💰 Mejor EV: {best_ev_value}% ({best_ev_market})")
    print("=" * 60)
    
    # Guardar resultados en archivo
    import json
    with open('prediction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Análisis completado. Resultados guardados en 'prediction_results.json'")
