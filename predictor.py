import json
from datetime import datetime

class BasketballGameAnalyzer:
    def __init__(self):
        self.data = {
            "teams": {
                "EIU": {
                    "name": "EIU Panthers",
                    "record": "2-7",
                    "offensive_stats": {
                        "points_per_game": 61.8,
                        "avg_score_margin": -18.3,
                        "assists_per_game": 11.1,
                        "total_rebounds_per_game": 28.6,
                        "effective_fg_percent": 44.3,
                        "off_rebound_percent": 21.1,
                        "fta_per_fga": 0.334,
                        "turnover_percent": 17.8
                    },
                    "defensive_stats": {
                        "opp_points_per_game": 80.0,
                        "opp_effective_fg_percent": 55.0,
                        "off_rebounds_per_game": 6.4,
                        "def_rebounds_per_game": 18.9,
                        "blocks_per_game": 2.8,
                        "steals_per_game": 5.5,
                        "personal_fouls_per_game": 17.0
                    }
                },
                "ISU": {
                    "name": "Iowa State Cyclones",
                    "record": "10-0",
                    "offensive_stats": {
                        "points_per_game": 91.7,
                        "avg_score_margin": 27.4,
                        "assists_per_game": 18.3,
                        "total_rebounds_per_game": 35.1,
                        "effective_fg_percent": 61.5,
                        "off_rebound_percent": 33.5,
                        "fta_per_fga": 0.328,
                        "turnover_percent": 12.1
                    },
                    "defensive_stats": {
                        "opp_points_per_game": 64.3,
                        "opp_effective_fg_percent": 49.5,
                        "off_rebounds_per_game": 9.4,
                        "def_rebounds_per_game": 21.9,
                        "blocks_per_game": 3.4,
                        "steals_per_game": 11.1,
                        "personal_fouls_per_game": 16.1
                    }
                }
            },
            "betting_lines": {
                "spread": {
                    "EIU": "+40.5",
                    "ISU": "-40.5"
                },
                "moneyline": {
                    "EIU": "+43.5 (-110)",
                    "ISU": "-43.5 (-110)"
                },
                "total_points": {
                    "over": "144.5 (-110)",
                    "under": "144.5 (-110)"
                },
                "alternate_total": {
                    "under": "141.5"
                }
            }
        }
    
    def calculate_stat_differences(self):
        """Calculate differences between team statistics"""
        differences = {}
        
        # Offensive stat differences
        differences["offensive"] = {}
        for stat in self.data["teams"]["EIU"]["offensive_stats"]:
            eiu_val = self.data["teams"]["EIU"]["offensive_stats"][stat]
            isu_val = self.data["teams"]["ISU"]["offensive_stats"][stat]
            diff = isu_val - eiu_val
            differences["offensive"][stat] = round(diff, 2)
        
        # Defensive stat differences
        differences["defensive"] = {}
        for stat in self.data["teams"]["EIU"]["defensive_stats"]:
            eiu_val = self.data["teams"]["EIU"]["defensive_stats"][stat]
            isu_val = self.data["teams"]["ISU"]["defensive_stats"][stat]
            
            # For defensive stats, lower is better for some metrics
            if "opp_" in stat or "fouls" in stat:
                diff = eiu_val - isu_val  # Negative diff means ISU is better
            else:
                diff = isu_val - eiu_val  # Positive diff means ISU is better
            
            differences["defensive"][stat] = round(diff, 2)
        
        return differences
    
    def generate_analysis(self):
        """Generate a textual analysis based on the data"""
        analysis = []
        
        # Overall comparison
        analysis.append("=== GAME ANALYSIS ===")
        analysis.append(f"Matchup: {self.data['teams']['EIU']['name']} ({self.data['teams']['EIU']['record']}) "
                       f"vs {self.data['teams']['ISU']['name']} ({self.data['teams']['ISU']['record']})")
        
        # Score margin analysis
        score_margin_diff = self.data["teams"]["ISU"]["offensive_stats"]["avg_score_margin"] - self.data["teams"]["EIU"]["offensive_stats"]["avg_score_margin"]
        analysis.append(f"\nAverage Score Margin Difference: +{score_margin_diff:.1f} in favor of Iowa State")
        
        # Offensive analysis
        analysis.append("\n--- OFFENSIVE ADVANTAGES ---")
        ppg_diff = self.data["teams"]["ISU"]["offensive_stats"]["points_per_game"] - self.data["teams"]["EIU"]["offensive_stats"]["points_per_game"]
        analysis.append(f"• Iowa State scores {ppg_diff:.1f} more points per game")
        
        efg_diff = self.data["teams"]["ISU"]["offensive_stats"]["effective_fg_percent"] - self.data["teams"]["EIU"]["offensive_stats"]["effective_fg_percent"]
        analysis.append(f"• Iowa State has {efg_diff:.1f}% better shooting efficiency")
        
        # Defensive analysis
        analysis.append("\n--- DEFENSIVE ADVANTAGES ---")
        opp_ppg_diff = self.data["teams"]["EIU"]["defensive_stats"]["opp_points_per_game"] - self.data["teams"]["ISU"]["defensive_stats"]["opp_points_per_game"]
        analysis.append(f"• Iowa State allows {opp_ppg_diff:.1f} fewer points per game")
        
        steals_diff = self.data["teams"]["ISU"]["defensive_stats"]["steals_per_game"] - self.data["teams"]["EIU"]["defensive_stats"]["steals_per_game"]
        analysis.append(f"• Iowa State averages {steals_diff:.1f} more steals per game")
        
        # Betting analysis
        analysis.append("\n--- BETTING INSIGHTS ---")
        analysis.append(f"• Spread: Iowa State favored by {self.data['betting_lines']['spread']['ISU']}")
        analysis.append(f"• Total Points: {self.data['betting_lines']['total_points']['over']}")
        analysis.append(f"• Moneyline: Iowa State {self.data['betting_lines']['moneyline']['ISU']}")
        
        # Prediction
        analysis.append("\n--- PREDICTION ---")
        analysis.append("Based on the statistics, Iowa State has significant advantages in:")
        analysis.append("1. Scoring offense (+29.9 PPG)")
        analysis.append("2. Defensive efficiency (-15.7 PPG allowed)")
        analysis.append("3. Turnover creation (+5.6 steals/game)")
        analysis.append("4. Rebounding (+6.5 total rebounds/game)")
        
        return "\n".join(analysis)
    
    def export_to_json(self, filename="game_data.json"):
        """Export all data to a JSON file"""
        export_data = {
            "metadata": {
                "game": "EIU vs Iowa State",
                "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "NCAA Statistics & Sportsbook Lines"
            },
            "teams": self.data["teams"],
            "betting_lines": self.data["betting_lines"],
            "stat_differences": self.calculate_stat_differences(),
            "analysis": self.generate_analysis()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Data exported to {filename}")
        return export_data
    
    def display_summary(self):
        """Display a summary in the console"""
        print("=" * 60)
        print("BASKETBALL GAME ANALYSIS: EIU vs IOWA STATE")
        print("=" * 60)
        
        print(f"\nTEAM RECORDS:")
        print(f"• {self.data['teams']['EIU']['name']}: {self.data['teams']['EIU']['record']}")
        print(f"• {self.data['teams']['ISU']['name']}: {self.data['teams']['ISU']['record']}")
        
        print(f"\nBETTING LINES:")
        print(f"• Spread: EIU {self.data['betting_lines']['spread']['EIU']} | ISU {self.data['betting_lines']['spread']['ISU']}")
        print(f"• Moneyline: EIU {self.data['betting_lines']['moneyline']['EIU']} | ISU {self.data['betting_lines']['moneyline']['ISU']}")
        print(f"• Total: Over {self.data['betting_lines']['total_points']['over']} | Under {self.data['betting_lines']['total_points']['under']}")
        
        print(f"\nKEY STATISTICAL DIFFERENCES:")
        diffs = self.calculate_stat_differences()
        
        print("Offensive:")
        print(f"  • Points/Game: +{diffs['offensive']['points_per_game']:.1f} ISU")
        print(f"  • Effective FG%: +{diffs['offensive']['effective_fg_percent']:.1f}% ISU")
        
        print("Defensive:")
        print(f"  • Opp Points/Game: {diffs['defensive']['opp_points_per_game']:+.1f} (negative favors ISU)")
        print(f"  • Steals/Game: +{diffs['defensive']['steals_per_game']:.1f} ISU")
        
        print("\n" + "=" * 60)

def main():
    """Main function to run the analyzer"""
    analyzer = BasketballGameAnalyzer()
    
    # Display summary in console
    analyzer.display_summary()
    
    # Export to JSON
    data = analyzer.export_to_json()
    
    # Generate and print analysis
    print("\n" + analyzer.generate_analysis())
    
    return data

if __name__ == "__main__":
    main()
