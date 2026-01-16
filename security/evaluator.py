import numpy as np
import matplotlib.pyplot as plt

class BiometricEvaluator:
    def __init__(self):
        self.results = []

    def add_test_case(self, score, is_real, liveness_passed):
        """
        Record a test result.
        is_real: True if the identity actually matches.
        liveness_passed: True if rPPG detected a pulse.
        """
        self.results.append({
            'score': score,
            'is_real': is_real,
            'liveness': liveness_passed
        })

    def calculate_metrics(self, threshold=0.65):
        apcer_count = 0  # False Matches (Spoofs accepted)
        bpcer_count = 0  # False Rejections (Real people blocked)
        total_spoofs = 0
        total_real = 0

        for res in self.results:
            verified = (res['score'] >= threshold) and res['liveness']
            
            if not res['is_real']:
                total_spoofs += 1
                if verified: apcer_count += 1
            else:
                total_real += 1
                if not verified: bpcer_count += 1

        apcer = apcer_count / total_spoofs if total_spoofs > 0 else 0
        bpcer = bpcer_count / total_real if total_real > 0 else 0
        acer = (apcer + bpcer) / 2

        print(f"\n--- Performance Report (Threshold: {threshold}) ---")
        print(f"APCER (Attack Rate): {apcer:.2%}")
        print(f"BPCER (Frustration Rate): {bpcer:.2%}")
        print(f"ACER (Total Error): {acer:.2%}")
        
        return acer

    def plot_distribution(self):
        real_scores = [r['score'] for r in self.results if r['is_real']]
        spoof_scores = [r['score'] for r in self.results if not r['is_real']]
        
        plt.hist(real_scores, alpha=0.5, label='Real Matches', color='green')
        plt.hist(spoof_scores, alpha=0.5, label='Spoof/Attack', color='red')
        plt.axvline(x=0.65, color='blue', linestyle='--', label='Threshold')
        plt.title("Identity Similarity Distribution")
        plt.legend()
        plt.show()