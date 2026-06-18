"""
Sustainable Search Reranking via Contextual Bandits
====================================================
FIXED VERSION 5 — Slower decay, no penalty for no-click, more candidates.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional


# ============================================================================
# PART 1: DATA GENERATION
# ============================================================================

class SearchDataGenerator:
    """Generates searches with clear green vs dirty signal."""
    
    GREEN_DOMAINS = [
        ("ecosia.org", 0.02, "green"),
        ("allbirds.com", 0.05, "green"),
        ("patagonia.com", 0.08, "green"),
        ("ethicalconsumer.org", 0.10, "green"),
        ("treehugger.com", 0.12, "green"),
        ("goodonyou.eco", 0.15, "green"),
    ]
    
    DIRTY_DOMAINS = [
        ("amazon.com", 0.85, "dirty"),
        ("walmart.com", 0.82, "dirty"),
        ("temu.com", 0.95, "dirty"),
        ("foxnews.com", 0.90, "dirty"),
        ("nike.com", 0.78, "dirty"),
        ("exxon.com", 0.98, "dirty"),
    ]
    
    MEDIUM_DOMAINS = [
        ("wikipedia.org", 0.35, "medium"),
        ("medium.com", 0.40, "medium"),
        ("nytimes.com", 0.50, "medium"),
        ("reddit.com", 0.55, "medium"),
        ("github.com", 0.45, "medium"),
    ]
    
    ALL_DOMAINS = GREEN_DOMAINS + MEDIUM_DOMAINS + DIRTY_DOMAINS
    
    QUERIES = [
        "sustainable running shoes",
        "best laptop 2024",
        "climate change facts",
        "how to reduce waste",
        "electric car reviews",
        "cheap flights",
        "eco friendly products",
        "news today",
        "best phone",
        "renewable energy",
    ]
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
    def generate_search(self) -> Dict:
        query = self.QUERIES[self.rng.randint(0, len(self.QUERIES))]
        indices = self.rng.choice(len(self.ALL_DOMAINS), size=10, replace=False)
        
        results = []
        for pos, idx in enumerate(indices):
            domain, base_carbon, domain_type = self.ALL_DOMAINS[idx]
            carbon = np.clip(base_carbon + self.rng.normal(0, 0.03), 0, 1)
            relevance = np.clip(
                1.0 - (pos * 0.08) + self.rng.normal(0, 0.12),
                0.1, 1.0
            )
            results.append({
                'domain': domain,
                'position': pos,
                'carbon_score': round(carbon, 3),
                'true_relevance': round(relevance, 3),
                'domain_type': domain_type,
            })
        
        results.sort(key=lambda x: x['true_relevance'], reverse=True)
        for i, r in enumerate(results):
            r['position'] = i
            
        return {'query': query, 'results': results}


# ============================================================================
# PART 2: FEATURES
# ============================================================================

class FeatureExtractor:
    @staticmethod
    def extract(result: Dict) -> np.ndarray:
        pos = result['position']
        carbon = result['carbon_score']
        return np.array([
            carbon,                # 0: carbon score
            1.0 - carbon,          # 1: green score
            pos / 10.0,            # 2: position
            1.0 - (pos / 10.0),    # 3: relevance proxy
            1.0,                   # 4: bias
        ])


# ============================================================================
# PART 3: BANDIT
# ============================================================================

class ContextualBandit:
    """LinUCB with epsilon-greedy exploration."""
    
    def __init__(
        self,
        n_actions: int = 3,
        n_features: int = 5,
        alpha: float = 2.0,
        lambda_value: float = 0.3,
        epsilon: float = 0.30,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha
        self.lambda_value = lambda_value
        self.epsilon = epsilon
        
        self.A = {a: np.identity(n_features) for a in range(n_actions)}
        self.b = {a: np.zeros(n_features) for a in range(n_actions)}
        self.theta = {a: np.zeros(n_features) for a in range(n_actions)}
        self.update_counts = {a: 0 for a in range(n_actions)}
        self.action_rewards = {a: [] for a in range(n_actions)}
        self.total_predicts = 0
        self.random_choices = 0
        
    def predict(self, context: np.ndarray) -> Tuple[int, np.ndarray]:
        self.total_predicts += 1
        context = context.flatten()
        
        if np.random.random() < self.epsilon:
            self.random_choices += 1
            action = np.random.randint(0, self.n_actions)
            scores = np.zeros(self.n_actions)
            scores[action] = 1.0
            return action, scores
        
        scores = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            expected = np.dot(self.theta[a], context)
            uncertainty = self.alpha * np.sqrt(np.dot(context, np.dot(A_inv, context)))
            scores[a] = expected + uncertainty
            
        return int(np.argmax(scores)), scores
    
    def update(self, action: int, context: np.ndarray, reward: float):
        """Update with pre-computed reward (can be fractional/counterfactual)."""
        context = context.flatten()
        
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context
        
        try:
            self.theta[action] = np.linalg.solve(self.A[action], self.b[action])
        except np.linalg.LinAlgError:
            self.theta[action] = np.linalg.lstsq(self.A[action], self.b[action], rcond=None)[0]
            
        self.update_counts[action] += 1
        self.action_rewards[action].append(reward)
    
    def decay_epsilon(self, decay: float = 0.9995):
        """Slower decay — was 0.999, now 0.9995."""
        self.epsilon = max(0.08, self.epsilon * decay)


# ============================================================================
# PART 4: GUARDRAIL
# ============================================================================

class CTRGuardrail:
    def __init__(self, min_ctr_ratio: float = 0.85, window: int = 100):
        self.min_ctr_ratio = min_ctr_ratio
        self.window = window
        self.recent: List[bool] = []
        self.baseline_ctr: Optional[float] = None
        
    def record(self, any_click: bool):
        self.recent.append(any_click)
        if len(self.recent) > self.window:
            self.recent.pop(0)
            
    def current_ctr(self) -> float:
        if not self.recent:
            return 0.0
        return sum(self.recent) / len(self.recent)
    
    def set_baseline(self):
        self.baseline_ctr = self.current_ctr()
        
    def is_allowed(self, boost: int) -> bool:
        if self.baseline_ctr is None or len(self.recent) < 50:
            return True
        current = self.current_ctr()
        threshold = self.baseline_ctr * self.min_ctr_ratio
        if current < threshold:
            return boost <= 1
        return True


# ============================================================================
# PART 5: USER SIMULATOR
# ============================================================================

class UserSimulator:
    def __init__(self, seed: int = 123):
        self.rng = np.random.RandomState(seed)
        
    def click(self, results: List[Dict]) -> Optional[int]:
        if self.rng.random() > 0.22:
            return None
        
        weights = []
        for result in results:
            pos = result.get('display_position', result['position'])
            relevance = result['true_relevance']
            pos_weight = math.exp(-0.6 * pos)
            weight = relevance * pos_weight
            weights.append(weight)
        
        total = sum(weights)
        if total == 0:
            return None
            
        probs = [w / total for w in weights]
        chosen_idx = self.rng.choice(len(results), p=probs)
        return results[chosen_idx].get('display_position', results[chosen_idx]['position'])


# ============================================================================
# PART 6: RERANKING WITH COUNTERFACTUAL REWARDS
# ============================================================================

class RerankingPipeline:
    """
    Counterfactual reward propagation.
    
    KEY FIXES v5:
    - 5 boost candidates instead of 3 (more signal)
    - No penalty for no-click (was teaching bandit that boosting = bad)
    - Distance-weighted partial credit for all boosted results near a click
    """
    
    def __init__(self, bandit, guardrail, extractor, boost_candidates: int = 5):
        self.bandit = bandit
        self.guardrail = guardrail
        self.extractor = extractor
        self.boost_candidates = boost_candidates
        
    def rerank(self, results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        decisions = []
        
        for r in results:
            r['display_position'] = r['position']
            decisions.append({
                'original_pos': r['position'],
                'action': 0,
                'new_pos': r['position'],
                'carbon': r['carbon_score'],
                'domain': r['domain'],
                'was_boosted': False,
            })
        
        n_results = len(results)
        candidate_indices = list(range(n_results))
        np.random.shuffle(candidate_indices)
        candidate_indices = candidate_indices[:self.boost_candidates]
        
        for i in candidate_indices:
            result = results[i]
            context = self.extractor.extract(result)
            action, scores = self.bandit.predict(context)
            
            if not self.guardrail.is_allowed(action):
                action = 0
                
            if action > 0:
                new_pos = max(0, i - action)
                result['display_position'] = new_pos
                decisions[i] = {
                    'original_pos': i,
                    'action': action,
                    'new_pos': new_pos,
                    'carbon': result['carbon_score'],
                    'domain': result['domain'],
                    'was_boosted': True,
                }
        
        reranked = sorted(results, key=lambda x: x['display_position'])
        return reranked, decisions
    
    def learn_from_click(
        self, 
        results: List[Dict], 
        decisions: List[Dict], 
        clicked_pos: Optional[int],
        lambda_value: float,
    ):
        """
        Counterfactual reward: ALL boosted results get partial credit
        based on their distance from the clicked position.
        
        KEY FIX: No penalty for no-click. Only give rewards when clicks happen.
        """
        self.guardrail.record(clicked_pos is not None)
        
        if clicked_pos is None:
            # No click — DON'T penalize. Just record and move on.
            # Penalizing teaches the bandit "boosting = bad" which is wrong.
            return
        
        # Click happened — compute counterfactual rewards for ALL boosted results
        for i, decision in enumerate(decisions):
            if not decision['was_boosted']:
                continue
                
            result = results[i]
            context = self.extractor.extract(result)
            
            # Distance between this boosted result's new position and click
            distance = abs(decision['new_pos'] - clicked_pos)
            
            # Counterfactual weight based on distance
            if distance == 0:
                weight = 1.0       # Direct click on boosted result
            elif distance == 1:
                weight = 0.6       # Clicked right next to it
            elif distance == 2:
                weight = 0.3       # Two positions away
            elif distance <= 4:
                weight = 0.15      # Within 4 positions
            else:
                weight = 0.05      # Far away — tiny signal
            
            # Carbon-aware reward
            base_reward = 1.0 - (lambda_value * result['carbon_score'])
            base_reward = max(0.0, base_reward)
            
            # Counterfactual reward
            cf_reward = weight * base_reward
            
            self.bandit.update(decision['action'], context, cf_reward)


# ============================================================================
# PART 7: EVALUATION
# ============================================================================

class Evaluator:
    def __init__(self):
        self.searches = 0
        self.total_clicks = 0
        self.clicked_carbons = []
        self.shown_carbons = []
        self.boosts = []
        self.rewards = []
        self.green_boosts = 0
        self.dirty_boosts = 0
        
    def record(self, results, decisions, clicked_pos, reward):
        self.searches += 1
        
        if clicked_pos is not None:
            self.total_clicks += 1
            for r in results:
                if r.get('display_position', r['position']) == clicked_pos:
                    self.clicked_carbons.append(r['carbon_score'])
                    break
                    
        for r in results:
            self.shown_carbons.append(r['carbon_score'])
            
        for d in decisions:
            self.boosts.append(d['action'])
            if d['action'] > 0:
                if d['carbon'] < 0.3:
                    self.green_boosts += 1
                elif d['carbon'] > 0.6:
                    self.dirty_boosts += 1
            
        self.rewards.append(reward)
        
    def summary(self) -> Dict:
        return {
            'searches': self.searches,
            'ctr': self.total_clicks / self.searches if self.searches > 0 else 0,
            'avg_carbon_shown': np.mean(self.shown_carbons) if self.shown_carbons else 0,
            'avg_carbon_clicked': np.mean(self.clicked_carbons) if self.clicked_carbons else 0,
            'carbon_reduction': (
                np.mean(self.shown_carbons) - np.mean(self.clicked_carbons)
                if self.clicked_carbons else 0
            ),
            'avg_boost': np.mean(self.boosts) if self.boosts else 0,
            'pct_boosted': sum(1 for b in self.boosts if b > 0) / len(self.boosts) if self.boosts else 0,
            'avg_reward': np.mean(self.rewards) if self.rewards else 0,
            'green_boost_ratio': self.green_boosts / max(1, self.green_boosts + self.dirty_boosts),
        }


# ============================================================================
# PART 8: EXPERIMENT
# ============================================================================

def run_experiment(
    n_searches: int = 3000,
    lambda_value: float = 0.5,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    
    gen = SearchDataGenerator(seed=seed)
    bandit = ContextualBandit(lambda_value=lambda_value, alpha=2.0, epsilon=0.30)
    guardrail = CTRGuardrail()
    extractor = FeatureExtractor()
    user = UserSimulator(seed=seed + 1)
    pipeline = RerankingPipeline(bandit, guardrail, extractor, boost_candidates=5)
    evaluator = Evaluator()
    
    if verbose:
        print("Warmup phase (100 searches, no boosting)...")
    
    for i in range(100):
        search = gen.generate_search()
        results = search['results']
        for r in results:
            r['display_position'] = r['position']
        clicked = user.click(results)
        guardrail.record(clicked is not None)
        
    guardrail.set_baseline()
    
    if verbose:
        print(f"Baseline CTR: {guardrail.baseline_ctr:.4f}")
        print(f"Candidates per search: 5")
        print(f"Using COUNTERFACTUAL rewards (no penalty for no-click)")
        print(f"Starting experiment with λ={lambda_value}...\n")
    
    for search_num in range(n_searches):
        search = gen.generate_search()
        results = search['results']
        
        reranked, decisions = pipeline.rerank(results)
        clicked_pos = user.click(reranked)
        
        # Direct reward for evaluation
        reward = 0.0
        if clicked_pos is not None:
            for r in results:
                if r.get('display_position', r['position']) == clicked_pos:
                    reward = 1.0 - (lambda_value * r['carbon_score'])
                    reward = max(0.0, reward)
                    break
        
        # Counterfactual learning
        pipeline.learn_from_click(results, decisions, clicked_pos, lambda_value)
        
        # Slower decay
        bandit.decay_epsilon(0.9995)
        
        evaluator.record(results, decisions, clicked_pos, reward)
        
        if verbose and (search_num + 1) % 300 == 0:
            s = evaluator.summary()
            print(f"Search {search_num + 1}: "
                  f"CTR={s['ctr']:.4f}, "
                  f"ClickC={s['avg_carbon_clicked']:.4f}, "
                  f"ShowC={s['avg_carbon_shown']:.4f}, "
                  f"Δ={s['carbon_reduction']:+.4f}, "
                  f"Boost={s['avg_boost']:.2f}, "
                  f"%Boost={s['pct_boosted']:.3f}, "
                  f"Green%={s['green_boost_ratio']:.3f}, "
                  f"ε={bandit.epsilon:.3f}")
    
    return {
        'evaluation': evaluator.summary(),
        'bandit_updates': bandit.update_counts,
        'bandit_rewards': {a: np.mean(rewards) if rewards else 0 
                          for a, rewards in bandit.action_rewards.items()},
        'bandit_theta': {a: bandit.theta[a].tolist() for a in range(bandit.n_actions)},
        'final_epsilon': bandit.epsilon,
    }


def run_lambda_sweep(lambda_values: List[float], n_searches: int = 2000):
    print("=" * 95)
    print("LAMBDA SWEEP — Counterfactual Rewards v5")
    print("=" * 95)
    print(f"{'λ':<6} {'CTR':<8} {'ClickC':<12} {'ShowC':<12} {'ΔCarbon':<10} {'Boost':<8} {'%Boost':<8} {'Green%':<8}")
    print("-" * 95)
    
    results = {}
    for lam in lambda_values:
        result = run_experiment(n_searches=n_searches, lambda_value=lam, verbose=False)
        ev = result['evaluation']
        results[lam] = ev
        print(f"{lam:<6.1f} {ev['ctr']:<8.4f} {ev['avg_carbon_clicked']:<12.4f} "
              f"{ev['avg_carbon_shown']:<12.4f} {ev['carbon_reduction']:<+10.4f} "
              f"{ev['avg_boost']:<8.2f} {ev['pct_boosted']:<8.3f} {ev['green_boost_ratio']:<8.3f}")
    
    return results


def debug_bandit_learning():
    """Detailed trace with counterfactual rewards and more boosts."""
    print("\n" + "=" * 70)
    print("DEBUG: Counterfactual Learning Trace (300 searches, 5 boosts each)")
    print("=" * 70)
    
    gen = SearchDataGenerator(seed=99)
    bandit = ContextualBandit(lambda_value=0.5, alpha=2.0, epsilon=0.30)
    extractor = FeatureExtractor()
    user = UserSimulator(seed=100)
    
    green_boosted = 0
    green_clicked_near = 0
    dirty_boosted = 0
    dirty_clicked_near = 0
    total_updates = 0
    
    for i in range(300):
        search = gen.generate_search()
        results = search['results']
        for r in results:
            r['display_position'] = r['position']
        
        # Boost 5 random results
        candidates = list(range(10))
        np.random.shuffle(candidates)
        candidates = candidates[:5]
        
        boosted = []
        for idx in candidates:
            target = results[idx]
            context = extractor.extract(target)
            action, _ = bandit.predict(context)
            
            if action > 0:
                new_pos = max(0, target['position'] - action)
                target['display_position'] = new_pos
                boosted.append((idx, action, new_pos, target['carbon_score']))
                
                if target['carbon_score'] < 0.3:
                    green_boosted += 1
                elif target['carbon_score'] > 0.6:
                    dirty_boosted += 1
        
        clicked_pos = user.click(results)
        
        # Counterfactual updates
        if clicked_pos is not None:
            for idx, action, new_pos, carbon in boosted:
                result = results[idx]
                context = extractor.extract(result)
                
                distance = abs(new_pos - clicked_pos)
                if distance == 0:
                    weight = 1.0
                elif distance == 1:
                    weight = 0.6
                elif distance == 2:
                    weight = 0.3
                elif distance <= 4:
                    weight = 0.15
                else:
                    weight = 0.05
                
                base_reward = 1.0 - (0.5 * carbon)
                cf_reward = weight * max(0.0, base_reward)
                
                bandit.update(action, context, cf_reward)
                total_updates += 1
                
                if weight >= 0.3:  # "Near" click
                    if carbon < 0.3:
                        green_clicked_near += 1
                    elif carbon > 0.6:
                        dirty_clicked_near += 1
        
        if i % 50 == 0:
            print(f"Search {i}: θ[1][0]={bandit.theta[1][0]:.4f}, "
                  f"θ[1][1]={bandit.theta[1][1]:.4f}, "
                  f"θ[2][0]={bandit.theta[2][0]:.4f}, "
                  f"θ[2][1]={bandit.theta[2][1]:.4f}, "
                  f"ε={bandit.epsilon:.3f}")
    
    print(f"\nTotal counterfactual updates: {total_updates}")
    print(f"Green boosted: {green_boosted}, near-clicked: {green_clicked_near}")
    print(f"Dirty boosted: {dirty_boosted}, near-clicked: {dirty_clicked_near}")
    
    print(f"\nFinal theta vectors:")
    for a in range(3):
        print(f"  Action {a}: carbon={bandit.theta[a][0]:.4f}, "
              f"green={bandit.theta[a][1]:.4f}, "
              f"position={bandit.theta[a][2]:.4f}, "
              f"relevance={bandit.theta[a][3]:.4f}")
    
    print(f"\nInterpretation:")
    for a in [1, 2]:
        carbon_sign = bandit.theta[a][0]
        green_sign = bandit.theta[a][1]
        
        if carbon_sign < -0.05 and green_sign > 0.05:
            print(f"  Action {a}: STRONG green preference ✓✓")
        elif carbon_sign < 0 and green_sign > 0:
            print(f"  Action {a}: Mild green preference ✓")
        elif carbon_sign < 0 or green_sign > 0:
            print(f"  Action {a}: Emerging green preference ~")
        else:
            print(f"  Action {a}: No clear preference ✗")
    
    # Show what the bandit would choose for green vs dirty
    print(f"\nPrediction test:")
    green_context = np.array([0.05, 0.95, 0.5, 0.5, 1.0])   # Very green
    dirty_context = np.array([0.90, 0.10, 0.5, 0.5, 1.0])   # Very dirty
    
    green_action, green_scores = bandit.predict(green_context)
    dirty_action, dirty_scores = bandit.predict(dirty_context)
    
    print(f"  Green result → action {green_action} (scores: {np.round(green_scores, 3)})")
    print(f"  Dirty result → action {dirty_action} (scores: {np.round(dirty_scores, 3)})")
    
    if green_action > dirty_action:
        print(f"  ✓ Bandit boosts green more than dirty!")
    elif green_action == dirty_action:
        print(f"  ~ Bandit treats them equally")
    else:
        print(f"  ✗ Bandit boosts dirty more — wrong direction")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SUSTAINABLE SEARCH RERANKING — v5 Counterfactual Rewards")
    print("=" * 70)
    print()
    print("v5 Changes:")
    print("  - 5 boost candidates per search (was 3)")
    print("  - No penalty for no-click (was -0.01)")
    print("  - Slower epsilon decay (0.9995 vs 0.999)")
    print("  - Green boost ratio tracking")
    print()
    
    print("--- EXPERIMENT 1: Single Run (λ=0.5) ---\n")
    result = run_experiment(n_searches=3000, lambda_value=0.5)
    
    ev = result['evaluation']
    print(f"\nFINAL RESULTS:")
    print(f"  CTR: {ev['ctr']:.4f}")
    print(f"  Avg Carbon (shown):  {ev['avg_carbon_shown']:.4f}")
    print(f"  Avg Carbon (clicked): {ev['avg_carbon_clicked']:.4f}")
    print(f"  Carbon Reduction: {ev['carbon_reduction']:+.4f}")
    print(f"  (Positive = clicked greener)")
    print(f"  Avg Boost: {ev['avg_boost']:.2f}")
    print(f"  % Boosted: {ev['pct_boosted']:.1%}")
    print(f"  Green Boost Ratio: {ev['green_boost_ratio']:.3f}")
    print(f"  (>0.5 means bandit prefers boosting green)")
    print(f"  Final epsilon: {result['final_epsilon']:.3f}")
    print(f"  Bandit updates: {result['bandit_updates']}")
    print(f"  Theta action 1: {[round(x, 4) for x in result['bandit_theta'][1]]}")
    print(f"  Theta action 2: {[round(x, 4) for x in result['bandit_theta'][2]]}")
    
    print("\n\n--- EXPERIMENT 2: Lambda Sweep ---\n")
    sweep = run_lambda_sweep([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], n_searches=2000)
    
    c0 = sweep[0.0]['avg_carbon_clicked']
    c1 = sweep[1.0]['avg_carbon_clicked']
    print(f"\n  λ=0.0 carbon clicked: {c0:.4f}")
    print(f"  λ=1.0 carbon clicked: {c1:.4f}")
    print(f"  Difference: {c0 - c1:+.4f}")
    if c0 - c1 > 0.01:
        print("  ✓ Lambda is working — higher λ reduces carbon of clicks!")
    elif c0 - c1 > 0:
        print("  ~ Lambda showing directional effect")
    else:
        print("  ~ No clear lambda effect yet")
    
    print("\n\n--- DEBUG: Learning Trace ---")
    debug_bandit_learning()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)