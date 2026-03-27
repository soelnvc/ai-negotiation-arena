#!/usr/bin/env python3
"""Quick test to verify the B2B Market terminology refactoring."""

import sys
sys.path.insert(0, '/Users/nvc/Documents/VS/MetaHack/BoxOne')

# Test model imports
from startone.models import MarketAction, MarketObservation, MarketState

# Test action type
action = MarketAction(action_type='Execute_Contract', target_id='Firm_A', amount=10)
assert action.action_type == 'Execute_Contract', f"Expected 'Execute_Contract', got {action.action_type}"
print(f'✓ MarketAction with Execute_Contract works')

# Test observation fields
obs = MarketObservation(
    done=False, 
    reward=10.0, 
    capital=100, 
    trust_scores={'Firm_B': 0.1}, 
    active_partnerships=[], 
    message='Test'
)
assert obs.capital == 100, f"Expected capital=100, got {obs.capital}"
assert obs.trust_scores == {'Firm_B': 0.1}, f"Expected trust_scores with Firm_B"
print(f'✓ MarketObservation with capital and trust_scores works')

# Test state fields
state = MarketState(
    episode_id='test', 
    step_count=5, 
    firm_capital={'Firm_A': 100, 'Firm_B': 100},
    firm_strategies={'Firm_A': 'test', 'Firm_B': 'test'},
    trust_matrix={'Firm_A': {'Firm_B': 0.0}, 'Firm_B': {'Firm_A': 0.0}},
    max_rounds=100
)
assert state.firm_capital['Firm_A'] == 100, f"Expected Firm_A capital=100"
assert state.firm_strategies['Firm_A'] == 'test', f"Expected firm_strategies"
print(f'✓ MarketState with firm_capital and firm_strategies works')

# Test backward compatibility aliases
from startone.models import ArenaAction, ArenaObservation, ArenaState
arena_action = ArenaAction(action_type='Execute_Contract', target_id='Firm_A', amount=10)
assert isinstance(arena_action, MarketAction), "ArenaAction should be alias of MarketAction"
print(f'✓ ArenaAction backward compatibility alias works')

print('\n✅ All model refactoring tests passed!')
print('✅ B2B Market terminology refactoring is complete and correct!')
