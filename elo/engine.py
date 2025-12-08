"""ELO Rating Engine for Poker AI Models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from common import (
    create_agent, load_model_agent, ModelCache,
    ELO_STARTING_STACK, ELO_BIG_BLIND, ELO_SMALL_BLIND,
    ELO_ROUNDS_PER_MATCH, ELO_MAX_HANDS_PER_ROUND,
    GameConfig, play_hand_direct, HandLogger,
)
import hashlib


@dataclass
class EloRating:
    """ELO rating with history tracking."""
    rating: float = 2500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    history: List[Tuple[int, float]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.history:
            self.history.append((0, self.rating))
    
    def update(self, new_rating: float, score: float, match_num: int):
        self.rating = new_rating
        self.games_played += 1
        self.history.append((match_num, new_rating))
        if score == 1.0:
            self.wins += 1
        elif score == 0.0:
            self.losses += 1
        else:
            self.draws += 1
    
    @property
    def win_rate(self) -> float:
        return self.wins / self.games_played if self.games_played > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rating': self.rating, 'games_played': self.games_played,
            'wins': self.wins, 'losses': self.losses, 'draws': self.draws, 'win_rate': self.win_rate,
        }


def calculate_elo(rating_a: float, rating_b: float, score_a: float, k: float = 40.0) -> Tuple[float, float]:
    """Calculate new ELO ratings. score_a: 1=win, 0.5=draw, 0=loss."""
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    return rating_a + k * (score_a - expected_a), rating_b + k * ((1 - score_a) - (1 - expected_a))


class PokerEloArena:
    """Arena for running poker matches and tracking ELO ratings."""
    
    def __init__(
        self,
        device: str = "cpu",
        k_factor: float = 40.0,
        initial_rating: float = 2500.0,
        max_cached_models: int = 2,
        *,
        base_seed: Optional[int] = 0,
        starting_stack: int = ELO_STARTING_STACK,
        small_blind: int = ELO_SMALL_BLIND,
        big_blind: int = ELO_BIG_BLIND,
        rounds_per_match: int = ELO_ROUNDS_PER_MATCH,
        max_hands_per_round: int = ELO_MAX_HANDS_PER_ROUND,
    ):
        self.device = torch.device(device)
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        # If set, all match/hand RNG is derived from this so runs are reproducible.
        # Set to None to use nondeterministic randomness.
        self.base_seed: Optional[int] = int(base_seed) if base_seed is not None else None
        self.starting_stack = int(starting_stack)
        self.rounds_per_match = int(rounds_per_match)
        self.max_hands_per_round = int(max_hands_per_round)
        # NOTE: We intentionally do NOT use the stateless `PokerSimulator` here:
        # Elo matches are freezeouts with changing, per-player stacks.
        # The stateless API only supports a single global startingChips value,
        # which breaks freezeout logic by resetting both players to min(stack_a, stack_b) each hand.
        self._game_cfg = GameConfig(
            num_players=2, small_blind=int(small_blind), big_blind=int(big_blind), starting_chips=self.starting_stack,
        )
        self.ratings: Dict[str, EloRating] = {}
        self.matches: List[Dict[str, Any]] = []
        self._cache = ModelCache(self.device, max_cached_models)
        self.hand_logger = HandLogger()
        # Deterministic seeding should not depend on global match scheduling order.
        # Track "how many times this pair has played" so the same pair gets the
        # same deck sequence regardless of when it's scheduled.
        self._pair_match_counts: Dict[Tuple[str, str], int] = {}

    def _seed_label(self, player_id: str, config: Dict[str, Any]) -> str:
        """
        Stable label used for deterministic deck seeding.

        By default we use `player_id`. Callers may override with `config['seed_id']`
        to keep deck sequences comparable across checkpoints (e.g. iter_1 vs iter_700
        against the same heuristic opponent).
        """
        try:
            sid = config.get("seed_id")
            if sid is not None and str(sid).strip() != "":
                return str(sid)
        except Exception:
            pass
        return str(player_id)

    def _pair_key(self, seed_a: str, seed_b: str) -> Tuple[str, str]:
        """Canonicalize pair identity so seeding is symmetric w.r.t player order."""
        a = str(seed_a)
        b = str(seed_b)
        return (a, b) if a <= b else (b, a)

    def _seed_for(self, pair: Tuple[str, str], pair_match_idx: int, round_idx: int, hand_idx: int) -> Optional[int]:
        """
        Deterministically derive a 32-bit seed for a specific hand.

        We intentionally avoid Python's built-in `hash()` because it is salted per process.
        """
        if self.base_seed is None:
            return None
        a, b = pair
        key = f"{self.base_seed}|{a}|{b}|pm{pair_match_idx}|r{round_idx}|h{hand_idx}".encode("utf-8")
        # Use 32-bit seed range for broad compatibility with backends.
        return int(hashlib.blake2b(key, digest_size=4).hexdigest(), 16)
    
    def get_or_create_rating(self, player_id: str) -> EloRating:
        if player_id not in self.ratings:
            self.ratings[player_id] = EloRating(rating=self.initial_rating)
        return self.ratings[player_id]
    
    def _create_agent(self, player_id: str, config: Dict) -> Any:
        """Create agent from config (model or heuristic)."""
        if config['type'] == 'model':
            model = config.get('model')
            if model is None:
                model, err = self._cache.get_with_error(config['path'])
                if model is None:
                    raise RuntimeError(f"Failed to load model checkpoint: {config['path']}\n{err or ''}".strip())
                config['model'] = model
            return load_model_agent(
                player_id,
                config.get('name', player_id),
                model=model,
                device=self.device,
                deterministic=config.get('deterministic', True),
            )
        return create_agent(config['type'], player_id, config.get('name')) or create_agent('random', player_id, config.get('name'))

    def _play_freezeout_round(
        self,
        agent_a,
        agent_b,
        config_a: Dict,
        config_b: Dict,
        *,
        player_a_id: str,
        player_b_id: str,
        match_idx: int,
        round_idx: int,
    ) -> Dict[str, Any]:
        """Play freezeout round until one player busts."""
        stack_a = stack_b = self.starting_stack
        hands = wins_a = wins_b = 0
        
        while stack_a > 0 and stack_b > 0 and hands < self.max_hands_per_round:
            # Swap positions every other hand
            swap = hands % 2 == 1
            agents = {'p0': agent_b if swap else agent_a, 'p1': agent_a if swap else agent_b}
            configs = {'p0': config_b if swap else config_a, 'p1': config_a if swap else config_b}
            stacks = {'p0': int(stack_b if swap else stack_a), 'p1': int(stack_a if swap else stack_b)}
            
            try:
                # Use direct simulation to respect per-player stacks.
                seed_a = self._seed_label(player_a_id, config_a)
                seed_b = self._seed_label(player_b_id, config_b)
                pair = self._pair_key(seed_a, seed_b)
                seed = self._seed_for(pair, int(match_idx), round_idx, hands)
                result = play_hand_direct(agents=agents, config=self._game_cfg, starting_chips=stacks, seed=seed)
                if not result.get('success'):
                    continue
                
                hands += 1
                p0_profit, p1_profit = result.get('profits', {}).get('p0', 0), result.get('profits', {}).get('p1', 0)
                
                if swap:
                    stack_b, stack_a = stack_b + p0_profit, stack_a + p1_profit
                    wins_b, wins_a = wins_b + (p0_profit > 0), wins_a + (p1_profit > 0)
                else:
                    stack_a, stack_b = stack_a + p0_profit, stack_b + p1_profit
                    wins_a, wins_b = wins_a + (p0_profit > 0), wins_b + (p1_profit > 0)
            except Exception as e:
                print(f"Error in hand: {e}")
        
        if hands == 0:
            return {'error': True}
        
        winner = 'a' if stack_a > stack_b else ('b' if stack_b > stack_a else 'draw')
        return {
            'error': False, 'hands_played': hands, 'winner': winner,
            'final_stack_a': stack_a, 'final_stack_b': stack_b,
            'hand_wins_a': wins_a, 'hand_wins_b': wins_b,
        }

    def play_match(self, player_a_id: str, player_b_id: str, config_a: Dict, config_b: Dict) -> Dict[str, Any]:
        """Play best-of-N freezeout match and update ELO ratings."""
        # Preload models (and fail loudly if a checkpoint doesn't load)
        for cfg in (config_a, config_b):
            if cfg.get('type') == 'model' and 'path' in cfg and cfg.get('model') is None:
                m, err = self._cache.get_with_error(cfg['path'])
                if m is None:
                    return {'error': True, 'message': f"Failed to load model: {cfg.get('name', cfg.get('path'))}\n{err or ''}".strip()}
                cfg['model'] = m
        
        agent_a, agent_b = self._create_agent('p0', config_a), self._create_agent('p1', config_b)
        
        round_wins_a = round_wins_b = total_hands = total_wins_a = total_wins_b = 0
        results = []

        # Deterministic seeding should be independent of global match scheduling order.
        seed_a = self._seed_label(player_a_id, config_a)
        seed_b = self._seed_label(player_b_id, config_b)
        pair = self._pair_key(seed_a, seed_b)
        pair_match_idx = int(self._pair_match_counts.get(pair, 0)) + 1
        self._pair_match_counts[pair] = pair_match_idx
        
        for round_idx in range(self.rounds_per_match):
            r = self._play_freezeout_round(
                agent_a, agent_b, config_a, config_b,
                player_a_id=player_a_id,
                player_b_id=player_b_id,
                match_idx=pair_match_idx,
                round_idx=round_idx,
            )
            if r.get('error'):
                continue
            results.append(r)
            total_hands += r['hands_played']
            total_wins_a += r['hand_wins_a']
            total_wins_b += r['hand_wins_b']
            round_wins_a += r['winner'] == 'a'
            round_wins_b += r['winner'] == 'b'
        
        if not results:
            return {'error': True}
        
        # Calculate ELO
        score_a = 1.0 if round_wins_a > round_wins_b else (0.0 if round_wins_b > round_wins_a else 0.5)
        
        rating_a, rating_b = self.get_or_create_rating(player_a_id), self.get_or_create_rating(player_b_id)
        old_a, old_b = rating_a.rating, rating_b.rating
        new_a, new_b = calculate_elo(old_a, old_b, score_a, self.k_factor)
        
        match_num = len(self.matches) + 1
        rating_a.update(new_a, score_a, match_num)
        rating_b.update(new_b, 1 - score_a, match_num)
        
        result = {
            'match_num': match_num, 'player_a': player_a_id, 'player_b': player_b_id,
            'rounds_played': len(results), 'round_wins_a': round_wins_a, 'round_wins_b': round_wins_b,
            'hands_played': total_hands, 'hand_wins_a': total_wins_a, 'hand_wins_b': total_wins_b,
            'score_a': score_a, 'old_rating_a': old_a, 'old_rating_b': old_b,
            'new_rating_a': new_a, 'new_rating_b': new_b,
            'rating_change_a': new_a - old_a, 'rating_change_b': new_b - old_b,
        }
        self.matches.append(result)
        return result
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get leaderboard sorted by rating."""
        return sorted([{'player_id': pid, **r.to_dict()} for pid, r in self.ratings.items()], key=lambda x: x['rating'], reverse=True)
