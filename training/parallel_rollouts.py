#!/usr/bin/env python3
"""Multi-Process Rollout Collection for Poker RL Training."""

import random
import traceback
import uuid
from collections import deque
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set

import torch
import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass


class Task(str, Enum):
    """Worker task types."""
    COLLECT = "collect"
    UPDATE_WEIGHTS = "update_weights"
    UPDATE_POOL = "update_pool"
    SHUTDOWN = "shutdown"


def _new_request_id() -> str:
    # Short, unique-enough ID for correlating request/response across processes.
    return uuid.uuid4().hex[:12]


def _worker_loop(
    worker_id: int,
    game_config: Dict[str, Any],
    model_config: Dict[str, Any],
    model_state: Dict,
    task_q: mp.Queue,
    result_q: mp.Queue,
    stop_event: mp.Event,
    opponent_pool: List[str],
    device: str = 'cpu',
    base_seed: Optional[int] = None,
):
    """Worker process that collects episodes."""
    try:
        from common import create_actor_critic
        from episode_collector import EpisodeCollector
        
        model = create_actor_critic(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            dropout=model_config.get('dropout', 0.1),
            gradient_checkpointing=False
        )
        model.load_state_dict(model_state)
        model.to(torch.device(device)).eval()

        # Make opponent sampling + environment seeds reproducible per worker.
        # EpisodeCollector internally advances its RNG each episode.
        worker_seed = None if base_seed is None else int(base_seed) + int(worker_id) * 100_000
        if worker_seed is not None:
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        
        collector = EpisodeCollector(
            model=model,
            game_config=game_config,
            device=torch.device(device),
            model_config=model_config,
            opponent_pool=list(opponent_pool),
            seed=worker_seed,
        )
        
        result_q.put({"type": "ready", "worker_id": worker_id})
        
        while not stop_event.is_set():
            try:
                task = task_q.get(timeout=0.1)
            except Exception:
                continue
            
            if task is None or task.get("type") == Task.SHUTDOWN.value:
                break
            
            task_type = task.get("type")
            req_id = task.get("req_id")
            
            if task_type == Task.COLLECT.value:
                result_q.put(
                    {
                        "type": "episode",
                        "req_id": req_id,
                        "worker_id": worker_id,
                        "episode": collector.collect_episode(),
                    }
                )
            
            elif task_type == Task.UPDATE_WEIGHTS.value:
                if state := task.get("state_dict"):
                    model.load_state_dict(state)
                    model.eval()
                result_q.put({"type": "weights_updated", "req_id": req_id, "worker_id": worker_id})
            
            elif task_type == Task.UPDATE_POOL.value:
                collector.update_opponent_pool(task.get("pool", []))
                result_q.put({"type": "pool_updated", "req_id": req_id, "worker_id": worker_id})
    
    except Exception as e:
        result_q.put(
            {
                "type": "error",
                "worker_id": worker_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


class ParallelRolloutManager:
    """Manages parallel workers for episode collection."""
    
    def __init__(
        self,
        num_workers: int,
        game_config: Dict[str, Any],
        model_config: Dict[str, Any],
        model: torch.nn.Module,
        device: str = 'cpu',
        opponent_pool: Optional[List[str]] = None,
        base_seed: Optional[int] = None,
    ):
        self.num_workers = num_workers
        self.game_config = game_config
        self.model_config = model_config
        self.model = model
        self.device = device
        self.opponent_pool = opponent_pool or []
        self.base_seed = base_seed
        
        self._procs: List[mp.Process] = []
        self._queues: List[mp.Queue] = []
        self._results = mp.Queue()
        self._stop = mp.Event()
        self._started = False
        self._inbox: Deque[Dict[str, Any]] = deque()
    
    def _get_message(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get next message from inbox/results queue."""
        if self._inbox:
            return self._inbox.popleft()
        return self._results.get(timeout=timeout) if timeout is not None else self._results.get()

    def _put_back(self, msg: Dict[str, Any]) -> None:
        """Put message back into local inbox for later processing."""
        self._inbox.append(msg)

    def _wait_for_ready(self) -> None:
        ready: Set[int] = set()
        while len(ready) < self.num_workers:
            r = self._get_message()
            if r.get("type") == "ready":
                ready.add(int(r.get("worker_id", -1)))
            elif r.get("type") == "error":
                raise RuntimeError(f"Worker {r.get('worker_id')} failed: {r.get('error')}\n{r.get('traceback','')}")
            else:
                self._put_back(r)

    def _wait_for_acks(self, ack_type: str, req_id: str) -> None:
        remaining: Set[int] = set(range(self.num_workers))
        while remaining:
            r = self._get_message()
            if r.get("type") == "error":
                raise RuntimeError(f"Worker {r.get('worker_id')} failed: {r.get('error')}\n{r.get('traceback','')}")
            if r.get("type") == ack_type and r.get("req_id") == req_id:
                wid = int(r.get("worker_id", -1))
                remaining.discard(wid)
            else:
                self._put_back(r)

    def start(self, *, state_dict: Optional[Dict] = None):
        """Start worker processes."""
        if self._started:
            return

        # NOTE: Workers always construct an *unwrapped* model (via create_actor_critic).
        # If the trainer model is wrapped (e.g. Accelerate/DDP), callers should pass an
        # unwrapped state_dict here to avoid key mismatches (e.g. "module." prefixes).
        raw_state = state_dict or self.model.state_dict()
        state = {k: v.cpu() for k, v in raw_state.items()}
        
        for i in range(self.num_workers):
            q = mp.Queue()
            self._queues.append(q)
            proc = mp.Process(
                target=_worker_loop,
                args=(i, self.game_config, self.model_config, state, q,
                      self._results, self._stop, list(self.opponent_pool), self.device, self.base_seed),
                daemon=True
            )
            proc.start()
            self._procs.append(proc)
        
        # Wait for all workers
        self._wait_for_ready()
        
        self._started = True
    
    def collect_episodes(self, count: int, verbose: bool = False) -> List[Dict[str, Any]]:
        """Collect episodes in parallel across workers."""
        if not self._started:
            raise RuntimeError("Manager not started")
        
        req_id = _new_request_id()

        # Distribute tasks
        for i in range(count):
            self._queues[i % self.num_workers].put({"type": Task.COLLECT.value, "req_id": req_id})
        
        # Collect results
        episodes = []
        received = 0
        while received < count:
            r = self._get_message()
            r_type = r.get("type")
            if r_type == "error":
                raise RuntimeError(f"Worker {r.get('worker_id')} failed: {r.get('error')}\n{r.get('traceback','')}")

            # Ignore unrelated messages (e.g., delayed acks from earlier requests)
            if r_type != "episode" or r.get("req_id") != req_id:
                self._put_back(r)
                continue

            received += 1
            ep = r.get("episode", {})
            if ep.get("success") and isinstance(ep.get("states"), torch.Tensor) and len(ep["states"]) > 0:
                episodes.append(ep)
            if verbose and received % 50 == 0:
                print(f"  Collected {received}/{count}...")
        
        return episodes
    
    def update_model_weights(self, state_dict: Optional[Dict] = None):
        """Sync model weights to all workers."""
        if not self._started:
            return
        
        req_id = _new_request_id()
        raw_state = state_dict or self.model.state_dict()
        state = {k: v.cpu() for k, v in raw_state.items()}
        for q in self._queues:
            q.put({"type": Task.UPDATE_WEIGHTS.value, "req_id": req_id, "state_dict": state})
        
        self._wait_for_acks("weights_updated", req_id)
    
    def update_opponent_pool(self, pool: List[str]):
        """Update opponent pool in all workers."""
        self.opponent_pool = pool
        if not self._started:
            return
        
        req_id = _new_request_id()
        for q in self._queues:
            q.put({"type": Task.UPDATE_POOL.value, "req_id": req_id, "pool": pool})
        
        self._wait_for_acks("pool_updated", req_id)
    
    def shutdown(self):
        """Shutdown all workers gracefully."""
        if not self._started:
            return
        
        self._stop.set()
        for q in self._queues:
            try:
                q.put({"type": Task.SHUTDOWN.value})
            except Exception:
                pass
        
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        for q in self._queues:
            q.close()
        self._results.close()
        self._started = False
