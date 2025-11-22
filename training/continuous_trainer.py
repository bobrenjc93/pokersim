#!/usr/bin/env python3
"""
Continuous Poker Model Training System

This program balances system resources between data generation and model training,
allowing the system to continuously improve by:
1. Generating new training data in the background
2. Training models on available data
3. Managing resources to prevent system overload
4. Rotating data files to manage disk space

The system runs indefinitely until interrupted, automatically balancing CPU,
memory, and I/O resources between the two processes.

Usage:
    python continuous_trainer.py [options]

Examples:
    # Run with default settings
    python continuous_trainer.py

    # Custom configuration
    python continuous_trainer.py --batch-size 1000 --epochs-per-cycle 50 --max-data-files 10

    # High resource mode (more aggressive)
    python continuous_trainer.py --cpu-threshold 0.9 --memory-threshold 0.9
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import queue

# Import model version from config
from config import MODEL_VERSION

# Try to import psutil, provide fallback if not available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  Warning: psutil not available. Resource monitoring will be limited.")
    print("   Install with: pip install psutil")
    print("")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SystemConfig:
    """Configuration for continuous training system"""
    # Resource thresholds
    cpu_threshold: float = 0.80  # Max CPU usage before throttling
    memory_threshold: float = 0.85  # Max memory usage before throttling
    
    # Data generation
    rollouts_per_batch: int = 5000  # Rollouts to generate per batch
    num_players: int = 3
    small_blind: int = 25
    big_blind: int = 50
    starting_chips: int = 2000
    agent_type: str = "mixed"
    
    # Training
    epochs_per_cycle: int = 50  # Epochs to train per cycle
    batch_size: int = 32
    learning_rate: float = 0.001
    hidden_dim: int = 256
    
    # Adaptive training schedule
    adaptive_schedule: bool = True  # Enable adaptive dataset growth
    initial_data_fraction: float = 0.15  # Start with 15% of data
    data_growth_factor: float = 1.5  # Grow by 1.5x when plateauing
    plateau_patience: int = 5  # Epochs without improvement before growing
    plateau_threshold: float = 0.001  # Minimum improvement threshold
    
    # Evaluation
    eval_enabled: bool = True  # Whether to run evaluation
    eval_interval: int = 1  # Run evaluation every N training cycles
    eval_num_hands: int = 100  # Number of hands to play in evaluation
    
    # File management
    max_data_files: int = 10  # Max training data files to keep (0 = keep all)
    data_dir: Path = Path(f"/tmp/pokersim/data_v{MODEL_VERSION}")
    model_dir: Path = Path(f"/tmp/pokersim/models_v{MODEL_VERSION}")
    accumulate_data: bool = True  # If True, train on all accumulated data; if False, train on single batches
    
    # API server
    api_port: int = 8080
    api_url: str = "http://localhost:8080/simulate"
    
    # Timing
    resource_check_interval: float = 5.0  # Seconds between resource checks
    min_data_gen_interval: float = 0.0  # Min seconds between data generation batches
    
    # Pipelining
    initial_buffer_size: int = 1  # Initial batches to generate before training starts
    max_concurrent_datagen: int = 2  # Max concurrent data generation processes
    
    # Priorities
    training_priority: int = 10  # Nice value for training (lower = higher priority)
    datagen_priority: int = 19  # Nice value for data generation


# =============================================================================
# Resource Monitor
# =============================================================================

class ResourceMonitor:
    """Monitor system resources and provide throttling recommendations"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.history = deque(maxlen=12)  # Keep last minute of samples (5s intervals)
    
    def check_resources(self) -> dict:
        """Check current system resource usage"""
        if HAS_PSUTIL:
            cpu_percent = psutil.cpu_percent(interval=1.0) / 100.0
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent / 100.0
        else:
            # Fallback: assume moderate usage if psutil not available
            cpu_percent = 0.5
            memory_percent = 0.5
            disk_percent = 0.5
        
        stats = {
            'cpu': cpu_percent,
            'memory': memory_percent,
            'disk': disk_percent,
            'timestamp': time.time()
        }
        
        self.history.append(stats)
        return stats
    
    def should_throttle_datagen(self) -> bool:
        """Determine if data generation should be throttled"""
        if not self.history:
            return False
        
        latest = self.history[-1]
        return (latest['cpu'] > self.config.cpu_threshold or 
                latest['memory'] > self.config.memory_threshold or
                latest['disk'] > 0.95)
    
    def can_start_training(self) -> bool:
        """Determine if training can start"""
        if not self.history:
            return True
        
        latest = self.history[-1]
        # More lenient for training since it's the priority
        return (latest['memory'] < 0.95 and latest['disk'] < 0.98)
    
    def get_summary(self) -> str:
        """Get a summary of current resource usage"""
        if not self.history:
            return "No data"
        
        latest = self.history[-1]
        return (f"CPU: {latest['cpu']*100:.1f}% | "
                f"Memory: {latest['memory']*100:.1f}% | "
                f"Disk: {latest['disk']*100:.1f}%")


# =============================================================================
# Process Manager
# =============================================================================

@dataclass
class DataGenJob:
    """Represents an active data generation job"""
    process: subprocess.Popen
    output_file: Path
    start_time: float


class ProcessManager:
    """Manage subprocesses for API server, data generation, and training"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.api_process: Optional[subprocess.Popen] = None
        self.datagen_jobs: List[DataGenJob] = []  # Track multiple concurrent data gen processes
        self.training_process: Optional[subprocess.Popen] = None
        self.project_root = Path(__file__).parent.parent
        self.datagen_lock = threading.Lock()  # Protect datagen_jobs access
    
    def start_api_server(self) -> bool:
        """Start the C++ API server"""
        try:
            # Build first if needed
            print("🔧 Building API server...")
            build_result = subprocess.run(
                ["make"],
                cwd=self.project_root / "api",
                capture_output=True,
                text=True
            )
            
            if build_result.returncode != 0:
                print(f"✗ Failed to build API server: {build_result.stderr}")
                return False
            
            print(f"🚀 Starting API server on port {self.config.api_port}...")
            self.api_process = subprocess.Popen(
                [str(self.project_root / "api/build/poker_api"), str(self.config.api_port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for server to be ready
            for i in range(20):
                try:
                    import requests
                    response = requests.post(
                        self.config.api_url,
                        json={"config": {"seed": 1}},
                        timeout=1
                    )
                    print(f"✓ API server ready (PID: {self.api_process.pid})")
                    return True
                except:
                    if i == 19:
                        print("✗ API server failed to start")
                        return False
                    time.sleep(0.5)
            
            return False
        except Exception as e:
            print(f"✗ Error starting API server: {e}")
            return False
    
    def stop_api_server(self):
        """Stop the API server"""
        if self.api_process:
            print("Stopping API server...")
            self.api_process.terminate()
            try:
                self.api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.api_process.kill()
            self.api_process = None
    
    def start_data_generation(self, output_file: Path) -> bool:
        """Start a data generation job (non-blocking)"""
        with self.datagen_lock:
            try:
                # Check if we're at max concurrent jobs
                if len(self.datagen_jobs) >= self.config.max_concurrent_datagen:
                    return False
                

                
                # Make output_file absolute if it's relative
                abs_output_file = output_file if output_file.is_absolute() else (self.project_root / output_file)
                
                cmd = [
                    "uv", "run", "--with", "requests", "python",
                    "generate_rollouts.py",
                    "--num-rollouts", str(self.config.rollouts_per_batch),
                    "--num-players", str(self.config.num_players),
                    "--small-blind", str(self.config.small_blind),
                    "--big-blind", str(self.config.big_blind),
                    "--starting-chips", str(self.config.starting_chips),
                    "--output", str(abs_output_file),
                    "--agent-type", self.config.agent_type,
                    "--api-url", self.config.api_url
                ]
                
                process = subprocess.Popen(
                    cmd,
                    cwd=self.project_root / "training",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                # Set lower priority for data generation
                if HAS_PSUTIL:
                    try:
                        p = psutil.Process(process.pid)
                        p.nice(self.config.datagen_priority)
                    except:
                        pass
                
                # Add to active jobs
                job = DataGenJob(process=process, output_file=output_file, start_time=time.time())
                self.datagen_jobs.append(job)
                return True
                
            except Exception as e:
                print(f"✗ Error starting data generation: {e}")
                return False
    
    def poll_data_generation(self) -> List[Path]:
        """Check all active data generation jobs and return completed ones"""
        completed_files = []
        
        with self.datagen_lock:
            remaining_jobs = []
            
            for job in self.datagen_jobs:
                # Check if process has completed
                returncode = job.process.poll()
                
                if returncode is not None:
                    # Process has finished
                    try:
                        stdout, _ = job.process.communicate(timeout=0.1)
                    except:
                        stdout = None
                    
                    if returncode == 0:
                        print(f"✓ Data batch generated: {job.output_file.name}")
                        completed_files.append(job.output_file)
                    else:
                        print(f"✗ Data generation failed for {job.output_file.name}")
                        if stdout:
                            print(f"  Output: {stdout[-500:]}")  # Last 500 chars
                else:
                    # Still running
                    remaining_jobs.append(job)
            
            self.datagen_jobs = remaining_jobs
        
        return completed_files
    
    def train_model(self, data_file: Path, model_output: Path) -> bool:
        """Train the model on a data file or directory"""
        try:
            # Determine what to pass to training
            if self.config.accumulate_data:
                # Count total data files for status message
                data_files = list(self.config.data_dir.glob("batch_*.json"))
                num_files = len(data_files)
                print(f"🎓 Training model on {num_files} accumulated data files for {self.config.epochs_per_cycle} epochs...")
                
                # Pass the entire data directory
                abs_data_path = self.config.data_dir if self.config.data_dir.is_absolute() else (self.project_root / self.config.data_dir)
            else:
                # Train on single file (original behavior)
                print(f"🎓 Training model for {self.config.epochs_per_cycle} epochs...")
                abs_data_path = data_file if data_file.is_absolute() else (self.project_root / data_file)
            
            abs_model_output = model_output if model_output.is_absolute() else (self.project_root / model_output)
            
            # Check for existing checkpoint to resume from
            main_model = self.config.model_dir / "poker_model.pt"
            checkpoint_path = None
            if main_model.exists():
                checkpoint_path = main_model if main_model.is_absolute() else (self.project_root / main_model)
                print(f"  📂 Resuming from checkpoint: {main_model.name}")
            
            # Use absolute paths directly for train.py
            cmd = [
                "uv", "run", "--with", "torch", "python",
                "train.py",
                "--data", str(abs_data_path),
                "--output", str(abs_model_output),
                "--epochs", str(self.config.epochs_per_cycle),
                "--batch-size", str(self.config.batch_size),
                "--lr", str(self.config.learning_rate),
                "--hidden-dim", str(self.config.hidden_dim)
            ]
            
            # Add adaptive schedule arguments if enabled
            if self.config.adaptive_schedule:
                cmd.extend([
                    "--adaptive-schedule",
                    "--initial-data-fraction", str(self.config.initial_data_fraction),
                    "--data-growth-factor", str(self.config.data_growth_factor),
                    "--plateau-patience", str(self.config.plateau_patience),
                    "--plateau-threshold", str(self.config.plateau_threshold)
                ])
            
            # Add checkpoint argument if available
            if checkpoint_path:
                cmd.extend(["--checkpoint", str(checkpoint_path)])
            
            self.training_process = subprocess.Popen(
                cmd,
                cwd=self.project_root / "training",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Set higher priority for training
            if HAS_PSUTIL:
                try:
                    p = psutil.Process(self.training_process.pid)
                    p.nice(self.config.training_priority)
                except:
                    pass
            
            # Stream output
            last_line = ""
            if self.training_process.stdout:
                for line in self.training_process.stdout:
                    line = line.strip()
                    if line:
                        # Only print epoch lines to reduce clutter
                        if "Epoch" in line or "loss" in line.lower():
                            print(f"  {line}")
                        last_line = line
            
            self.training_process.wait()
            success = self.training_process.returncode == 0
            self.training_process = None
            
            if success:
                print(f"✓ Training complete: {model_output.name}")
            else:
                print(f"✗ Training failed")
            
            return success
        except Exception as e:
            print(f"✗ Error training model: {e}")
            self.training_process = None
            return False
    
    def evaluate_model(self, model_path: Path) -> bool:
        """Evaluate the model by playing against random agents"""
        try:
            print(f"\n{'='*70}")
            print(f"🎯 Evaluating model: {model_path.name}")
            print(f"{'='*70}\n")
            
            # Make path absolute if relative
            abs_model_path = model_path if model_path.is_absolute() else (self.project_root / model_path)
            
            if not abs_model_path.exists():
                print(f"✗ Model not found: {abs_model_path}")
                return False
            
            cmd = [
                "uv", "run", "--with", "torch", "--with", "requests", "python",
                "eval.py",
                "--model", str(abs_model_path),
                "--num-hands", str(self.config.eval_num_hands),
                "--num-players", str(self.config.num_players),
                "--api-url", self.config.api_url,
                "--small-blind", str(self.config.small_blind),
                "--big-blind", str(self.config.big_blind),
                "--starting-chips", str(self.config.starting_chips)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root / "training",
                capture_output=False,  # Let output go to stdout
                text=True
            )
            
            success = result.returncode == 0
            
            if success:
                print(f"\n✓ Evaluation complete\n")
            else:
                print(f"\n✗ Evaluation failed\n")
            
            return success
        except Exception as e:
            print(f"✗ Error evaluating model: {e}")
            return False
    
    def cleanup(self):
        """Clean up all processes"""
        # Terminate all data generation jobs
        with self.datagen_lock:
            for job in self.datagen_jobs:
                try:
                    job.process.terminate()
                except:
                    pass
            self.datagen_jobs = []
        
        if self.training_process:
            self.training_process.terminate()
            self.training_process = None
        self.stop_api_server()


# =============================================================================
# Data Manager
# =============================================================================

class DataManager:
    """Manage training data files and rotation"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.data_queue = queue.Queue()
        self.data_counter = 0
        self.queued_files = set()  # Track files currently in queue to prevent deletion
        self.queue_lock = threading.Lock()  # Protect queued_files set
        
        # Ensure directories exist
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
    
    def get_next_data_file(self) -> Path:
        """Get the path for the next data file to generate"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_counter += 1
        filename = f"batch_{timestamp}_{self.data_counter:04d}.json"
        return self.config.data_dir / filename
    
    def get_next_model_file(self) -> Path:
        """Get the path for the next model file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"poker_model_{timestamp}.pt"
        return self.config.model_dir / filename
    
    def add_data_file(self, filepath: Path):
        """Add a data file to the training queue"""
        # Make path absolute if it's relative
        project_root = Path(__file__).parent.parent
        abs_path = filepath if filepath.is_absolute() else (project_root / filepath)
        if abs_path.exists():
            with self.queue_lock:
                self.data_queue.put(filepath)
                self.queued_files.add(abs_path)  # Track this file to prevent deletion
    
    def get_data_file(self, block: bool = False, timeout: Optional[float] = None) -> Optional[Path]:
        """Get a data file from the queue"""
        try:
            return self.data_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def mark_file_complete(self, filepath: Path):
        """Mark a file as complete (no longer needed after training)"""
        # Make path absolute if it's relative
        project_root = Path(__file__).parent.parent
        abs_path = filepath if filepath.is_absolute() else (project_root / filepath)
        with self.queue_lock:
            self.queued_files.discard(abs_path)
    
    def cleanup_old_data(self):
        """Remove old data files to manage disk space (if not accumulating data)"""
        # Skip cleanup if accumulating all data or max_data_files is 0
        if self.config.accumulate_data or self.config.max_data_files == 0:
            return
        
        data_files = sorted(self.config.data_dir.glob("batch_*.json"), key=lambda p: p.stat().st_mtime)
        
        while len(data_files) > self.config.max_data_files:
            old_file = data_files.pop(0)
            
            # RACE CONDITION PROTECTION: Don't delete files still in the training queue
            with self.queue_lock:
                if old_file in self.queued_files:
                    print(f"⏭️  Skipping {old_file.name} (still in training queue)")
                    continue
            
            try:
                old_file.unlink()
                print(f"🗑️  Removed old data file: {old_file.name}")
            except Exception as e:
                print(f"⚠️  Failed to remove {old_file.name}: {e}")
    
    def get_latest_model(self) -> Optional[Path]:
        """Get the most recent model file"""
        model_files = sorted(self.config.model_dir.glob("poker_model_*.pt"), key=lambda p: p.stat().st_mtime)
        return model_files[-1] if model_files else None


# =============================================================================
# Continuous Trainer
# =============================================================================

class ContinuousTrainer:
    """Main coordinator for continuous training system"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.resource_monitor = ResourceMonitor(config)
        self.process_manager = ProcessManager(config)
        self.data_manager = DataManager(config)
        
        self.running = False
        self.stats = {
            'batches_generated': 0,
            'training_cycles': 0,
            'start_time': None,
            'last_data_gen': 0,
            'last_training': 0
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals gracefully"""
        print("\n\n🛑 Shutdown signal received, cleaning up...")
        self.running = False
    
    def _print_status(self) -> None:
        """Print current status"""
        uptime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        
        resources = self.resource_monitor.get_summary()
        latest_model = self.data_manager.get_latest_model()
        
        # Count accumulated data files
        data_files = list(self.config.data_dir.glob("batch_*.json"))
        num_data_files = len(data_files)
        
        print(f"\n{'='*70}")
        print(f"  Continuous Training Status")
        print(f"{'='*70}")
        print(f"  Uptime: {hours}h {minutes}m")
        print(f"  Data batches generated: {self.stats['batches_generated']}")
        print(f"  Data files accumulated: {num_data_files}")
        print(f"  Training mode: {'Accumulating all data' if self.config.accumulate_data else 'Single batch'}")
        print(f"  Training cycles completed: {self.stats['training_cycles']}")
        print(f"  Latest model: {latest_model.name if latest_model else 'None'}")
        print(f"  Resources: {resources}")
        print(f"  Data queue size: {self.data_manager.data_queue.qsize()}")
        print(f"{'='*70}\n")
    
    def _data_generation_loop(self):
        """Background thread for continuous data generation"""
        while self.running:
            try:
                # First, poll for completed jobs
                completed_files = self.process_manager.poll_data_generation()
                for output_file in completed_files:
                    self.data_manager.add_data_file(output_file)
                    self.stats['batches_generated'] += 1
                    
                    # Cleanup old data
                    self.data_manager.cleanup_old_data()
                
                # Check if we should start a new generation job
                time_since_last = time.time() - self.stats['last_data_gen']
                if time_since_last < self.config.min_data_gen_interval:
                    time.sleep(1)
                    continue
                
                # Check resources
                if self.resource_monitor.should_throttle_datagen():
                    time.sleep(self.config.resource_check_interval)
                    continue
                
                
                # Start a new data generation job (non-blocking)
                output_file = self.data_manager.get_next_data_file()
                if self.process_manager.start_data_generation(output_file):
                    self.stats['last_data_gen'] = time.time()
                else:
                    # At max concurrent jobs, wait a bit
                    time.sleep(1)
                
            except Exception as e:
                print(f"✗ Error in data generation loop: {e}")
                time.sleep(10)
    
    def run(self) -> int:
        """Run the continuous training system"""
        print("="*70)
        print("  Continuous Poker Training System")
        print("="*70)
        print()
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Start API server
        if not self.process_manager.start_api_server():
            print("✗ Failed to start API server. Exiting.")
            return 1
        
        # Start data generation thread
        datagen_thread = threading.Thread(target=self._data_generation_loop, daemon=True)
        datagen_thread.start()
        
        print("\n✓ System initialized")
        print("  - API server running")
        print("  - Data generation thread started")
        print(f"  - Building initial data buffer ({self.config.initial_buffer_size} batches)...\n")
        
        # Wait for initial buffer to build up (pipelining)
        # This ensures training never has to wait for data
        last_reported = 0
        while self.running and self.data_manager.data_queue.qsize() < self.config.initial_buffer_size:
            time.sleep(2)
            current = self.data_manager.data_queue.qsize()
            if current > last_reported:
                print(f"  Buffer: {current}/{self.config.initial_buffer_size} batches ready...")
                last_reported = current
        
        if not self.running:
            return 0
        
        print(f"\n✓ Initial buffer ready ({self.data_manager.data_queue.qsize()} batches)")
        print("  Training loop starting - data generation continues in background\n")
        
        # Main training loop
        last_status_print = time.time()
        
        try:
            while self.running:
                # Check resources periodically
                self.resource_monitor.check_resources()
                
                # Print status every minute
                if time.time() - last_status_print > 60:
                    self._print_status()
                    last_status_print = time.time()
                
                # Check if we can train
                if not self.resource_monitor.can_start_training():
                    print("⏸️  Training paused due to high resource usage")
                    time.sleep(self.config.resource_check_interval)
                    continue
                
                # Get data file to train on
                data_file = self.data_manager.get_data_file(block=True, timeout=5.0)
                if data_file is None:
                    continue
                
                # Train model
                model_file = self.data_manager.get_next_model_file()
                if self.process_manager.train_model(data_file, model_file):
                    self.stats['training_cycles'] += 1
                    self.stats['last_training'] = time.time()
                    
                    # Mark data file as complete (safe to delete now)
                    self.data_manager.mark_file_complete(data_file)
                    
                    # Copy to main model location (only if new model was actually saved)
                    main_model = self.data_manager.config.model_dir / "poker_model.pt"
                    main_model.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    if model_file.exists():
                        shutil.copy(model_file, main_model)
                        print(f"✓ Updated main model: {main_model}")
                    else:
                        print(f"ℹ️  No model improvement - keeping existing best model")
                    
                    # Run evaluation if enabled and at the right interval
                    if (self.config.eval_enabled and 
                        self.stats['training_cycles'] % self.config.eval_interval == 0):
                        self.process_manager.evaluate_model(main_model)
                else:
                    print("⚠️  Training failed, will retry with next batch...")
                    # Even on failure, mark as complete to avoid keeping it forever
                    self.data_manager.mark_file_complete(data_file)
                
                # Small delay before next cycle
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
        finally:
            self.running = False
            self._print_status()
            print("\n🧹 Cleaning up...")
            self.process_manager.cleanup()
            print("✓ Shutdown complete\n")
        
        return 0


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Continuous poker model training system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Resource thresholds
    parser.add_argument('--cpu-threshold', type=float, default=0.80,
                       help='CPU threshold for throttling (0.0-1.0, default: 0.80)')
    parser.add_argument('--memory-threshold', type=float, default=0.85,
                       help='Memory threshold for throttling (0.0-1.0, default: 0.85)')
    
    # Data generation
    parser.add_argument('--batch-size', type=int, default=5000,
                       help='Rollouts per batch (default: 5000)')
    parser.add_argument('--num-players', type=int, default=3,
                       help='Number of players (default: 3)')
    parser.add_argument('--agent-type', type=str, default='mixed',
                       choices=['random', 'call', 'tight', 'aggressive', 'mixed'],
                       help='Agent type (default: mixed)')
    
    # Training
    parser.add_argument('--epochs-per-cycle', type=int, default=50,
                       help='Epochs per training cycle (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden layer dimension (default: 256)')
    
    # Adaptive training schedule
    parser.add_argument('--no-adaptive-schedule', action='store_true',
                       help='Disable adaptive training schedule (enabled by default)')
    parser.add_argument('--initial-data-fraction', type=float, default=0.15,
                       help='Initial fraction of data to train on (default: 0.15)')
    parser.add_argument('--data-growth-factor', type=float, default=1.5,
                       help='Factor to grow dataset by when plateauing (default: 1.5)')
    parser.add_argument('--plateau-patience', type=int, default=5,
                       help='Epochs without improvement before increasing data (default: 5)')
    parser.add_argument('--plateau-threshold', type=float, default=0.001,
                       help='Minimum improvement threshold (default: 0.001)')
    
    # Evaluation
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip evaluation (default: False)')
    parser.add_argument('--eval-interval', type=int, default=5,
                       help='Run evaluation every N training cycles (default: 5)')
    parser.add_argument('--eval-num-hands', type=int, default=100,
                       help='Number of hands to play in evaluation (default: 100)')
    
    # File management
    parser.add_argument('--max-data-files', type=int, default=10,
                       help='Max data files to keep (default: 10, 0 = keep all)')
    parser.add_argument('--data-dir', type=str, default=f'/tmp/pokersim/data_v{MODEL_VERSION}',
                       help=f'Directory for training data (default: /tmp/pokersim/data_v{MODEL_VERSION})')
    parser.add_argument('--no-accumulate', action='store_true',
                       help='Train on single batches instead of accumulating all data (default: False)')
    
    # Pipelining
    parser.add_argument('--max-queue-size', type=int, default=5,
                       help='Max batches to queue ahead (default: 5)')
    parser.add_argument('--initial-buffer-size', type=int, default=1,
                       help='Initial batches before training starts (default: 1)')
    
    # API server
    parser.add_argument('--api-port', type=int, default=8080,
                       help='API server port (default: 8080)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = SystemConfig(
        cpu_threshold=args.cpu_threshold,
        memory_threshold=args.memory_threshold,
        rollouts_per_batch=args.batch_size,
        num_players=args.num_players,
        agent_type=args.agent_type,
        epochs_per_cycle=args.epochs_per_cycle,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        adaptive_schedule=not args.no_adaptive_schedule,
        initial_data_fraction=args.initial_data_fraction,
        data_growth_factor=args.data_growth_factor,
        plateau_patience=args.plateau_patience,
        plateau_threshold=args.plateau_threshold,
        eval_enabled=not args.skip_eval,
        eval_interval=args.eval_interval,
        eval_num_hands=args.eval_num_hands,
        max_data_files=args.max_data_files,
        accumulate_data=not args.no_accumulate,
        initial_buffer_size=args.initial_buffer_size,
        api_port=args.api_port,
        api_url=f"http://localhost:{args.api_port}/simulate",
        data_dir=Path(args.data_dir)
    )
    
    # Check dependencies
    try:
        import requests
        import torch
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Install with: pip install requests torch")
        if not HAS_PSUTIL:
            print("  Optional: pip install psutil (for resource monitoring)")
        return 1
    
    # Run continuous trainer
    trainer = ContinuousTrainer(config)
    return trainer.run()


if __name__ == "__main__":
    sys.exit(main())

