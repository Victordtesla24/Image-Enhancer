import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from collections import deque
import threading
import logging
from .error_handler import (
    ErrorHandler,
    ErrorCategory,
    ErrorSeverity,
    with_error_handling
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.metrics = {
            'gpu_utilization': deque(maxlen=history_size),
            'memory_usage': deque(maxlen=history_size),
            'operation_latency': deque(maxlen=history_size),
            'throughput': deque(maxlen=history_size)
        }
        self.start_time = time.time()
        self.operation_count = 0
        self._monitor_thread = None
        self._stop_monitoring = False
        
    def start_monitoring(self):
        """Start background monitoring thread"""
        self._stop_monitoring = False
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._stop_monitoring = True
        if self._monitor_thread:
            self._monitor_thread.join()
            logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while not self._stop_monitoring:
            try:
                if torch.cuda.is_available():
                    # Get GPU utilization
                    gpu_util = torch.cuda.utilization()
                    self.metrics['gpu_utilization'].append(gpu_util)
                    
                    # Get memory usage
                    memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
                    self.metrics['memory_usage'].append(memory_used)
                    
                    # Calculate throughput (operations per second)
                    elapsed_time = time.time() - self.start_time
                    throughput = self.operation_count / elapsed_time if elapsed_time > 0 else 0
                    self.metrics['throughput'].append(throughput)
                else:
                    # Fallback values for CPU
                    self.metrics['gpu_utilization'].append(0)
                    self.metrics['memory_usage'].append(0)
                    self.metrics['throughput'].append(0)
                
                # Initialize operation_latency if empty
                if not self.metrics['operation_latency']:
                    self.metrics['operation_latency'].append(0)
                    
            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                
            time.sleep(1)  # Update every second
    
    def record_operation_latency(self, latency: float):
        """Record the latency of an operation"""
        self.metrics['operation_latency'].append(latency)
        self.operation_count += 1
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return {
            'gpu_utilization': float(np.mean(self.metrics['gpu_utilization'])) if self.metrics['gpu_utilization'] else 0,
            'memory_usage': float(np.mean(self.metrics['memory_usage'])) if self.metrics['memory_usage'] else 0,
            'operation_latency': float(np.mean(self.metrics['operation_latency'])) if self.metrics['operation_latency'] else 0,
            'throughput': float(np.mean(self.metrics['throughput'])) if self.metrics['throughput'] else 0
        }

class DeviceManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.available_memory = self._get_available_memory()
        self.error_handler = ErrorHandler()
        self.error_handler.start_recovery_worker()
        
    def __del__(self):
        if hasattr(self, 'error_handler'):
            self.error_handler.stop_recovery_worker()
    
    @with_error_handling(category=ErrorCategory.MEMORY)
    def _get_available_memory(self) -> int:
        """Get available GPU memory in bytes."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return torch.cuda.get_device_properties(0).total_memory
        return 0
    
    def get_device(self) -> torch.device:
        """Return the current device."""
        return self.device
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return torch.cuda.is_available()

class MemoryManager:
    def __init__(self):
        self.allocated_memory: Dict[str, int] = {}
        self.device_manager = DeviceManager()
        self.performance_monitor = PerformanceMonitor()
        self.error_handler = ErrorHandler()
        self.error_handler.start_recovery_worker()
        
    def __del__(self):
        if hasattr(self, 'error_handler'):
            self.error_handler.stop_recovery_worker()
    
    @with_error_handling(category=ErrorCategory.MEMORY)
    def allocate(self, task_id: str, required_memory: int) -> bool:
        """
        Allocate memory for a task with adaptive sizing based on performance metrics.
        
        Args:
            task_id: Unique identifier for the task
            required_memory: Required memory in bytes
            
        Returns:
            bool: True if allocation successful, False otherwise
        """
        metrics = self.performance_monitor.get_current_metrics()
        memory_usage = metrics['memory_usage']
        
        # Adjust allocation based on current memory usage
        if memory_usage > 80:  # High memory pressure
            required_memory = int(required_memory * 0.8)  # Reduce allocation by 20%
        elif memory_usage < 30:  # Low memory pressure
            required_memory = int(required_memory * 1.2)  # Increase allocation by 20%
            
        available = self.device_manager._get_available_memory()
        if available >= required_memory:
            self.allocated_memory[task_id] = required_memory
            return True
        return False
    
    @with_error_handling(category=ErrorCategory.MEMORY)
    def release(self, task_id: str) -> None:
        """Release memory allocated for a task."""
        if task_id in self.allocated_memory:
            del self.allocated_memory[task_id]
            torch.cuda.empty_cache()

class ComputeScheduler:
    def __init__(self):
        self.active_tasks: Dict[str, Dict] = {}
        self.device_manager = DeviceManager()
        self.performance_monitor = PerformanceMonitor()
        self.error_handler = ErrorHandler()
        self.error_handler.start_recovery_worker()
        
    def __del__(self):
        if hasattr(self, 'error_handler'):
            self.error_handler.stop_recovery_worker()
    
    @with_error_handling(category=ErrorCategory.COMPUTATION)
    def schedule_task(self, task_id: str, priority: int = 0) -> None:
        """
        Schedule a compute task with adaptive priority based on performance metrics.
        
        Args:
            task_id: Unique identifier for the task
            priority: Task priority (higher number = higher priority)
        """
        metrics = self.performance_monitor.get_current_metrics()
        gpu_util = metrics['gpu_utilization']
        
        # Adjust priority based on GPU utilization
        if gpu_util > 90:  # High GPU load
            priority = max(0, priority - 1)  # Decrease priority
        elif gpu_util < 50:  # Low GPU load
            priority += 1  # Increase priority
            
        self.active_tasks[task_id] = {
            "priority": priority,
            "status": "scheduled",
            "device": self.device_manager.get_device()
        }
    
    def get_next_task(self) -> Optional[str]:
        """Get the next task to execute based on priority."""
        if not self.active_tasks:
            return None
        
        return max(self.active_tasks.items(), 
                  key=lambda x: x[1]["priority"])[0]

class GPUAccelerator:
    def __init__(self):
        self.device_manager = DeviceManager()
        self.memory_manager = MemoryManager()
        self.compute_scheduler = ComputeScheduler()
        self.performance_monitor = PerformanceMonitor()
        self.error_handler = ErrorHandler()
        
        # Start monitoring if GPU is available
        if self.device_manager.is_gpu_available():
            self.performance_monitor.start_monitoring()
            self.error_handler.start_recovery_worker()
    
    def __del__(self):
        """Cleanup when accelerator is destroyed"""
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.stop_monitoring()
        if hasattr(self, 'error_handler'):
            self.error_handler.stop_recovery_worker()
    
    @with_error_handling(category=ErrorCategory.RESOURCE)
    def allocate_resources(self, task_requirements: Dict) -> Dict:
        """
        Allocate GPU resources for a task with performance monitoring and error handling.
        
        Args:
            task_requirements: Dictionary containing memory and compute requirements
            
        Returns:
            Dict containing allocated resources information
        """
        start_time = time.time()
        task_id = task_requirements.get("task_id", str(id(task_requirements)))
        required_memory = task_requirements.get("memory", 1024 * 1024 * 1024)  # Default 1GB
        
        # Get current performance metrics
        metrics = self.performance_monitor.get_current_metrics()
        
        # Check system health
        health_metrics = self.error_handler.get_health_metrics()
        if health_metrics['gpu_health'] < 0.5:
            logger.warning("GPU health is poor, attempting recovery...")
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device_manager.get_device())
            except Exception as e:
                self.error_handler.handle_error(e, ErrorCategory.RESOURCE, ErrorSeverity.HIGH)
                return {"status": "failed", "reason": "gpu_health_critical"}
        
        # Adjust resource allocation based on metrics
        if metrics['gpu_utilization'] > 80:
            required_memory = int(required_memory * 0.8)  # Reduce memory requirement under high load
        
        try:
            if self.memory_manager.allocate(task_id, required_memory):
                self.compute_scheduler.schedule_task(task_id, task_requirements.get("priority", 0))
                
                # Record operation latency
                self.performance_monitor.record_operation_latency(time.time() - start_time)
                
                return {
                    "task_id": task_id,
                    "device": self.device_manager.get_device(),
                    "allocated_memory": required_memory,
                    "status": "allocated",
                    "performance_metrics": metrics,
                    "health_metrics": health_metrics
                }
            return {"status": "failed", "reason": "insufficient_resources"}
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.RESOURCE, ErrorSeverity.HIGH)
            return {"status": "failed", "reason": str(e)}
    
    @with_error_handling(category=ErrorCategory.COMPUTATION)
    def optimize_compute(self, operations: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Optimize compute operations for GPU execution with error handling.
        
        Args:
            operations: List of tensor operations to optimize
            
        Returns:
            Optimized tensor operations
        """
        start_time = time.time()
        device = self.device_manager.get_device()
        optimized_ops = []
        
        try:
            for op in operations:
                # Move operation to GPU if available
                op = op.to(device)
                # Enable autograd for training operations
                op.requires_grad_(True)
                optimized_ops.append(op)
            
            # Record operation latency
            self.performance_monitor.record_operation_latency(time.time() - start_time)
                
            return optimized_ops
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.COMPUTATION, ErrorSeverity.MEDIUM)
            # Fallback to CPU if GPU optimization fails
            return [op.cpu() for op in operations]
    
    @with_error_handling(category=ErrorCategory.MEMORY)
    def manage_memory(self, data_size: int) -> Tuple[bool, str]:
        """
        Manage GPU memory for data processing with error handling.
        
        Args:
            data_size: Size of data in bytes
            
        Returns:
            Tuple of (success status, message)
        """
        start_time = time.time()
        
        if not self.device_manager.is_gpu_available():
            return False, "GPU not available"
            
        try:
            metrics = self.performance_monitor.get_current_metrics()
            available_memory = self.device_manager._get_available_memory()
            
            # Check system health
            health_metrics = self.error_handler.get_health_metrics()
            if health_metrics['memory_pressure'] > 0.9:
                logger.warning("Critical memory pressure detected")
                torch.cuda.empty_cache()
                available_memory = self.device_manager._get_available_memory()
            
            # Adjust memory management based on metrics
            if metrics['memory_usage'] > 80:
                torch.cuda.empty_cache()
                available_memory = self.device_manager._get_available_memory()
                
            success = available_memory >= data_size
            
            # Record operation latency
            self.performance_monitor.record_operation_latency(time.time() - start_time)
            
            if success:
                return True, "Memory allocated successfully"
            return False, "Insufficient memory"
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.MEMORY, ErrorSeverity.HIGH)
            return False, str(e)
    
    @with_error_handling(category=ErrorCategory.RESOURCE)
    def release_resources(self, task_id: str) -> None:
        """Release allocated GPU resources with error handling."""
        start_time = time.time()
        
        try:
            self.memory_manager.release(task_id)
            if task_id in self.compute_scheduler.active_tasks:
                del self.compute_scheduler.active_tasks[task_id]
                
            # Record operation latency
            self.performance_monitor.record_operation_latency(time.time() - start_time)
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.RESOURCE, ErrorSeverity.MEDIUM)
