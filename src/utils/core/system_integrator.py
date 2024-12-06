"""System integration and coordination module"""

import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque
import numpy as np
import torch
import torch.distributed as dist
from .error_handler import ErrorHandler, ErrorCategory, ErrorSeverity, with_error_handling
from .gpu_accelerator import GPUAccelerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingNode:
    """Represents a processing node in the distributed system"""
    node_id: str
    gpu_accelerator: GPUAccelerator
    is_master: bool = False
    status: str = "idle"
    current_load: float = 0.0
    tasks_processed: int = 0

class LoadBalancer:
    """Manages workload distribution across processing nodes"""
    def __init__(self):
        self.nodes: Dict[str, ProcessingNode] = {}
        self.error_handler = ErrorHandler()
        self.task_queue = deque()
        self._stop_balancing = False
        self._balance_thread = None
    
    def start_balancing(self):
        """Start the load balancing thread"""
        self._stop_balancing = False
        self._balance_thread = threading.Thread(target=self._balance_loop)
        self._balance_thread.daemon = True
        self._balance_thread.start()
    
    def stop_balancing(self):
        """Stop the load balancing thread"""
        self._stop_balancing = True
        if self._balance_thread:
            self._balance_thread.join()
    
    def _balance_loop(self):
        """Main load balancing loop"""
        while not self._stop_balancing:
            if self.task_queue:
                task = self.task_queue.popleft()
                node = self._select_optimal_node()
                if node:
                    self._assign_task(task, node)
            time.sleep(0.1)
    
    @with_error_handling(category=ErrorCategory.RESOURCE)
    def _select_optimal_node(self) -> Optional[ProcessingNode]:
        """Select the optimal node for task execution"""
        if not self.nodes:
            return None
            
        return min(
            self.nodes.values(),
            key=lambda n: (n.current_load, n.tasks_processed)
        )
    
    @with_error_handling(category=ErrorCategory.RESOURCE)
    def _assign_task(self, task: Dict, node: ProcessingNode) -> None:
        """Assign a task to a specific node"""
        try:
            node.status = "processing"
            node.current_load += task.get("estimated_load", 0.1)
            node.tasks_processed += 1
            
            # Allocate resources through GPU accelerator
            result = node.gpu_accelerator.allocate_resources(task)
            
            if result["status"] != "allocated":
                raise RuntimeError(f"Failed to allocate resources: {result['reason']}")
                
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.RESOURCE)
            node.status = "error"
        finally:
            node.current_load = max(0.0, node.current_load - task.get("estimated_load", 0.1))

class PipelineCoordinator:
    """Coordinates processing pipelines across nodes"""
    def __init__(self):
        self.pipelines: Dict[str, List[str]] = {}
        self.load_balancer = LoadBalancer()
        self.error_handler = ErrorHandler()
    
    @with_error_handling(category=ErrorCategory.SYSTEM)
    def create_pipeline(self, pipeline_id: str, stages: List[str]) -> None:
        """Create a new processing pipeline"""
        self.pipelines[pipeline_id] = stages
        logger.info(f"Created pipeline {pipeline_id} with {len(stages)} stages")
    
    @with_error_handling(category=ErrorCategory.SYSTEM)
    def execute_pipeline(self, pipeline_id: str, data: Any) -> Optional[Any]:
        """Execute a processing pipeline"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
            
        result = data
        for stage in self.pipelines[pipeline_id]:
            task = {
                "stage": stage,
                "data": result,
                "estimated_load": 0.2
            }
            self.load_balancer.task_queue.append(task)
            # Wait for stage completion
            while self.load_balancer.task_queue:
                time.sleep(0.1)
            
        return result

class DistributedManager:
    """Manages distributed processing setup and coordination"""
    def __init__(self, world_size: int = 1):
        self.world_size = world_size
        self.error_handler = ErrorHandler()
        self.nodes: Dict[str, ProcessingNode] = {}
        
    @with_error_handling(category=ErrorCategory.SYSTEM)
    def initialize_distributed(self) -> None:
        """Initialize distributed processing environment"""
        if not dist.is_available():
            raise RuntimeError("PyTorch distributed not available")
            
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    
    @with_error_handling(category=ErrorCategory.SYSTEM)
    def setup_node(self, node_id: str, is_master: bool = False) -> None:
        """Setup a new processing node"""
        accelerator = GPUAccelerator()
        node = ProcessingNode(
            node_id=node_id,
            gpu_accelerator=accelerator,
            is_master=is_master
        )
        self.nodes[node_id] = node
        logger.info(f"Set up node {node_id} (master: {is_master})")

class SystemMonitor:
    """Monitors overall system health and performance"""
    def __init__(self):
        self.metrics: Dict[str, deque] = {
            'system_load': deque(maxlen=100),
            'pipeline_latency': deque(maxlen=100),
            'error_rate': deque(maxlen=100),
            'node_health': deque(maxlen=100)
        }
        self.error_handler = ErrorHandler()
        self._monitor_thread = None
        self._stop_monitoring = False
    
    def start_monitoring(self):
        """Start system monitoring"""
        self._stop_monitoring = False
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self._stop_monitoring = True
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_monitoring:
            try:
                # Collect system metrics
                self.metrics['system_load'].append(self._get_system_load())
                self.metrics['node_health'].append(self._get_node_health())
                
                # Check error handler metrics
                error_metrics = self.error_handler.get_health_metrics()
                self.metrics['error_rate'].append(error_metrics['error_rate'])
                
                # Log critical issues
                if error_metrics['error_rate'] > 0.5:
                    logger.warning("High error rate detected")
                
            except Exception as e:
                self.error_handler.handle_error(e, ErrorCategory.SYSTEM)
            
            time.sleep(5)  # Update every 5 seconds
    
    @with_error_handling(category=ErrorCategory.SYSTEM)
    def _get_system_load(self) -> float:
        """Get current system load"""
        if torch.cuda.is_available():
            return torch.cuda.utilization() / 100.0
        return 0.0
    
    @with_error_handling(category=ErrorCategory.SYSTEM)
    def _get_node_health(self) -> float:
        """Get overall node health metric"""
        if torch.cuda.is_available():
            memory_pressure = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            return 1.0 - memory_pressure
        return 1.0
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        return {
            'system_load': float(np.mean(self.metrics['system_load'])) if self.metrics['system_load'] else 0,
            'error_rate': float(np.mean(self.metrics['error_rate'])) if self.metrics['error_rate'] else 0,
            'node_health': float(np.mean(self.metrics['node_health'])) if self.metrics['node_health'] else 1.0
        }

class SystemIntegrator:
    """Main system integration controller"""
    def __init__(self):
        self.distributed_manager = DistributedManager()
        self.pipeline_coordinator = PipelineCoordinator()
        self.system_monitor = SystemMonitor()
        self.error_handler = ErrorHandler()
        self.fault_tolerance_enabled = False
        
    def initialize_system(self, world_size: int = 1) -> None:
        """Initialize the entire system"""
        try:
            # Initialize distributed processing
            self.distributed_manager.initialize_distributed()
            
            # Setup nodes
            for i in range(world_size):
                self.distributed_manager.setup_node(
                    f"node_{i}",
                    is_master=(i == 0)
                )
            
            # Start monitoring and load balancing
            self.system_monitor.start_monitoring()
            self.pipeline_coordinator.load_balancer.start_balancing()
            
            logger.info("System initialization complete")
            
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.SYSTEM, ErrorSeverity.FATAL)
            raise
    
    def shutdown_system(self) -> None:
        """Shutdown the system gracefully"""
        try:
            self.system_monitor.stop_monitoring()
            self.pipeline_coordinator.load_balancer.stop_balancing()
            
            # Cleanup distributed resources
            if dist.is_initialized():
                dist.destroy_process_group()
                
            logger.info("System shutdown complete")
            
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.SYSTEM)
            raise
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            'metrics': self.system_monitor.get_system_metrics(),
            'health': self.error_handler.get_health_metrics(),
            'nodes': len(self.distributed_manager.nodes)
        }

    @with_error_handling(category=ErrorCategory.SYSTEM)
    def get_processing_capacity(self) -> float:
        """Get current processing capacity of the system"""
        total_capacity = 0.0
        
        if torch.cuda.is_available():
            # Calculate based on GPU utilization and memory
            for node in self.distributed_manager.nodes.values():
                metrics = node.gpu_accelerator.performance_monitor.get_current_metrics()
                available_capacity = 1.0 - (metrics['gpu_utilization'] / 100.0)
                total_capacity += available_capacity
                
        return total_capacity / max(1, len(self.distributed_manager.nodes))

    @with_error_handling(category=ErrorCategory.SYSTEM)
    def distribute_workload(self, workloads: List[Callable]) -> Dict[str, List[Callable]]:
        """Distribute workload across available nodes"""
        if not self.distributed_manager.nodes:
            raise RuntimeError("No processing nodes available")
            
        distribution: Dict[str, List[Callable]] = {
            node_id: [] for node_id in self.distributed_manager.nodes
        }
        
        # Simple round-robin distribution
        node_ids = list(self.distributed_manager.nodes.keys())
        for i, workload in enumerate(workloads):
            node_id = node_ids[i % len(node_ids)]
            distribution[node_id].append(workload)
            
        return distribution

    @with_error_handling(category=ErrorCategory.SYSTEM)
    def get_network_capacity(self) -> float:
        """Get current network capacity"""
        if not dist.is_initialized():
            return 0.0
            
        try:
            # Estimate network capacity through ping test
            if len(self.distributed_manager.nodes) > 1:
                start_time = time.time()
                test_tensor = torch.ones(1)
                if torch.cuda.is_available():
                    test_tensor = test_tensor.cuda()
                dist.broadcast(test_tensor, src=0)
                latency = time.time() - start_time
                
                # Convert latency to a capacity metric (0.0 to 1.0)
                # Lower latency = higher capacity
                return max(0.0, 1.0 - (latency / 0.1))  # Assuming 100ms is the threshold
            return 1.0  # Single node system
            
        except Exception:
            return 0.0

    @with_error_handling(category=ErrorCategory.SYSTEM)
    def enable_fault_tolerance(self) -> None:
        """Enable fault tolerance mechanisms"""
        self.fault_tolerance_enabled = True
        
        # Configure fault tolerance for each node
        for node in self.distributed_manager.nodes.values():
            # Set up error handling with recovery
            node.gpu_accelerator.error_handler.start_recovery_worker()
            
        logger.info("Fault tolerance enabled")

    @with_error_handling(category=ErrorCategory.SYSTEM)
    def measure_performance(self) -> Dict[str, float]:
        """Measure system performance metrics"""
        performance_metrics = {}
        
        # Collect metrics from all nodes
        for node_id, node in self.distributed_manager.nodes.items():
            metrics = node.gpu_accelerator.performance_monitor.get_current_metrics()
            performance_metrics[node_id] = {
                'processing_capacity': self.get_processing_capacity(),
                'network_capacity': self.get_network_capacity(),
                'gpu_utilization': metrics['gpu_utilization'],
                'memory_usage': metrics['memory_usage'],
                'operation_latency': metrics['operation_latency']
            }
            
        # Calculate system-wide averages
        if performance_metrics:
            avg_metrics = {
                'avg_processing_capacity': np.mean([m['processing_capacity'] for m in performance_metrics.values()]),
                'avg_network_capacity': np.mean([m['network_capacity'] for m in performance_metrics.values()]),
                'avg_gpu_utilization': np.mean([m['gpu_utilization'] for m in performance_metrics.values()]),
                'avg_memory_usage': np.mean([m['memory_usage'] for m in performance_metrics.values()]),
                'avg_operation_latency': np.mean([m['operation_latency'] for m in performance_metrics.values()])
            }
            performance_metrics['system_averages'] = avg_metrics
            
        return performance_metrics
