"""System integration and coordination module"""

import logging
import time
from typing import Any, Dict, List, Optional

import psutil
import torch

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitors system resources and performance."""

    def __init__(self, sampling_interval: float = 1.0):
        """Initialize system monitor.
        
        Args:
            sampling_interval: Sampling interval in seconds
        """
        self.sampling_interval = sampling_interval
        self.metrics_history = []
        self.last_update = 0
        self.update_interval = 1.0  # seconds
        self.initialized = False
        self.start_time = time.time()

    def initialize(self):
        """Initialize monitoring system."""
        if self.initialized:
            return
            
        self.metrics_history = []
        self.last_update = time.time()
        self.start_time = time.time()
        self.initialized = True

    def get_system_metrics(self) -> dict:
        """Get current system metrics.
        
        Returns:
            Dictionary of system metrics
        """
        if not self.initialized:
            self.initialize()
            
        import psutil
        metrics = {
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "used": psutil.virtual_memory().used,
                "percent": psutil.virtual_memory().percent,
            },
            "cpu": {
                "usage_percent": psutil.cpu_percent(),
                "count": psutil.cpu_count(),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                "load_avg": psutil.getloadavg(),
            },
            "disk": {
                "total": psutil.disk_usage("/").total,
                "used": psutil.disk_usage("/").used,
                "free": psutil.disk_usage("/").free,
                "percent": psutil.disk_usage("/").percent,
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv,
                "packets_sent": psutil.net_io_counters().packets_sent,
                "packets_recv": psutil.net_io_counters().packets_recv,
            },
            "timestamp": time.time(),
            "uptime": time.time() - psutil.boot_time(),
        }

        # Add GPU metrics if available
        try:
            import torch
            if torch.cuda.is_available():
                metrics["gpu"] = {
                    "count": torch.cuda.device_count(),
                    "devices": [{
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated": torch.cuda.memory_allocated(i),
                        "memory_reserved": torch.cuda.memory_reserved(i),
                    } for i in range(torch.cuda.device_count())]
                }
        except:
            pass

        # Always update history
        self.metrics_history.append({
            "timestamp": time.time(),
            "metrics": metrics.copy()
        })

        return metrics

    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get metrics history.
        
        Args:
            limit: Optional limit on number of records
            
        Returns:
            List of metric dictionaries
        """
        if limit:
            return self.metrics_history[-limit:]
        return self.metrics_history

    def get_resource_utilization(self) -> dict:
        """Get resource utilization trends.
        
        Returns:
            Dictionary of resource utilization trends
        """
        metrics = self.get_system_metrics()
        
        utilization = {
            "cpu_usage": metrics["cpu"]["usage_percent"],
            "memory_usage": metrics["memory"]["percent"],
            "disk_usage": metrics["disk"]["percent"],
            "uptime": time.time() - self.start_time,
        }
        
        if "gpu" in metrics:
            utilization["gpu_usage"] = [
                device["memory_allocated"] / device["memory_reserved"]
                for device in metrics["gpu"]["devices"]
                if device["memory_reserved"] > 0
            ]
            
        return utilization


class ResourceManager:
    """Manages system resources for processing."""

    def __init__(self, memory_limit: float = 0.8, cpu_limit: float = 0.9):
        """Initialize resource manager.
        
        Args:
            memory_limit: Maximum memory usage fraction (0-1)
            cpu_limit: Maximum CPU usage fraction (0-1)
        """
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.allocated_resources = {}
        self.resource_limits = {}
        self.initialized = False
        self.monitor = SystemMonitor()

    def initialize(self):
        """Initialize resource management system."""
        if self.initialized:
            return
        
        self.monitor.initialize()
        self.resource_limits = {
            'memory': self._get_available_memory(),
            'gpu_memory': self._get_available_gpu_memory(),
            'cpu_cores': self._get_available_cpu_cores()
        }
        self.initialized = True

    def check_resource_availability(self, required_memory: int = 0) -> bool:
        """Check if required resources are available.
        
        Args:
            required_memory: Required memory in bytes
            
        Returns:
            True if resources are available
        """
        if not self.initialized:
            self.initialize()
            
        available_memory = self._get_available_memory()
        if required_memory > available_memory * self.memory_limit:
            return False
            
        import psutil
        if psutil.cpu_percent() > self.cpu_limit * 100:
            return False
            
        return True

    def allocate_resources(self, task_id: str, memory: int = 0) -> bool:
        """Allocate resources for a task.
        
        Args:
            task_id: Task identifier
            memory: Required memory in bytes
            
        Returns:
            True if allocation successful
        """
        if not self.check_resource_availability(memory):
            return False
            
        self.allocated_resources[task_id] = {
            'memory': memory,
            'timestamp': time.time()
        }
        return True

    def release_resources(self, task_id: str):
        """Release resources allocated to a task.
        
        Args:
            task_id: Task identifier
        """
        self.allocated_resources.pop(task_id, None)

    def get_allocation_status(self) -> dict:
        """Get current resource allocation status.
        
        Returns:
            Dictionary containing allocation status
        """
        if not self.initialized:
            self.initialize()
            
        import psutil
        return {
            'allocated_resources': self.allocated_resources,
            'available_memory': self._get_available_memory(),
            'memory_usage_percent': psutil.virtual_memory().percent,
            'cpu_usage_percent': psutil.cpu_percent(),
            'gpu_memory_available': self._get_available_gpu_memory()
        }

    def _get_available_memory(self) -> int:
        """Get available system memory in bytes."""
        import psutil
        return psutil.virtual_memory().available

    def _get_available_gpu_memory(self) -> int:
        """Get available GPU memory in bytes."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory
        except:
            pass
        return 0

    def _get_available_cpu_cores(self) -> int:
        """Get number of available CPU cores."""
        import multiprocessing
        return multiprocessing.cpu_count()


class SystemIntegrator:
    """Integrates system components and manages resources."""

    def __init__(self):
        """Initialize system integrator."""
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.components = {}
        self.monitor = SystemMonitor()
        self.resource_manager = ResourceManager()

    def register_component(self, name: str, component: object):
        """Register a system component.

        Args:
            name: Component name
            component: Component instance
        """
        self.components[name] = component
        self.logger.info(f"Registered component: {name}")

    def get_component(self, name: str) -> Optional[object]:
        """Get a registered component.

        Args:
            name: Component name

        Returns:
            Component instance or None
        """
        return self.components.get(name)

    def remove_component(self, name: str):
        """Remove a registered component.

        Args:
            name: Component name
        """
        if name in self.components:
            del self.components[name]
            self.logger.info(f"Removed component: {name}")

    def get_system_info(self) -> Dict:
        """Get comprehensive system information.

        Returns:
            System information dictionary
        """
        info = {
            "device": str(self.device),
            "components": list(self.components.keys()),
            "system_metrics": self.monitor.get_system_metrics(),
            "resource_utilization": self.monitor.get_resource_utilization(),
            "allocation_status": self.resource_manager.get_allocation_status(),
        }

        return info

    def check_system_health(self) -> Dict:
        """Check system health status.

        Returns:
            System health information
        """
        metrics = self.monitor.get_system_metrics()

        # Define health thresholds
        thresholds = {
            "cpu_warning": 80,
            "cpu_critical": 90,
            "memory_warning": 80,
            "memory_critical": 90,
            "disk_warning": 80,
            "disk_critical": 90,
        }

        health_status = {"status": "healthy", "warnings": [], "metrics": metrics}

        # Check CPU usage
        cpu_usage = metrics["cpu"]["usage_percent"]
        if cpu_usage > thresholds["cpu_critical"]:
            health_status["status"] = "critical"
            health_status["warnings"].append(f"Critical CPU usage: {cpu_usage}%")
        elif cpu_usage > thresholds["cpu_warning"]:
            health_status["status"] = "warning"
            health_status["warnings"].append(f"High CPU usage: {cpu_usage}%")

        # Check memory usage
        memory_usage = metrics["memory"]["percent"]
        if memory_usage > thresholds["memory_critical"]:
            health_status["status"] = "critical"
            health_status["warnings"].append(f"Critical memory usage: {memory_usage}%")
        elif memory_usage > thresholds["memory_warning"]:
            health_status["status"] = "warning"
            health_status["warnings"].append(f"High memory usage: {memory_usage}%")

        # Check disk usage
        disk_usage = metrics["disk"]["percent"]
        if disk_usage > thresholds["disk_critical"]:
            health_status["status"] = "critical"
            health_status["warnings"].append(f"Critical disk usage: {disk_usage}%")
        elif disk_usage > thresholds["disk_warning"]:
            health_status["status"] = "warning"
            health_status["warnings"].append(f"High disk usage: {disk_usage}%")

        return health_status

    def cleanup(self):
        """Clean up system resources."""
        for name, component in self.components.items():
            if hasattr(component, "cleanup"):
                component.cleanup()
            self.logger.info(f"Cleaned up component: {name}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def analyze_system_performance(self):
        """Analyze system performance metrics."""
        try:
            metrics = self._collect_performance_metrics()
            self._process_metrics(metrics)
            # Remove unused variable
            self._update_performance_stats()
        except Exception as e:
            self.logger.error(f"Error analyzing system performance: {e}")
