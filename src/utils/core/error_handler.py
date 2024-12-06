"""Advanced error handling and recovery system"""

import logging
import time
import threading
from typing import Dict, Optional, Callable, Any
from enum import Enum
import traceback
from dataclasses import dataclass
from collections import deque
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = 1      # Minor issues, can continue
    MEDIUM = 2   # Significant issues, needs attention
    HIGH = 3     # Critical issues, needs immediate action
    FATAL = 4    # Unrecoverable, must halt

class ErrorCategory(Enum):
    """Categories of errors for better organization"""
    MEMORY = "memory"
    COMPUTATION = "computation"
    RESOURCE = "resource"
    MODEL = "model"
    DATA = "data"
    SYSTEM = "system"

@dataclass
class ErrorContext:
    """Context information for errors"""
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    stacktrace: str
    recovery_attempts: int = 0
    resolved: bool = False

class SystemHealthStatus:
    """System health monitoring"""
    def __init__(self):
        self.metrics = {
            'memory_pressure': 0.0,
            'gpu_health': 1.0,
            'error_rate': 0.0,
            'recovery_success_rate': 1.0
        }
        self.error_history = deque(maxlen=100)
        
    def update_metrics(self, error_context: ErrorContext):
        """Update health metrics based on new error"""
        self.error_history.append(error_context)
        
        # Update error rate
        recent_errors = len([e for e in self.error_history 
                           if time.time() - e.timestamp < 300])  # Last 5 minutes
        self.metrics['error_rate'] = recent_errors / 100
        
        # Update recovery success rate
        if self.error_history:
            resolved = len([e for e in self.error_history if e.resolved])
            self.metrics['recovery_success_rate'] = resolved / len(self.error_history)
        
        # Update GPU health if available
        if torch.cuda.is_available():
            try:
                memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                self.metrics['memory_pressure'] = memory_used
                self.metrics['gpu_health'] = 1.0 - (memory_used * 0.8)  # Degrade health as memory pressure increases
            except Exception:
                self.metrics['gpu_health'] = 0.0

class RecoveryStrategy:
    """Base class for recovery strategies"""
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
    
    def can_attempt_recovery(self, context: ErrorContext) -> bool:
        """Check if recovery can be attempted"""
        return context.recovery_attempts < self.max_attempts
    
    def execute(self, context: ErrorContext) -> bool:
        """Execute recovery strategy"""
        raise NotImplementedError

class MemoryRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for memory-related issues"""
    def execute(self, context: ErrorContext) -> bool:
        try:
            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Verify memory was freed
                if torch.cuda.memory_allocated() < torch.cuda.get_device_properties(0).total_memory * 0.8:
                    return True
            return False
        except Exception:
            return False

class ComputationRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for computation-related issues"""
    def execute(self, context: ErrorContext) -> bool:
        try:
            if torch.cuda.is_available():
                # Reset GPU device
                device = torch.device('cuda')
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                return True
            return False
        except Exception:
            return False

class ResourceRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for resource-related issues"""
    def execute(self, context: ErrorContext) -> bool:
        try:
            # Release and reallocate resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
                # Try to reinitialize CUDA
                device = torch.device('cuda')
                torch.cuda.reset_peak_memory_stats(device)
                
                # Verify GPU is responsive
                test_tensor = torch.zeros((1,), device=device)
                del test_tensor
                return True
            return False
        except Exception:
            return False

class ErrorHandler:
    """Central error handling system"""
    def __init__(self):
        self.health_status = SystemHealthStatus()
        self.recovery_strategies = {
            ErrorCategory.MEMORY: MemoryRecoveryStrategy(),
            ErrorCategory.COMPUTATION: ComputationRecoveryStrategy(),
            ErrorCategory.RESOURCE: ResourceRecoveryStrategy()
        }
        self._error_queue = deque(maxlen=1000)
        self._recovery_thread = None
        self._stop_recovery = False
        
    def start_recovery_worker(self):
        """Start background recovery worker"""
        self._stop_recovery = False
        self._recovery_thread = threading.Thread(target=self._recovery_worker)
        self._recovery_thread.daemon = True
        self._recovery_thread.start()
        
    def stop_recovery_worker(self):
        """Stop background recovery worker"""
        self._stop_recovery = True
        if self._recovery_thread:
            self._recovery_thread.join()
    
    def _recovery_worker(self):
        """Background worker for processing recovery attempts"""
        while not self._stop_recovery:
            if self._error_queue:
                error_context = self._error_queue.popleft()
                self._attempt_recovery(error_context)
            time.sleep(0.1)
    
    def _attempt_recovery(self, context: ErrorContext):
        """Attempt to recover from an error"""
        strategy = self.recovery_strategies.get(context.category)
        if strategy and strategy.can_attempt_recovery(context):
            context.recovery_attempts += 1
            success = strategy.execute(context)
            if success:
                context.resolved = True
                logger.info(f"Successfully recovered from {context.category.value} error")
            else:
                logger.warning(f"Recovery attempt {context.recovery_attempts} failed for {context.category.value} error")
                if context.recovery_attempts < strategy.max_attempts:
                    self._error_queue.append(context)
    
    def handle_error(self, error: Exception, category: ErrorCategory, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> None:
        """Handle an error with the appropriate strategy"""
        context = ErrorContext(
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=str(error),
            stacktrace=traceback.format_exc()
        )
        
        # Update health metrics
        self.health_status.update_metrics(context)
        
        # Log error
        logger.error(f"Error occurred - Category: {category.value}, Severity: {severity.name}")
        logger.error(f"Message: {context.message}")
        logger.error(f"Stacktrace: {context.stacktrace}")
        
        # Queue for recovery if appropriate
        if severity != ErrorSeverity.FATAL:
            self._error_queue.append(context)
        else:
            logger.critical("Fatal error occurred - immediate attention required")
    
    def get_health_metrics(self) -> Dict[str, float]:
        """Get current system health metrics"""
        return self.health_status.metrics

def with_error_handling(category: ErrorCategory, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator for automatic error handling"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = ErrorHandler()
                error_handler.handle_error(e, category, severity)
                if severity == ErrorSeverity.FATAL:
                    raise
                return None
        return wrapper
    return decorator
