"""Image processing core module."""

import copy
import logging
import time
import yaml
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .error_handler import ErrorHandler
from .gpu_accelerator import GPUAccelerator
from .system_integrator import ResourceManager, SystemMonitor
from ..quality_management.quality_manager import QualityManager

logger = logging.getLogger(__name__)


class DatasetHandler(Dataset):
    """Handles dataset operations with chunked loading."""

    def __init__(self, file_paths: List[str], chunk_size: int = 4):
        """Initialize dataset handler.

        Args:
            file_paths: List of file paths to process
            chunk_size: Number of items to load at once
        """
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.current_chunk: Dict[int, torch.Tensor] = {}
        self.current_chunk_start = -1

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of items in dataset
        """
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get dataset item.

        Args:
            idx: Item index

        Returns:
            Tensor for the item
        """
        chunk_start = (idx // self.chunk_size) * self.chunk_size
        if chunk_start != self.current_chunk_start:
            self._load_chunk(chunk_start)
        return self.current_chunk[idx]

    def _load_chunk(self, start_idx: int) -> None:
        """Load a chunk of data.

        Args:
            start_idx: Starting index of chunk
        """
        self.current_chunk = {}
        self.current_chunk_start = start_idx
        end_idx = min(start_idx + self.chunk_size, len(self.file_paths))
        
        for idx in range(start_idx, end_idx):
            data = np.load(self.file_paths[idx])
            self.current_chunk[idx] = torch.from_numpy(data)


class SessionManager:
    """Manages processing sessions."""

    def __init__(self, cleanup_interval: int = 3600):
        """Initialize session manager.
        
        Args:
            cleanup_interval: Interval in seconds for session cleanup
        """
        self.sessions = {}
        self.initialized = False
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()

    def initialize(self):
        """Initialize session manager."""
        if self.initialized:
            return
            
        self.sessions = {}
        self.initialized = True

    def create_session(self, session_id: str, config: Dict[str, Any]) -> bool:
        """Create a new session.
        
        Args:
            session_id: Session identifier
            config: Session configuration
            
        Returns:
            True if session created successfully
        """
        if session_id in self.sessions:
            return False
            
        self.sessions[session_id] = {
            'config': copy.deepcopy(config),
            'start_time': time.time(),
            'processed_items': 0,
            'errors': [],
            'metrics': [],
            'status': 'active',
            'last_update': time.time(),
        }
        return True

    def update_session(self, session_id: str, processed_items: Optional[int] = None) -> None:
        """Update session progress.
        
        Args:
            session_id: Session identifier
            processed_items: Optional number of processed items
        """
        if session_id not in self.sessions:
            return
            
        if processed_items is not None:
            self.sessions[session_id]['processed_items'] += processed_items
        self.sessions[session_id]['last_update'] = time.time()

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session information dictionary or None if session doesn't exist or is inactive
        """
        session = self.sessions.get(session_id)
        if session is None:
            return None
            
        # Return None for inactive sessions
        if session.get('status') != 'active':
            return None
            
        return session.copy()  # Return a copy to prevent modification

    def end_session(self, session_id: str) -> None:
        """End a session.
        
        Args:
            session_id: Session identifier
        """
        session = self.sessions.get(session_id)
        if session:
            session['status'] = 'inactive'
            session['end_time'] = time.time()
            self.cleanup_sessions()  # Try cleanup immediately

    def cleanup_sessions(self) -> None:
        """Clean up old sessions."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
            
        to_remove = []
        for session_id, session in self.sessions.items():
            # Remove inactive sessions that haven't been updated
            if (session.get('status') != 'active' and 
                current_time - session.get('last_update', 0) > self.cleanup_interval):
                to_remove.append(session_id)
                
        for session_id in to_remove:
            del self.sessions[session_id]
            
        self.last_cleanup = current_time


class ConfigSchema:
    """Configuration schema validator."""

    @staticmethod
    def validate_type(value: Any, expected_type: type, field_path: str) -> None:
        """Validate type of a configuration value.

        Args:
            value: Value to validate
            expected_type: Expected type
            field_path: Path to field in config for error messages

        Raises:
            TypeError: If value is not of expected type
        """
        if not isinstance(value, expected_type):
            msg = (
                f"Invalid type for {field_path}: "
                f"expected {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
            raise TypeError(msg)


class Processor:
    """Core processor for image enhancement."""

    def __init__(self):
        """Initialize processor."""
        self.resource_manager = ResourceManager()
        self.system_monitor = SystemMonitor()
        self.error_handler = ErrorHandler()
        self.gpu_accelerator = GPUAccelerator()
        self.quality_manager = QualityManager()
        self.session_manager = SessionManager()
        self.config = {}
        self.initialized = False

    def initialize(self):
        """Initialize processor components."""
        if self.initialized:
            return

        # Initialize components
        self.resource_manager.initialize()
        self.system_monitor.initialize()
        self.gpu_accelerator.initialize()
        self.quality_manager.initialize()
        self.session_manager.initialize()

        # Load default configuration
        self.config = {
            'batch_size': self._calculate_optimal_batch_size(),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'precision': 'float32',
            'memory_limit': 0.8,
            'cpu_limit': 0.9,
        }

        self.initialized = True

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available resources.
        
        Returns:
            Optimal batch size
        """
        # Get available memory
        available_memory = self.resource_manager._get_available_memory()
        
        # Estimate memory per image (assuming 1024x1024 RGB float32)
        mem_per_image = 1024 * 1024 * 3 * 4  # Height * Width * Channels * Bytes
        
        # Use 70% of available memory
        usable_memory = available_memory * 0.7
        
        # Calculate batch size
        optimal_size = max(1, int(usable_memory / (mem_per_image * 2)))
        return min(optimal_size, 32)  # Cap at 32 to prevent excessive memory usage

    def process_dataset(self, file_paths: List[str], session_id: str) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Process a dataset of files.
        
        Args:
            file_paths: List of file paths to process
            session_id: Session identifier
            
        Returns:
            Tuple of (processed tensors, processing info)
        """
        if not self.initialized:
            self.initialize()

        try:
            # Create session
            if not self.session_manager.create_session(session_id, self.config):
                raise ValueError(f"Session {session_id} already exists")
            
            # Initialize dataset
            dataset = DatasetHandler(file_paths, chunk_size=self._calculate_optimal_batch_size())
            dataloader = DataLoader(dataset, batch_size=self._calculate_optimal_batch_size(), shuffle=False)
            
            results = []
            start_time = time.time()
            error = None
            
            # Get initial system metrics
            system_metrics = {
                'initial': self.system_monitor.get_system_metrics()
            }
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Check resource availability
                    if not self.resource_manager.check_resource_availability():
                        logger.warning("Resource limits reached, waiting...")
                        time.sleep(1)
                        continue
                    
                    # Process batch
                    processed = self._process_batch(batch)
                    results.extend(processed)
                    
                    # Update session
                    self.session_manager.update_session(session_id, len(processed))
                    
                    # Update system metrics every 10 batches
                    if batch_idx % 10 == 0:
                        system_metrics[f'batch_{batch_idx}'] = self.system_monitor.get_system_metrics()
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    error = str(e)
                    self.error_handler.handle_error(e, {
                        'session_id': session_id,
                        'batch_size': len(batch),
                        'current_results': len(results)
                    })
                    break  # Stop processing on error
            
            duration = time.time() - start_time
            
            # Get final system metrics
            system_metrics['final'] = self.system_monitor.get_system_metrics()
            
            return results, {
                'total_items': len(file_paths),
                'processed_items': len(results),
                'duration': duration,
                'errors': len(self.error_handler.error_history),
                'system_metrics': system_metrics,
                'error': error
            }
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            self.error_handler.handle_error(e, {
                'session_id': session_id,
                'total_files': len(file_paths)
            })
            
            # Get final system metrics even in error case
            system_metrics = {
                'error': self.system_monitor.get_system_metrics()
            }
            
            return [], {
                'total_items': len(file_paths),
                'processed_items': 0,
                'duration': 0,
                'errors': 1,
                'error': str(e),
                'system_metrics': system_metrics
            }

    def _process_batch(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """Process a batch of tensors.
        
        Args:
            batch: Input batch tensor
            
        Returns:
            List of processed tensors
        """
        try:
            # Move to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch = batch.to(device)
            
            # Apply processing
            processed = self._apply_processing(batch)
            
            # Move back to CPU
            processed = [p.cpu() for p in processed]
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            self.error_handler.handle_error(e, {
                'batch_size': len(batch),
                'device': str(device)
            })
            raise  # Re-raise the exception to be caught by process_dataset

    def _apply_processing(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """Apply processing to a batch of tensors.

        Args:
            batch: Input batch tensor

        Returns:
            List of processed tensors
        """
        # Placeholder for processing implementation
        return [batch]

    def get_processing_history(self) -> List[Dict[str, Any]]:
        """Get processing history.

        Returns:
            List of processing history entries
        """
        return copy.deepcopy(self.processing_history)

    def clear_history(self) -> None:
        """Clear processing history."""
        self.processing_history = []

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration.

        Args:
            config: New configuration
        """
        validated = self._validate_config(config)
        self.config = validated

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration.

        Args:
            config: Configuration to validate

        Returns:
            Validated configuration

        Raises:
            ValueError: If config is invalid
            TypeError: If config has invalid types
        """
        # Create deep copy to avoid modifying input
        config = copy.deepcopy(config)
        validated = {
            "enhancement": {},
            "quality": {"target_metrics": {}},
            "preferences": {},
        }

        # Check required sections
        required_sections = ["enhancement", "quality", "preferences"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        # Validate enhancement config
        valid_models = ["super_resolution", "detail", "color"]

        # Check for invalid models first
        for model in config["enhancement"]:
            if model not in valid_models:
                msg = f"Invalid enhancement model: must be one of {valid_models}"
                raise ValueError(msg)

        # Then validate each required model
        for model in valid_models:
            if model not in config["enhancement"]:
                raise ValueError(f"Missing enhancement model: {model}")

            model_config = config["enhancement"][model]
            ConfigSchema.validate_type(model_config, dict, f"enhancement.{model}")

            if "enabled" not in model_config:
                raise ValueError(f"Missing 'enabled' in enhancement.{model}")

            ConfigSchema.validate_type(
                model_config["enabled"], bool, f"enhancement.{model}.enabled"
            )

            if model_config["enabled"]:
                if "strength" not in model_config:
                    msg = f"Missing 'strength' in enabled enhancement.{model}"
                    raise ValueError(msg)

                strength = model_config["strength"]
                if not isinstance(strength, (int, float)):
                    msg = (
                        f"Invalid type for enhancement.{model}.strength: "
                        f"expected numeric, got {type(strength).__name__}"
                    )
                    raise TypeError(msg)

                if strength < 0 or strength > 1:
                    msg = (
                        f"Invalid value for enhancement.{model}.strength: "
                        "must be between 0 and 1"
                    )
                    raise ValueError(msg)

            validated["enhancement"][model] = model_config

        # Validate quality config
        if "target_metrics" not in config["quality"]:
            raise ValueError("Missing target_metrics in quality config")

        target_metrics = config["quality"]["target_metrics"]
        ConfigSchema.validate_type(target_metrics, dict, "quality.target_metrics")

        valid_metrics = [
            "sharpness",
            "color_accuracy",
            "detail_preservation",
            "noise_level",
        ]
        for metric in target_metrics:
            if metric not in valid_metrics:
                msg = f"Invalid quality metric: must be one of {valid_metrics}"
                raise ValueError(msg)

            value = target_metrics[metric]
            if not isinstance(value, (int, float)):
                msg = (
                    f"Invalid type for quality.target_metrics.{metric}: "
                    f"expected numeric, got {type(value).__name__}"
                )
                raise TypeError(msg)

            if value < 0 or value > 1:
                msg = (
                    f"Invalid value for quality.target_metrics.{metric}: "
                    "must be between 0 and 1"
                )
                raise ValueError(msg)

        # Check for missing metrics
        for metric in valid_metrics:
            if metric not in target_metrics:
                raise ValueError(f"Missing required metric: {metric}")

        validated["quality"]["target_metrics"] = target_metrics

        # Validate preferences
        preferences = config.get("preferences", {})
        ConfigSchema.validate_type(preferences, dict, "preferences")

        required_prefs = {
            "auto_enhance": bool,
            "save_history": bool,
            "learning_enabled": bool,
            "feedback_frequency": str,
        }

        valid_frequencies = ["always", "low_quality", "never"]

        for pref, pref_type in required_prefs.items():
            if pref not in preferences:
                preferences[pref] = self._get_default_config()["preferences"][pref]
                continue

            value = preferences[pref]
            ConfigSchema.validate_type(value, pref_type, f"preferences.{pref}")

            if pref == "feedback_frequency" and value not in valid_frequencies:
                msg = (
                    f"Invalid feedback frequency: "
                    f"must be one of {valid_frequencies}"
                )
                raise ValueError(msg)

        validated["preferences"] = {
            "auto_enhance": preferences["auto_enhance"],
            "save_history": preferences["save_history"],
            "learning_enabled": preferences["learning_enabled"],
            "feedback_frequency": preferences["feedback_frequency"],
        }

        return validated

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "enhancement": {
                "super_resolution": {
                    "enabled": True,
                    "strength": 0.5,
                },
                "detail": {
                    "enabled": True,
                    "strength": 0.5,
                },
                "color": {
                    "enabled": True,
                    "strength": 0.5,
                },
            },
            "quality": {
                "target_metrics": {
                    "sharpness": 0.8,
                    "color_accuracy": 0.8,
                    "detail_preservation": 0.8,
                    "noise_level": 0.2,
                },
            },
            "preferences": {
                "auto_enhance": True,
                "save_history": True,
                "learning_enabled": True,
                "feedback_frequency": "low_quality",
            },
        }
