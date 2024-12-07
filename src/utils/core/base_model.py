"""Base AI model class definition"""

import logging

logger = logging.getLogger(__name__)


class AIModel:
    """Base class for all enhancement models."""

    def __init__(self, name: str, description: str = "", model_params: dict = None):
        """Initialize the base model with parameters.

        Args:
            name: Model name
            description: Model description
            model_params: Optional model parameters
        """
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.description = description
        self.model_params = model_params or {}
        self._validate_params()

    def _validate_params(self):
        """Validate model parameters."""
        required_params = self._get_required_params()
        for param in required_params:
            if param not in self.model_params:
                raise ValueError(f"Missing required parameter: {param}")

    def _get_required_params(self):
        """Get list of required parameters.

        Returns:
            List of parameter names
        """
        return []

    def process(self, input_data):
        """Process input data.

        Args:
            input_data: Input data dictionary

        Returns:
            Processed data dictionary
        """
        raise NotImplementedError("Process method must be implemented")

    def update_parameters(self, parameters: dict):
        """Update model parameters.

        Args:
            parameters: New parameter values
        """
        self.model_params.update(parameters)
        self._validate_params()
