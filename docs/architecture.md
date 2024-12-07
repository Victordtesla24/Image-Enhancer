# Architecture Documentation

## Core Components

### Processor Package
The processor functionality has been split into multiple modules for better maintainability:

- `base.py`: Contains the `BaseProcessor` class with core initialization and cleanup functionality
- `batch.py`: Contains the `BatchProcessor` class for handling batch processing operations
- `__init__.py`: Combines functionality into the main `Processor` class

This modular approach provides several benefits:
- Better code organization
- Improved maintainability
- Memory efficiency
- Easier testing
- Clear separation of concerns

### Component Relationships
mermaid
graph TD
    A[Processor] --> B[BatchProcessor]
    B --> C[BaseProcessor]
    C --> D[ModelManager]
    C --> E[DeviceManager]
    C --> F[SessionManager]
    C --> G[ErrorHandler]
```

## Memory Management
- Each module is kept under 300 lines for efficient memory usage
- Batch processing is done incrementally to manage memory
- Resources are cleaned up after each operation

## Error Handling
- Centralized error handling through ErrorHandler
- Consistent error reporting across components
- Proper cleanup in error cases

## Testing
- Each module has its own test file
- Mock objects used for dependencies
- Resource cleanup verified
- Edge cases covered

## Configuration
Default configuration includes:
- Batch size: 4
- Max memory usage: 80%
- Max CPU usage: 90%
- Operation timeout: 30s
- Retry attempts: 3
- Default log level: INFO
```