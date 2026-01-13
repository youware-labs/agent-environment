"""Environment abstractions for file operations and shell execution.

This module provides Protocol-based interfaces and implementations for
environment operations, allowing different backends (local, remote, S3, SSH, etc.)
to be used interchangeably.
"""

from agent_environment.base import (
    DEFAULT_INSTRUCTIONS_MAX_DEPTH,
    DEFAULT_INSTRUCTIONS_SKIP_DIRS,
    BaseResource,
    Environment,
    FileOperator,
    FileStat,
    InstructableResource,
    LocalTmpFileOperator,
    Resource,
    ResourceEntry,
    ResourceFactory,
    ResourceRegistry,
    ResourceRegistryState,
    ResumableResource,
    Shell,
    TmpFileOperator,
    TruncatedResult,
    generate_filetree,
)
from agent_environment.exceptions import (
    EnvironmentError as EnvironmentError,
)
from agent_environment.exceptions import (
    EnvironmentNotEnteredError,
    FileOperationError,
    PathNotAllowedError,
    ShellExecutionError,
    ShellTimeoutError,
)

__all__ = [
    "DEFAULT_INSTRUCTIONS_MAX_DEPTH",
    "DEFAULT_INSTRUCTIONS_SKIP_DIRS",
    "BaseResource",
    "Environment",
    "EnvironmentError",
    "EnvironmentNotEnteredError",
    "FileOperationError",
    "FileOperator",
    "FileStat",
    "InstructableResource",
    "LocalTmpFileOperator",
    "PathNotAllowedError",
    "Resource",
    "ResourceEntry",
    "ResourceFactory",
    "ResourceRegistry",
    "ResourceRegistryState",
    "ResumableResource",
    "Shell",
    "ShellExecutionError",
    "ShellTimeoutError",
    "TmpFileOperator",
    "TruncatedResult",
    "generate_filetree",
]
