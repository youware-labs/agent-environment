"""Environment abstractions for file operations and shell execution.

This module provides Protocol-based interfaces and implementations for
environment operations, allowing different backends (local, remote, S3, SSH, etc.)
to be used interchangeably.
"""

from agent_environment.environment import Environment
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
from agent_environment.file_operator import (
    DEFAULT_INSTRUCTIONS_MAX_DEPTH,
    DEFAULT_INSTRUCTIONS_SKIP_DIRS,
    FileOperator,
    LocalTmpFileOperator,
)
from agent_environment.protocols import (
    DEFAULT_CHUNK_SIZE,
    InstructableResource,
    Resource,
    ResumableResource,
    TmpFileOperator,
)
from agent_environment.resources import (
    BaseResource,
    ResourceEntry,
    ResourceFactory,
    ResourceRegistry,
    ResourceRegistryState,
)
from agent_environment.shell import Shell
from agent_environment.types import FileStat, TruncatedResult
from agent_environment.utils import generate_filetree

__all__ = [
    "DEFAULT_CHUNK_SIZE",
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
