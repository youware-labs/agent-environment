"""Shell abstraction for environment module.

This module provides an abstract base class for shell command execution.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class Shell(ABC):
    """Abstract base class for shell command execution."""

    def __init__(
        self,
        default_cwd: Path,
        allowed_paths: list[Path] | None = None,
        default_timeout: float = 30.0,
        skip_instructions: bool = False,
    ):
        """Initialize Shell.

        Args:
            default_cwd: Default working directory for command execution. Required.
                Always included in allowed_paths.
            allowed_paths: Directories allowed as working directories.
                If None, defaults to [default_cwd].
            default_timeout: Default timeout in seconds.
            skip_instructions: If True, get_context_instructions returns None.
        """
        self._default_cwd = default_cwd.resolve()

        # Build allowed_paths, ensuring default_cwd is included
        if allowed_paths is None:
            self._allowed_paths = [self._default_cwd]
        else:
            resolved_paths = [p.resolve() for p in allowed_paths]
            if self._default_cwd not in resolved_paths:
                resolved_paths.append(self._default_cwd)
            self._allowed_paths = resolved_paths

        self._default_timeout = default_timeout
        self._skip_instructions = skip_instructions

    @abstractmethod
    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        """Execute a command and return (exit_code, stdout, stderr).

        Args:
            command: Command string to execute via shell.
            timeout: Timeout in seconds (uses default if None).
            env: Environment variables.
            cwd: Working directory (relative or absolute path).

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        ...

    async def get_context_instructions(self) -> str | None:
        """Return instructions for the agent about shell capabilities."""
        if self._skip_instructions:
            return None
        paths_str = "\n".join(f"    <path>{p}</path>" for p in self._allowed_paths)
        return f"""<shell-execution>
  <allowed-working-directories>
{paths_str}
  </allowed-working-directories>
  <default-working-directory>{self._default_cwd}</default-working-directory>
  <default-timeout>{self._default_timeout}s</default-timeout>
  <note>Commands will be executed with the working directory validated.</note>
</shell-execution>"""

    async def close(self) -> None:  # noqa: B027
        """Clean up resources owned by this Shell.

        Subclasses can override this to clean up additional resources
        (e.g., persistent shell sessions, SSH connections).
        Always call super().close() when overriding.
        """
