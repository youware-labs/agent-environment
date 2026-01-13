"""Tests for LocalTmpFileOperator and FileOperator."""

from pathlib import Path

from agent_environment import (
    LocalTmpFileOperator,
)


async def test_local_tmp_file_operator_basic_operations(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support basic file operations."""
    op = LocalTmpFileOperator(tmp_path)

    # Write and read text
    await op.write_file("test.txt", "Hello World")
    content = await op.read_file("test.txt")
    assert content == "Hello World"

    # Check exists
    assert await op.exists("test.txt")
    assert not await op.exists("nonexistent.txt")

    # Check is_file/is_dir
    assert await op.is_file("test.txt")
    assert not await op.is_dir("test.txt")


async def test_local_tmp_file_operator_read_bytes(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support reading bytes."""
    op = LocalTmpFileOperator(tmp_path)

    # Write bytes
    await op.write_file("binary.bin", b"\x00\x01\x02\x03")
    content = await op.read_bytes("binary.bin")
    assert content == b"\x00\x01\x02\x03"


async def test_local_tmp_file_operator_read_with_offset(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support offset and length."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("test.txt", "0123456789")

    # Read with offset
    content = await op.read_file("test.txt", offset=3)
    assert content == "3456789"

    # Read with length
    content = await op.read_file("test.txt", length=5)
    assert content == "01234"

    # Read with offset and length
    content = await op.read_file("test.txt", offset=2, length=4)
    assert content == "2345"


async def test_local_tmp_file_operator_append(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support appending."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("test.txt", "Hello")
    await op.append_file("test.txt", " World")
    content = await op.read_file("test.txt")
    assert content == "Hello World"


async def test_local_tmp_file_operator_delete(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support deleting files and dirs."""
    op = LocalTmpFileOperator(tmp_path)

    # Delete file
    await op.write_file("test.txt", "content")
    assert await op.exists("test.txt")
    await op.delete("test.txt")
    assert not await op.exists("test.txt")

    # Delete empty directory
    await op.mkdir("emptydir")
    assert await op.is_dir("emptydir")
    await op.delete("emptydir")
    assert not await op.exists("emptydir")


async def test_local_tmp_file_operator_mkdir(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support creating directories."""
    op = LocalTmpFileOperator(tmp_path)

    await op.mkdir("newdir")
    assert await op.is_dir("newdir")

    # With parents
    await op.mkdir("a/b/c", parents=True)
    assert await op.is_dir("a/b/c")


async def test_local_tmp_file_operator_list_dir(tmp_path: Path) -> None:
    """LocalTmpFileOperator should list directory contents."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("a.txt", "a")
    await op.write_file("b.txt", "b")
    await op.mkdir("subdir")

    entries = await op.list_dir(".")
    assert sorted(entries) == ["a.txt", "b.txt", "subdir"]


async def test_local_tmp_file_operator_move(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support moving files."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("source.txt", "content")
    await op.move("source.txt", "dest.txt")

    assert not await op.exists("source.txt")
    assert await op.exists("dest.txt")
    assert await op.read_file("dest.txt") == "content"


async def test_local_tmp_file_operator_copy(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support copying files."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("source.txt", "content")
    await op.copy("source.txt", "dest.txt")

    assert await op.exists("source.txt")
    assert await op.exists("dest.txt")
    assert await op.read_file("dest.txt") == "content"


async def test_local_tmp_file_operator_stat(tmp_path: Path) -> None:
    """LocalTmpFileOperator should return file stats."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("test.txt", "hello")
    stat = await op.stat("test.txt")

    assert stat["size"] == 5
    assert stat["is_file"] is True
    assert stat["is_dir"] is False
    assert stat["mtime"] > 0


async def test_local_tmp_file_operator_glob(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support glob patterns."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("a.txt", "a")
    await op.write_file("b.txt", "b")
    await op.write_file("c.py", "c")

    txt_files = await op.glob("*.txt")
    assert sorted(txt_files) == ["a.txt", "b.txt"]


async def test_local_tmp_file_operator_truncate_to_tmp(tmp_path: Path) -> None:
    """LocalTmpFileOperator should truncate large content and save to tmp."""
    op = LocalTmpFileOperator(tmp_path)

    # Small content - no truncation
    small = "Hello"
    result = await op.truncate_to_tmp(small, "small.txt", max_length=100)
    assert result == small

    # Large content - should truncate
    large = "x" * 200
    result = await op.truncate_to_tmp(large, "large.txt", max_length=100)

    assert isinstance(result, dict)
    assert len(result["content"]) <= 100
    assert "large.txt" in result["file_path"]
    assert "truncated" in result["message"].lower()


async def test_local_tmp_file_operator_is_managed_path(tmp_path: Path) -> None:
    """LocalTmpFileOperator should correctly identify managed paths."""
    op = LocalTmpFileOperator(tmp_path)

    # Path within tmp_dir
    is_managed, _path = op.is_managed_path("subdir/file.txt", tmp_path)
    assert is_managed is True

    # Path outside tmp_dir
    is_managed, _path = op.is_managed_path("/etc/passwd", tmp_path)
    assert is_managed is False


async def test_local_tmp_file_operator_tmp_dir_property(tmp_path: Path) -> None:
    """LocalTmpFileOperator should expose tmp_dir property."""
    op = LocalTmpFileOperator(tmp_path)
    assert op.tmp_dir == str(tmp_path)


async def test_local_tmp_operator_read_bytes_with_offset(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support reading bytes with offset and length."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("test.bin", b"0123456789")

    # Read with offset
    content = await op.read_bytes("test.bin", offset=3)
    assert content == b"3456789"

    # Read with offset and length
    content = await op.read_bytes("test.bin", offset=3, length=4)
    assert content == b"3456"


async def test_local_tmp_operator_write_bytes(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support writing bytes."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("test.bin", b"\x00\x01\x02\x03")
    content = await op.read_bytes("test.bin")
    assert content == b"\x00\x01\x02\x03"


async def test_local_tmp_operator_append_bytes(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support appending bytes."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("test.bin", b"hello")
    await op.append_file("test.bin", b"world")
    content = await op.read_bytes("test.bin")
    assert content == b"helloworld"


async def test_local_tmp_operator_append_text(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support appending text."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("test.txt", "hello")
    await op.append_file("test.txt", "world")
    content = await op.read_file("test.txt")
    assert content == "helloworld"


async def test_local_tmp_operator_delete_dir(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support deleting empty directories."""
    op = LocalTmpFileOperator(tmp_path)

    await op.mkdir("empty_dir")
    assert await op.is_dir("empty_dir")

    await op.delete("empty_dir")
    assert not await op.exists("empty_dir")
