"""Phase 0 reconnaissance helper: inspects soup-cli version, CLI commands, and package structure."""
import importlib.metadata
import inspect
import os
import platform
import subprocess
import sys
from pathlib import Path

import soup_cli
import torch


def run_cli(args: list[str]) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "soup_cli"] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_soup_entry(args: list[str]) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"
    soup_exe = Path(sys.executable).parent / ("soup.exe" if os.name == "nt" else "soup")
    cmd = [str(soup_exe)] + args if soup_exe.exists() else [sys.executable, "-m", "soup_cli.cli"] + args
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def main() -> None:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("PHASE 0: ENVIRONMENT & SOUP-CLI RECONNAISSANCE")
    lines.append("=" * 80)
    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"Python: {sys.version}")
    lines.append(f"Executable: {sys.executable}")
    lines.append(f"soup-cli version: {importlib.metadata.version('soup-cli')}")
    lines.append(f"torch version: {torch.__version__} (cuda.is_available={torch.cuda.is_available()})")
    if torch.cuda.is_available():
        lines.append(f"CUDA device: {torch.cuda.get_device_name(0)}")

    pkg_dir = Path(inspect.getfile(soup_cli)).parent
    lines.append(f"\nsoup_cli package path: {pkg_dir}")
    lines.append("\n--- Package File Tree ---")
    for p in sorted(pkg_dir.rglob("*.py")):
        rel = p.relative_to(pkg_dir)
        lines.append(f"  {rel} ({p.stat().st_size} bytes)")

    lines.append("\n--- Entry Points ---")
    dist = importlib.metadata.distribution("soup-cli")
    for ep in dist.entry_points:
        lines.append(f"  {ep.group}: {ep.name} -> {ep.value}")

    for args in [["--version"], ["--help"]]:
        rc, out, err = run_soup_entry(args)
        lines.append("\n" + "=" * 80)
        lines.append(f"COMMAND: soup {' '.join(args)} (exit_code={rc})")
        lines.append("-" * 80)
        lines.append("STDOUT:")
        lines.append(out)
        if err:
            lines.append("STDERR:")
            lines.append(err)

    # Discover subcommands from typer app if available
    try:
        from soup_cli.cli import app
        subcommands = []
        for cmd_info in getattr(app, "registered_commands", []):
            name = cmd_info.name or (cmd_info.callback.__name__ if cmd_info.callback else "unknown")
            subcommands.append(name)
        for grp_info in getattr(app, "registered_groups", []):
            if grp_info.name:
                subcommands.append(grp_info.name)
        lines.append(f"\nDiscovered Typer commands/groups: {subcommands}")
        for sub in subcommands:
            rc, out, err = run_soup_entry([sub, "--help"])
            lines.append("\n" + "=" * 80)
            lines.append(f"COMMAND: soup {sub} --help (exit_code={rc})")
            lines.append("-" * 80)
            lines.append("STDOUT:")
            lines.append(out)
            if err:
                lines.append("STDERR:")
                lines.append(err)
    except Exception as exc:
        lines.append(f"\nError inspecting soup_cli.cli.app: {exc!r}")

    log_path = Path("raw_logs/00_env_and_cli_help.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_text = "\n".join(lines) + "\n"
    log_path.write_text(full_text, encoding="utf-8")
    print(f"Wrote {len(lines)} lines to {log_path}")


if __name__ == "__main__":
    main()
