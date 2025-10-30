#!/usr/bin/env python3
"""
ContextPilot – An intelligent context manager for AI assistants.
"""
from __future__ import annotations

import argparse
import sys
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Third-party libraries (pip install tiktoken pyperclip pathspec)
try:
    import pyperclip
    import tiktoken
    import pathspec
except ImportError:
    print(
        "Error: Required packages not found. Please run 'pip install tiktoken pyperclip pathspec'",
        file=sys.stderr,
    )
    sys.exit(1)

# Local imports
from . import prompts

# ── CONFIGURATION ───────────────────────────────────────────────────
DEFAULT_EXT = [".py", ".toml", ".yaml", ".json", ".md", ".sh", ".txt", ".properties", ".html"]
DEFAULT_IGNORE_DIRS = [".git", "venv", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist", ".eggs"]
DEFAULT_IGNORE_FILES: List[str] = ["pilot.py", "prompt.txt", "prompts.py"]
TOKEN_CONVERSION_FACTOR = 1.28

# ── DATA STRUCTURES ─────────────────────────────────────────────────
@dataclass
class AppConfig:
    """Encapsulates application configuration and shared state."""
    root: Path
    args: argparse.Namespace
    ignore_spec: pathspec.PathSpec = field(init=False)

    def __post_init__(self):
        """Load ignore patterns after initialization."""
        gitignore_path = self.root / ".gitignore"
        ignore_patterns = DEFAULT_IGNORE_DIRS + DEFAULT_IGNORE_FILES
        if hasattr(self.args, 'output') and self.args.output:
            ignore_patterns.append(self.args.output.name)

        if gitignore_path.is_file():
            with gitignore_path.open("r", encoding="utf-8") as f:
                ignore_patterns.extend(f.read().splitlines())
        
        self.ignore_spec = pathspec.PathSpec.from_lines('gitwildmatch', ignore_patterns)

    def is_ignored(self, path: Path) -> bool:
        """Check if a path should be ignored based on the loaded spec."""
        # pathspec works with relative paths
        relative_path = path.relative_to(self.root) if path.is_absolute() else path
        return self.ignore_spec.match_file(relative_path)

# ── Path & Git Helpers ────────────────────────────────────────────────
def _run_git(cmd: list[str]) -> str:
    """Run a git command; exit gracefully on error."""
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE, cwd=Path.cwd())
    except FileNotFoundError:
        print("Error: 'git' command not found. Is Git installed and in your PATH?", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e.stderr}", file=sys.stderr)
        return ""

def build_tree(config: AppConfig) -> str:
    """Return an ASCII tree of the project structure."""
    lines = ["."]
    
    def walk(dir_path: Path, prefix: str = ""):
        # Filter out ignored directories before iterating
        try:
            entries = sorted(
                [p for p in dir_path.iterdir() if not config.is_ignored(p)],
                key=lambda x: (x.is_file(), x.name.lower())
            )
        except PermissionError:
            return # Skip directories we can't read

        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                walk(entry, prefix + extension)

    walk(config.root)
    return "\n".join(lines)


def collect_project_files(config: AppConfig) -> list[Path]:
    """Collect all project files matching the given extensions."""
    project_files = []
    for fpath in config.root.rglob("*"):
        if fpath.is_file() and not config.is_ignored(fpath):
            if any(fpath.name.endswith(ext) for ext in config.args.ext):
                project_files.append(fpath)
    return sorted(project_files)

# ── Token Counter ───────────────────────────────────────────────────
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return int(len(encoding.encode(text)) * TOKEN_CONVERSION_FACTOR)

# ── Prompt Builders ───────────────────────────────────────────────────
def make_full_context_prompt(task: str, tree: str, files: List[Path], root: Path) -> str:
    """Builds the file context and formats it with the main prompt template."""
    file_contents = []
    for fpath in files:
        rel_path = fpath.relative_to(root)
        try:
            content = fpath.read_text(encoding="utf-8").strip()
            file_contents.append(f"#### File: `{rel_path}`\n```\n{content}\n```")
        except (UnicodeDecodeError, IOError) as e:
            print(f"Warning: Could not read file {rel_path}: {e}", file=sys.stderr)
            continue
    all_contents = "\n\n".join(file_contents)
    return prompts.get_full_context_prompt(task, tree, all_contents)

def make_files_prompt(files: list[Path], root: Path) -> str:
    """Generates a prompt containing only the content of specifically requested files."""
    file_contents = []
    for fpath in files:
        rel_path = fpath.relative_to(root)
        try:
            # Ensure file exists before trying to read
            if not fpath.is_file():
                file_contents.append(f"## `{rel_path}`:\n```\nError: File not found at this path.\n```")
                continue
            content = fpath.read_text(encoding="utf-8").strip()
            file_contents.append(f"## `{rel_path}`:\n```\n{content}\n```")
        except (UnicodeDecodeError, IOError) as e:
            file_contents.append(f"## `{rel_path}`:\n```\nError: Could not read this file: {e}\n```")
    return prompts.get_files_prompt(file_contents)

def make_git_prompt(staged_only: bool = False) -> str:
    """Gathers git diff and formats it with the git prompt template."""
    diff_cmd = ["git", "diff", "--no-color", "--unified=3"]
    diff = _run_git(diff_cmd + ["--cached"]) if staged_only else ""
    if not staged_only:
        diff += _run_git(diff_cmd)
    return prompts.get_git_prompt(diff)

# ── Command Handlers ───────────────────────────────────────────────────
def handle_assist(config: AppConfig):
    """Handler for the 'assist' command."""
    print("🚀 Starting analysis...")
    tree = build_tree(config)
    project_files = collect_project_files(config)
    print(f"🔍 Found {len(project_files)} relevant files.")
    
    # Estimate token count for deciding which prompt to use
    full_context_text = [tree, config.args.task]
    for fpath in project_files:
        try:
            full_context_text.append(fpath.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, IOError):
            pass # Silently ignore unreadable files for token counting
    
    estimated_token_count = count_tokens("\n".join(full_context_text))

    if estimated_token_count < config.args.threshold:
        print("✅ Project fits in context window. Generating full-context prompt.")
        prompt = make_full_context_prompt(config.args.task, tree, project_files, config.root)
    else:
        print("⚠️ Project is large. Generating interactive 'Guided Discovery' prompt.")
        print(f"   (Full context would have been ~{estimated_token_count:,} tokens, threshold is {config.args.threshold:,})")
        prompt = prompts.get_discovery_prompt(config.args.task, tree)
    
    output_prompt(prompt, config.args)

def handle_files(config: AppConfig):
    """Handler for the 'files' command."""
    explicit_files = [(config.root / f).resolve() for f in config.args.files]
    prompt = make_files_prompt(explicit_files, config.root)
    output_prompt(prompt, config.args)

def handle_git(config: AppConfig):
    """Handler for the 'git' command."""
    prompt = make_git_prompt(staged_only=config.args.staged)
    output_prompt(prompt, config.args)

def output_prompt(prompt: str, args: argparse.Namespace):
    """Handles writing the prompt to file or clipboard."""
    token_count = count_tokens(prompt)
    print(f"📊 Estimated token count: {token_count:,}")
    
    if args.copy:
        pyperclip.copy(prompt)
        print(f"✨ Prompt copied to clipboard ({len(prompt):,} chars)")
    else:
        args.output.write_text(prompt, encoding="utf-8")
        print(f"✨ Prompt written to {args.output} ({len(prompt):,} chars)")

# ── CLI Entrypoint ────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(prog="pilot", description="An intelligent context manager for AI assistants.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root (default: cwd)")
    
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("-o", "--output", type=Path, default="prompt.txt")
    output_parser.add_argument("-c", "--copy", action="store_true", help="Copy the prompt to the clipboard.")

    sub = parser.add_subparsers(dest="mode", required=True, metavar="{assist,files,git}")

    p_assist = sub.add_parser("assist", help="Build context for a given task.", parents=[output_parser])
    p_assist.add_argument("task", type=str, help="The high-level task for the AI assistant.")
    p_assist.add_argument("--ext", nargs="+", default=DEFAULT_EXT, help="File extensions to include.")
    p_assist.add_argument("--threshold", type=int, default=1_000_000, help="Token threshold for interactive mode.")
    p_assist.set_defaults(func=handle_assist)
    
    p_files = sub.add_parser("files", help="Provide specific files to the AI.", parents=[output_parser])
    p_files.add_argument("files", nargs="+", type=Path, help="File paths relative to --root.")
    p_files.set_defaults(func=handle_files)

    p_git = sub.add_parser("git", help="Generate a prompt for commit messages.", parents=[output_parser])
    p_git.add_argument("--staged", action="store_true", help="Analyze only staged changes.")
    p_git.set_defaults(func=handle_git)

    p_help = sub.add_parser("help", help="Show help for a specific command.")
    p_help.add_argument("command", nargs="?", choices=["assist", "files", "git"], help="The command to get help for.")

    args = parser.parse_args()

    if args.mode == "help":
        parsers = {"assist": p_assist, "files": p_files, "git": p_git}
        target_parser = parsers.get(args.command, parser)
        target_parser.print_help()
        sys.exit(0)

    config = AppConfig(root=args.root.resolve(), args=args)
    args.func(config)

if __name__ == "__main__":
    main()