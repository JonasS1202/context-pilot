#!/usr/bin/env python3
"""
ContextPilot – An intelligent context manager for AI assistants.

Usage
-----

# 1. The main "assist" command (Recommended)
#    Automatically builds the best prompt based on project size.
pilot assist "Refactor the authentication logic to use a service class." [--ext .py .toml]

# 2. The "files" command (for interactive sessions)
#    Used after the AI requests specific files in a large project.
pilot files src/main.py src/utils.py

# 3. The "git" command (for commit messages)
#    Generates a prompt to suggest commit messages for staged/unstaged changes.
pilot git
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
import subprocess
from collections import defaultdict

# Third-party libraries (pip install tiktoken pyperclip)
try:
    import pyperclip
    import tiktoken
except ImportError:
    print(
        "Error: Required packages not found. Please run 'pip install tiktoken pyperclip'",
        file=sys.stderr,
    )
    sys.exit(1)


# ── CONFIGURATION ───────────────────────────────────────────────────
# Default directories to ignore. For best results, rely on a project's .gitignore file.
DEFAULT_IGNORE_DIRS = [".git", "venv", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist", ".eggs"]
DEFAULT_IGNORE_FILES: List[str] = ["pilot.py", "chatgpt_prompt.txt"]
DEFAULT_ONLY_FROM_DIRS: Optional[List[str]] = None

# Context window threshold in tokens. If the project exceeds this,
# it will switch to the interactive "Guided Discovery" mode.
CONTEXT_THRESHOLD = 100_000
# ──────────────────────────────────────────────────────────────────────


# ── Path & Git Helpers ────────────────────────────────────────────────
def _is_ignored(p: Path, root: Path, ignore_dirs: list[Path], ignore_files: list[Path]) -> bool:
    """Check if a path should be ignored."""
    rel = p.relative_to(root)
    # NOTE: For a more robust solution, use the `pathspec` library to parse .gitignore
    is_dir_ignored = any(part in [ign.name for ign in ignore_dirs] for part in rel.parts)
    is_file_ignored = any(rel.name == ign.name for ign in ignore_files)
    return is_dir_ignored or is_file_ignored

def _in_scope_dir(p: Path, root: Path, only_from: Optional[List[str]]) -> bool:
    """Check if a path is within the desired scope."""
    if only_from is None or p == root:
        return True
    top_level_dir = p.relative_to(root).parts[0] if p.relative_to(root).parts else ""
    return top_level_dir in only_from

def load_gitignore_patterns(root: Path) -> list[str]:
    """Loads and cleans patterns from the .gitignore file."""
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []
    patterns = []
    with open(gitignore_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line.rstrip('/'))
    return patterns

def _run_git(cmd: list[str]) -> str:
    """Run a git command; exit gracefully on error."""
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("Error: 'git' command not found. Is Git installed and in your PATH?", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e.stderr}", file=sys.stderr)
        return ""


# ── Core Logic ────────────────────────────────────────────────────────
def build_tree(root: Path, ignore_dirs: list[Path], ignore_files: list[Path], only_from: Optional[list[str]]) -> str:
    """Return an ASCII tree of the project structure."""
    lines = ["."]
    
    def walk(dir_path: Path, prefix: str = ""):
        entries = sorted([p for p in dir_path.iterdir()], key=lambda x: (x.is_file(), x.name.lower()))
        filtered_entries = [e for e in entries if not _is_ignored(e, root, ignore_dirs, ignore_files) and _in_scope_dir(e, root, only_from)]

        for i, entry in enumerate(filtered_entries):
            connector = "└── " if i == len(filtered_entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if i == len(filtered_entries) - 1 else "│   "
                walk(entry, prefix + extension)

    walk(root)
    return "\n".join(lines)


def collect_project_files(
    root: Path,
    ignore_dirs: list[Path],
    ignore_files: list[Path],
    only_from: Optional[list[str]],
    extensions: list[str],
) -> list[Path]:
    """Collect all project files matching the given extensions."""
    project_files = []
    for dirpath, _, filenames in os.walk(root):
        d_path = Path(dirpath)
        if _is_ignored(d_path, root, ignore_dirs, ignore_files) or not _in_scope_dir(d_path, root, only_from):
            continue
        for fname in filenames:
            if any(fname.endswith(ext) for ext in extensions):
                fpath = d_path / fname
                if not _is_ignored(fpath, root, ignore_dirs, ignore_files):
                    project_files.append(fpath)
    return sorted(project_files)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# ── Prompt Builders ───────────────────────────────────────────────────
def make_full_context_prompt(task: str, tree: str, files: list[Path], root: Path) -> str:
    """Generates a prompt with the full project context for smaller projects."""
    file_contents = []
    for fpath in files:
        rel_path = fpath.relative_to(root)
        try:
            content = fpath.read_text(encoding="utf-8").strip()
            file_contents.append(f"## `{rel_path}`:\n```\n{content}\n```")
        except UnicodeDecodeError:
            continue # Skip binary files

    all_contents = "\n\n".join(file_contents)
    return (
        f"# Your Task\n{task}\n\n"
        "You are a world-class Principal Software Engineer known for your meticulous attention to detail, clarity, and producing production-ready code. You have been given the complete context of a project and a task to perform.\n\n"
        f"# Project Context\n\n## Folder Structure\n```\n{tree}\n```\n\n## File Contents\n{all_contents}\n\n"
        "---\n"
        "## Your Mission (Read Carefully)\n\n"
        "Your mission is to execute the task by first creating a flawless plan and then implementing it perfectly. You must follow the two-phase process below without deviation.\n\n"
        "**Phase 1: The Plan**\n"
        "First, respond *only* with a comprehensive, step-by-step plan. A perfect plan includes:\n"
        "1.  A **High-Level Summary** of your proposed solution.\n"
        "2.  A numbered list of **Implementation Steps**. For each step, you must specify:\n"
        "    -   The **Goal** of the step (e.g., \"Create a new service class for authentication\").\n"
        "    -   The specific **File(s)** that will be created, modified, or deleted.\n"
        "    -   A brief **Reasoning** for why this step is necessary.\n"
        "**Do not write any code in the planning phase.**\n\n"
        "**Phase 2: The Execution**\n"
        "After presenting the plan, you will execute it. You must address each step from your plan one by one. For every file modification, you *must* use the following rigid format:\n\n"
        "**File:** `path/to/the/file.ext`\n"
        "**Action:** [Create file | Replace function `function_name` | Add after line X | Delete lines A-B]\n"
        "```python\n"
        "// ... your complete, final code block goes here ...\n"
        "```\n"
        "**Reasoning:** [A brief sentence connecting this code to the plan.]\n\n"
        "---\n\n"
        "### GOLDEN RULES (Non-Negotiable)\n"
        "1.  **Plan First, Then Code:** You must always provide the complete plan first.\n"
        "2.  **No Diffs, No Patches:** All code blocks must be complete and final. Never use `+`/`-` prefixes.\n"
        "3.  **Adhere to the Format:** The `File:`, `Action:`, `Code Block`, `Reasoning:` format is mandatory for all code changes."
    )


def make_discovery_prompt(task: str, tree: str) -> str:
    """Generates the initial prompt for large projects, kicking off an interactive session."""
    return (
        f"# Your Task\n{task}\n\n"
        "You are a world-class Principal Software Engineer known for your meticulous analysis and problem-solving skills. You are tasked with solving a problem in a large, unfamiliar codebase. You must gather information methodically before proposing a solution.\n\n"
        f"# Project Folder Structure\n```\n{tree}\n```\n\n"
        "---\n"
        "## Your Mission (Read Carefully)\n\n"
        "Your mission is to gather sufficient information to create a flawless plan, and only then, execute that plan. You must follow the phases below without deviation.\n\n"
        "**Phase 1: Information Gathering**\n\n"
        "1.  **Analyze the Structure:** Review the folder structure to form hypotheses about the project's architecture. Identify potential files of interest.\n"
        "2.  **Request Files Incrementally:** To test your hypotheses, request file contents a few at a time using this exact command on its own line:\n"
        "    `pilot files path/to/file.py`\n"
        "3.  **Never Assume:** As a meticulous engineer, you must verify every assumption. If you are unsure about something, request the relevant file. Do not proceed with incomplete information.\n\n"
        "**Phase 2: The Plan**\n\n"
        "4.  **Signal Readiness & Create Plan:** Once you are certain you have all the information needed, you must start your response with `✅ I have enough information.` and then immediately provide a comprehensive, step-by-step plan. A perfect plan includes:\n"
        "    -   A **High-Level Summary** of your proposed solution.\n"
        "    -   A numbered list of **Implementation Steps**. For each step, you must specify:\n"
        "        -   The **Goal** of the step.\n"
        "        -   The specific **File(s)** that will be created or modified.\n"
        "        -   A brief **Reasoning** for why this step is necessary.\n"
        "    **Do not write any code in the planning phase.**\n\n"
        "**Phase 3: The Execution**\n\n"
        "5.  **Execute the Plan:** After presenting the plan, you will execute it. You must address each step from your plan one by one. For every file modification, you *must* use the following rigid format:\n\n"
        "**File:** `path/to/the/file.ext`\n"
        "**Action:** [Create file | Replace function `function_name` | Add after line X]\n"
        "```python\n"
        "// ... your complete, final code block goes here ...\n"
        "```\n"
        "**Reasoning:** [A brief sentence connecting this code to the plan.]\n\n"
        "---\n\n"
        "### GOLDEN RULES (Non-Negotiable)\n"
        "1.  **Gather, Plan, then Code:** You must follow the phases in order.\n"
        "2.  **No Diffs, No Patches:** All code blocks must be complete and final. Never use `+`/`-` prefixes.\n"
        "3.  **Adhere to the Format:** The `File:`, `Action:`, `Code Block`, `Reasoning:` format is mandatory for all code changes during execution."
    )


def make_files_prompt(files: list[Path], root: Path) -> str:
    """Generates a prompt containing only the content of specifically requested files."""
    file_contents = []
    for fpath in files:
        rel_path = fpath.relative_to(root)
        try:
            content = fpath.read_text(encoding="utf-8").strip()
            file_contents.append(f"## `{rel_path}`:\n```\n{content}\n```")
        except (UnicodeDecodeError, FileNotFoundError):
            file_contents.append(f"## `{rel_path}`:\n```\nError: Could not read this file.\n```")

    return "\n\n".join(file_contents)


def make_git_prompt(staged_only: bool = False) -> str:
    """Builds a prompt to suggest commit messages based on git diffs."""
    diff_cmd = ["git", "diff", "--no-color", "--unified=3"]
    diff = _run_git(diff_cmd + ["--cached"])
    if not staged_only:
        diff += _run_git(diff_cmd)

    if not diff.strip():
        return "No Git changes detected."

    return (
        f"## Full Git Diff\n```diff\n{diff}\n```\n\n"
        "---\n"
        "## Instructions\n\n"
        "Analyze the git diff and suggest one or more commit messages in the Conventional Commits format. "
        "Group related file changes into logical commits. Respond **only** with the commit plan in this exact format:\n\n"
        "Commit 1:\n"
        "files: path/to/file1.py path/to/file2.py\n"
        "message: \"feat: add user authentication endpoint\"\n\n"
        "Commit 2:\n"
        "files: path/to/docs.md\n"
        "message: \"docs: update API documentation for auth\"\n"
    )

# ── CLI Entrypoint ────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(prog="pilot", description="An intelligent context manager for AI assistants.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root (default: cwd)")
    sub = parser.add_subparsers(dest="mode", required=True, metavar="{assist,files,git}")

    # --- Assist Mode ---
    p_assist = sub.add_parser("assist", help="Intelligently build context for a given task.")
    p_assist.add_argument("task", type=str, help="The high-level task for the AI assistant.")
    p_assist.add_argument("-o", "--output", type=Path, default="chatgpt_prompt.txt")
    p_assist.add_argument("-c", "--copy", action="store_true", help="Copy the prompt to the clipboard.")
    p_assist.add_argument("--ext", nargs="+", default=[".py", ".toml", ".yaml", ".json", ".md", ".sh", ".txt"], help="File extensions to include.")
    
    # --- Files Mode ---
    p_files = sub.add_parser("files", help="Provide specific files to the AI during an interactive session.")
    p_files.add_argument("files", nargs="+", type=Path, help="File paths relative to --root.")
    p_files.add_argument("-o", "--output", type=Path, default="chatgpt_prompt.txt")
    p_files.add_argument("-c", "--copy", action="store_true", help="Copy the prompt to the clipboard.")

    # --- Git Mode ---
    p_git = sub.add_parser("git", help="Generate a prompt to suggest commit messages.")
    p_git.add_argument("-o", "--output", type=Path, default="chatgpt_prompt.txt")
    p_git.add_argument("-c", "--copy", action="store_true", help="Copy the prompt to the clipboard.")
    p_git.add_argument("--staged", action="store_true", help="Analyze only staged changes.")

    args = parser.parse_args()
    root: Path = args.root.resolve()
    
    # --- Shared Setup ---
    gitignore_patterns = load_gitignore_patterns(root)
    ignore_dirs_names = DEFAULT_IGNORE_DIRS + gitignore_patterns
    ignore_files_names = DEFAULT_IGNORE_FILES + [args.output.name]
    ignore_dirs = [root / d for d in ignore_dirs_names]
    ignore_files = [root / f for f in ignore_files_names]

    prompt = ""
    if args.mode == "assist":
        print("🚀 Starting analysis...")
        tree = build_tree(root, ignore_dirs, ignore_files, DEFAULT_ONLY_FROM_DIRS)
        project_files = collect_project_files(root, ignore_dirs, ignore_files, DEFAULT_ONLY_FROM_DIRS, args.ext)
        
        print(f"🔍 Found {len(project_files)} relevant files.")
        
        full_context = tree + "\n" + args.task
        for fpath in project_files:
            try:
                full_context += fpath.read_text(encoding="utf-8")
            except Exception:
                pass
        
        token_count = count_tokens(full_context)
        print(f"📊 Estimated token count: {token_count:,}")

        if token_count < CONTEXT_THRESHOLD:
            print("✅ Project fits in context window. Generating full-context prompt.")
            prompt = make_full_context_prompt(args.task, tree, project_files, root)
        else:
            print("⚠️ Project is large. Generating interactive 'Guided Discovery' prompt.")
            prompt = make_discovery_prompt(args.task, tree)
    
    elif args.mode == "files":
        explicit_files = [(root / f).resolve() for f in args.files]
        prompt = make_files_prompt(explicit_files, root)

    elif args.mode == "git":
        prompt = make_git_prompt(staged_only=args.staged)

    # --- Output ---
    if args.copy:
        pyperclip.copy(prompt)
        print(f"✨ Prompt copied to clipboard ({len(prompt):,} chars)")
    else:
        args.output.write_text(prompt, encoding="utf-8")
        print(f"✨ Prompt written to {args.output} ({len(prompt):,} chars)")


if __name__ == "__main__":
    main()