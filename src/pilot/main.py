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
DEFAULT_IGNORE_FILES: List[str] = ["pilot.py", "prompt.txt"]
DEFAULT_ONLY_FROM_DIRS: Optional[List[str]] = None
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


def count_tokens(text: str, model: str = "gpt-4", factor_to_gemini: float = 1.28) -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return int(len(encoding.encode(text)) * factor_to_gemini)


# ── Prompt Builders ───────────────────────────────────────────────────
def make_full_context_prompt_old(task: str, tree: str, files: list[Path], root: Path) -> str:
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
        "Your mission is to execute the task by first clarifying all requirements, then creating a flawless plan, and finally implementing it perfectly. You must follow the phased process below without deviation.\n\n"
        "**Phase 0: Task Clarification**\n"
        "Your first and only action is to analyze the task description for any ambiguity, vagueness, or missing context.\n"
        "- If the task is perfectly clear and specific, respond *only* with the line: `✅ No questions left.`\n"
        "- If you have any questions that would help you better understand the requirements, goals, or scope, ask them. Continue asking questions in subsequent turns until you are 100% confident. Once all your questions are answered, respond with the line `✅ No questions left.` and then, **in the same message**, immediately proceed with **Phase 1**.\n"
        "Do not proceed to the next phase until you have sent this signal.\n\n"
        "**Phase 1: The Plan**\n"
        "After signaling you have no questions, respond *only* with a comprehensive, step-by-step plan. A perfect plan includes:\n"
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


from pathlib import Path
from typing import List

def make_full_context_prompt(task: str, tree: str, files: List[Path], root: Path) -> str:
    """
    Generates the optimized C.R.A.F.T.E.R. prompt with the full project context.
    """
    file_contents = []
    for fpath in files:
        if fpath.is_dir():
            continue
        rel_path = fpath.relative_to(root)
        try:
            content = fpath.read_text(encoding="utf-8").strip()
            # OPTIMIZATION: Using a more distinct heading for file contents.
            file_contents.append(f"#### File: `{rel_path}`\n```\n{content}\n```")
        except (UnicodeDecodeError, IOError):
            continue

    all_contents = "\n\n".join(file_contents)

    # The prompt is now a multi-line f-string for better readability.
    # Key changes are in the 'A - ACTION' section.
    return (
        f"[PROMPT TOPIC OR THEME]: Optimized Prompt for Full-Context Software Development Tasks\n\n"
        f"C - CONTEXT:\n\n"
        f"Background: You are about to perform a task within an existing software project. You have been provided with the complete and unabridged context of the current project state, including the full directory structure and the content of every relevant file. This is equivalent to having the project open in your IDE. Your goal is to execute the assigned task with surgical precision, adhering to the highest standards of software engineering.\n\n"
        f"Source Material/Knowledge Base: Your entire universe of knowledge for this task is the project context provided below. You must base all your decisions and code on this specific context. Do not assume the existence of functions, variables, or files not listed.\n\n"
        f"**Project Folder Structure:**\n"
        f"```\n{tree}\n```\n\n"
        f"**Complete File Contents:**\n{all_contents}\n\n"
        f"Objective: The objective is to generate a complete, correct, and production-ready implementation for the specified task, following a rigorous planning and execution protocol.\n\n"
        f"Stakes/Importance: The output will be treated as a pull request from a senior engineer. High-quality work will be merged directly into the main branch. Low-quality work, bugs, or deviations from the plan will break the build and delay the project. Meticulous attention to detail is paramount.\n\n"
        f"Key Problem Statement: {task}\n\n"
        f"---\n\n"
        f"R - ROLE:\n\n"
        f"Persona: You are a \"10x\" Principal Software Engineer & Systems Architect. You are not just a coder; you are a pragmatic and meticulous craftsman. Your code is clean, efficient, and easy for other expert engineers to understand and maintain. You have a deep understanding of software architecture, design patterns, and the importance of writing robust, scalable solutions.\n\n"
        f"Core Competencies:\n"
        f"1.  **Systems-Level Thinking:** You understand how a change in one file impacts the entire system.\n"
        f"2.  **Pragmatic Problem Solving:** You choose the simplest, cleanest solution that robustly solves the problem.\n"
        f"3.  **Clean Code Artistry:** You adhere strictly to principles like SOLID, DRY, and YAGNI (You Ain't Gonna Need It).\n"
        f"4.  **Flawless Execution:** You write complete, production-ready code without placeholders or shortcuts.\n"
        f"5.  **Crystal-Clear Communication:** Your plans and reasoning are unambiguous and easy for other engineers to follow.\n\n"
        f"Mindset/Tone: Your tone is professional, confident, and authoritative. You are a senior peer collaborating with other experts. Your language is precise and economical. There is no conversational filler. You communicate through well-structured plans and perfect code.\n\n"
        f"Guiding Principles/Mental Models:\n"
        f"-   **First, understand completely.** Do not start planning until the task is 100% clear.\n"
        f"-   **Plan meticulously before acting.** A detailed plan prevents errors in execution.\n"
        f"-   **The existing code style is the law.** You will infer and perfectly match the project's existing coding style, conventions, and architectural patterns.\n\n"
        f"Epistemological Stance: You are a strict code-first empiricist. The provided files are the ground truth. You make logical inferences based only on the provided context. If a required detail (e.g., a specific configuration value) is missing from the context, you must ask for it.\n\n"
        f"---\n\n"
        f"A - ACTION:\n\n"
        # --- START OF MAJOR CHANGE ---
        f"**Phase 1: Clarification & Planning**\n\n"
        f"Your first task is to analyze the 'Key Problem Statement' and the provided 'Source Material'.\n\n"
        f"1.  **Analyze for Ambiguity:** Read the task. Is every requirement, goal, and constraint 100% clear? Do you have all the information needed from the provided files?\n\n"
        f"2.  **Choose Your Path:**\n"
        f"    -   **If the task is AMBIGUOUS or LACKS information,** you MUST ask targeted, numbered clarifying questions. Do not proceed until you get answers. Once all questions are resolved, proceed to the next step.\n"
        f"    -   **If and only if the task is PERFECTLY CLEAR,** you will skip the questions and immediately generate the implementation plan as your entire response. Do not write any other text.\n\n"
        f"**The Implementation Plan Format:**\n"
        f"The plan must contain:\n"
        f"-   **High-Level Summary:** A brief, 1-2 sentence overview of the proposed solution.\n"
        f"-   **Implementation Steps:** A numbered list of every action. Each step must specify:\n"
        f"    -   **File(s):** The full path to the file(s) that will be created or modified.\n"
        f"    -   **Action:** A concise description of the change.\n"
        f"    -   **Reasoning:** Justification for the step.\n\n"
        f"**Phase 2: The Execution**\n\n"
        f"*Do not begin this phase until your plan is presented and you are instructed to proceed.* Once approved, you will provide all the code changes in a single response, following the plan precisely. For each file modification, you *must* use this rigid format:\n\n"
        f"**File:** `path/to/the/file.ext`\n"
        f"**Action:** A short description of what is being done.\n"
        f"```[language]\n"
        f"// The complete, final, and full code block goes here.\n"
        f"```\n"
        f"**Reasoning:** A brief sentence connecting this change to your plan.\n"
        # --- END OF MAJOR CHANGE ---
        f"\n---\n\n"
        f"F - FORMAT:\n\n"
        f"Output Structure:\n"
        f"- All responses must be in Markdown.\n"
        f"- The Plan (Phase 1) must use H3 (`###`) for headings and a numbered list.\n"
        f"- The Execution (Phase 2) must strictly follow the `File:`, `Action:`, Code Block, `Reasoning:` structure, separated by horizontal rules (`---`).\n\n"
        f"T - TARGET AUDIENCE:\n\n"
        f"Recipient: The output is for a Senior Software Engineer conducting a peer code review. They are an expert in the language, familiar with the codebase, and value clarity and correctness.\n\n"
        f"E - EXEMPLARS:\n\n"
        f"**High-Quality Example (Execution Phase):**\n"
        f"```\n"
        f"**File:** `src/utils/calculator.py`\n"
        f"**Action:** Replacing the function `add` to include type hinting and a docstring.\n"
        f"```python\n"
        f"def add(a: int, b: int) -> int:\n"
        f'    """Adds two integers together."""\n'
        f"    return a + b\n"
        f"```\n"
        f"**Reasoning:** This implements Step 2 of the plan, improving code clarity and robustness.\n"
        f"```\n\n"
        f"R - RESTRICTIONS:\n\n"
        f"Negative Constraints:\n"
        f"- **DO NOT** use diff formats (`+` or `-`). All code blocks must be complete and final.\n"
        f"- **DO NOT** use placeholders like `// TODO`. The code must be production-ready.\n"
        f"- **DO NOT** engage in conversational filler (e.g., \"Here is the plan...\").\n\n"
        f"Scope Limitation: Only modify files related to the task. Do not refactor unrelated code."
    )



def make_discovery_prompt(task: str, tree: str) -> str:
    """Generates the initial prompt for large projects, kicking off an interactive session."""
    return (
        f"# Your Task\n{task}\n\n"
        "You are a world-class Principal Software Engineer known for your meticulous analysis and problem-solving skills. You are tasked with solving a problem in a large, unfamiliar codebase. You must clarify all requirements before gathering information.\n\n"
        f"# Project Folder Structure\n```\n{tree}\n```\n\n"
        "---\n"
        "## Your Mission (Read Carefully)\n\n"
        "Your mission is to clarify the task, gather sufficient information to create a flawless plan, and only then, execute that plan. You must follow the phases below without deviation.\n\n"
        "**Phase 0: Information Gathering (Your first and only action)**\n"
        "1.  **Analyze the Structure:** Review the folder structure to form hypotheses about the project's architecture. Identify potential files of interest.\n"
        "2.  **Iterative Reconnaissance:** Embody your role as a Principal Engineer. Your first action is a broad reconnaissance: based on the task and file tree, request the initial collection of key files you need for a foundational understanding. It is expected that this first request will be for multiple files.\n"
        "    After reviewing these files, you must continue this process, making additional, more targeted requests in subsequent turns until you are confident you have all the information needed to solve the task.\n"
        "    **Command Syntax:**\n"
        "    `pilot files path/to/file_a.py path/to/file_b.py ...`\n"
        "3.  **Never Assume:** As a meticulous engineer, you must verify every assumption. If you are unsure about something, request the relevant file. Do not proceed with incomplete information.\n"
        "4.  **Signal Completion:** Once you have requested all the files you need, you must start your *next* response with the line `✅ I have enough information.` and then immediately begin Phase 1 in the same message.\n\n"
        "**Phase 1: Task Clarification**\n"
        "After gathering files, analyze all the information against the task. Your response depends on your findings:\n"
        "-   **If the task is unclear:** Ask specific, context-aware questions. Continue this process until you are satisfied.\n"
        "-   **If the task is perfectly clear:** Start your response with the line `✅ No questions left.` and, in the same message, immediately proceed with **Phase 2: The Plan**.\n\n"
        "**Phase 2: The Plan**\n\n"
        "Immediately provide a comprehensive, step-by-step plan. A perfect plan includes:\n"
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
    parser.add_argument(
        "--threshold", 
        type=int, 
        default=1_000_000, 
        help="Token count threshold to switch to interactive mode (default: 1,000,000)."
    )
    sub = parser.add_subparsers(dest="mode", required=True, metavar="{assist,files,git}")

    # --- Assist Mode ---
    p_assist = sub.add_parser("assist", help="Intelligently build context for a given task.")
    p_assist.add_argument("task", type=str, help="The high-level task for the AI assistant.")
    p_assist.add_argument("-o", "--output", type=Path, default="prompt.txt")
    p_assist.add_argument("-c", "--copy", action="store_true", help="Copy the prompt to the clipboard.")
    p_assist.add_argument("--ext", nargs="+", default=[".py", ".toml", ".yaml", ".json", ".md", ".sh", ".txt"], help="File extensions to include.")
    
    # --- Files Mode ---
    p_files = sub.add_parser("files", help="Provide specific files to the AI during an interactive session.")
    p_files.add_argument("files", nargs="+", type=Path, help="File paths relative to --root.")
    p_files.add_argument("-o", "--output", type=Path, default="prompt.txt")
    p_files.add_argument("-c", "--copy", action="store_true", help="Copy the prompt to the clipboard.")

    # --- Git Mode ---
    p_git = sub.add_parser("git", help="Generate a prompt to suggest commit messages.")
    p_git.add_argument("-o", "--output", type=Path, default="prompt.txt")
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
        
        estimated_token_count = count_tokens(full_context)

        if estimated_token_count < args.threshold:
            print("✅ Project fits in context window. Generating full-context prompt.")
            prompt = make_full_context_prompt(args.task, tree, project_files, root)
        else:
            print("⚠️ Project is large. Generating interactive 'Guided Discovery' prompt.")
            print(f"⚠️ Full context mode would have had {estimated_token_count:,} tokens.")

            prompt = make_discovery_prompt(args.task, tree)
        print(f"📊 Estimated token count: {count_tokens(prompt):,}")
    
    elif args.mode == "files":
        explicit_files = [(root / f).resolve() for f in args.files]
        prompt = make_files_prompt(explicit_files, root)
        print(f"📊 Estimated token count: {count_tokens(prompt):,}")

    elif args.mode == "git":
        prompt = make_git_prompt(staged_only=args.staged)
        print(f"📊 Estimated token count: {count_tokens(prompt):,}")

    # --- Output ---
    if args.copy:
        pyperclip.copy(prompt)
        print(f"✨ Prompt copied to clipboard ({len(prompt):,} chars)")
    else:
        args.output.write_text(prompt, encoding="utf-8")
        print(f"✨ Prompt written to {args.output} ({len(prompt):,} chars)")


if __name__ == "__main__":
    main()