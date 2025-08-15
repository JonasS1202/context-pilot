\<div align="center"\>

# 🚀 ContextPilot

**An intelligent context manager for AI assistants.**

\</div\>

\<p align="center"\>
\<img src="[https://img.shields.io/badge/Python-3.7+-blue.svg](https://www.google.com/search?q=https://img.shields.io/badge/Python-3.7%2B-blue.svg)" alt="Python Version"\>
\<img src="[https://img.shields.io/badge/License-MIT-green.svg](https://www.google.com/search?q=https://img.shields.io/badge/License-MIT-green.svg)" alt="License"\>
\<img src="[https://img.shields.io/badge/Version-1.0.0-informational.svg](https://www.google.com/search?q=https://img.shields.io/badge/Version-1.0.0-informational.svg)" alt="Version"\>
\</p\>

`ContextPilot` is a command-line tool that bridges the gap between your local development environment and your AI assistant. It intelligently gathers and formats project context, ensuring your AI has the right information to perform complex coding tasks efficiently.

Stop manually copying and pasting files\! Let `ContextPilot` build the perfect prompt for you.

-----

## ✨ Key Features

  * **🤖 Smart Context Strategy**: Automatically detects project size. It provides the full project context for small projects and initiates an interactive "Guided Discovery" session for large ones to avoid token limits.
  * **🧠 Intelligent Filtering**: Respects your project's `.gitignore` file and uses sensible defaults to exclude irrelevant files and directories (like `venv`, `.git`, `__pycache__`).
  * **📝 Structured Prompting**: Generates sophisticated, multi-phased prompts that instruct the AI to clarify requirements, create a plan, and then execute it, leading to higher-quality results.
  * **💬 Interactive Workflow**: In "Guided Discovery" mode, the AI can request specific files, allowing you to iteratively build the context it needs for large or unfamiliar codebases.
  * **Git Integration**: Includes a `git` command that creates a prompt from your staged or unstaged changes to generate conventional commit messages.
  * **📋 Clipboard & File Output**: Easily copy the generated prompt to your clipboard or save it to a file.

-----

## 🛠️ Installation

You can install `ContextPilot` directly from the GitHub repository. You will also need to install its dependencies, `tiktoken` and `pyperclip`.

```bash
# Install the tool and its dependencies in one go
"pip install git+https://github.com/JonasS1202/context-pilot.git"
```

*Note: Ensure Git is installed and accessible in your system's PATH.*

-----

## Usage

`ContextPilot` has three main commands designed for different scenarios.

### 1\. The `assist` Command (Primary)

This is your main entry point. It analyzes your project and builds the most effective prompt for a given task.

**Usage:**

```bash
pilot assist "Your high-level task description goes here." [options]
```

**Example:**

```bash
pilot assist "Refactor the authentication logic to use a new service class." --ext .py .toml
```

Based on your project's total token count, this command will generate one of two prompts:

1.  **Full-Context Prompt**: If the project is small enough, it includes the project tree and the full content of all relevant files.
2.  **Discovery Prompt**: If the project is large, it provides only the file tree and instructs the AI to request the specific files it needs to begin its analysis.

### 2\. The `files` Command (Interactive)

This command is used during the "Guided Discovery" workflow for large projects. After you run `assist` and the AI responds by asking for specific files, you use this command to provide them.

**Usage:**

```bash
pilot files <path/to/file1.py> <path/to/file2.py> ...
```

**Example:**
Imagine the AI responds with: `pilot files src/auth/main.py src/models/user.py`
You would then run that exact command in your terminal:

```bash
pilot files src/auth/main.py src/models/user.py --copy
```

This generates a new prompt containing the content of the requested files, which you then paste back to the AI.

### 3\. The `git` Command (Commit Messages)

This command uses your `git diff` to generate a prompt for creating conventional commit messages. It's a great way to summarize your work.

**Usage:**

```bash
# Analyze all staged and unstaged changes
pilot git

# Analyze only staged changes
pilot git --staged
```

-----

## ⚙️ Workflow Examples

### Small Project Workflow (Full Context)

For smaller projects, the process is simple and direct.

1.  **You**: Run the `assist` command with your task.
    ```bash
    pilot assist "Add a health check endpoint to the API" -c
    ```
2.  **AI**: Receives the full project context and your task. It will then:
      * Ask clarifying questions or confirm it understands.
      * Provide a step-by-step implementation plan.
      * Execute the plan by providing complete, final code for each modified file.

### Large Project Workflow (Guided Discovery)

For larger projects, `ContextPilot` initiates an interactive session to build context gradually.

1.  **You**: Run the `assist` command. `ContextPilot` detects the project is large and generates a discovery prompt.

    ```bash
    pilot assist "Integrate a new payment gateway into the checkout process" -c
    ```

2.  **AI**: Receives the project's file tree and your task. It analyzes the structure and requests the most relevant files to start.

    > **AI's Response:** Based on the file tree, I need to see the main checkout controller and the existing payment service. Please provide them using the `pilot` command:
    > `pilot files src/controllers/checkout.py src/services/payment_provider.py`

3.  **You**: Run the command the AI provided in your terminal to gather the requested files and copy them to your clipboard.

    ```bash
    pilot files src/controllers/checkout.py src/services/payment_provider.py -c
    ```

4.  **AI**: Receives the content of the requested files. It may now have enough information to create a plan, or it might ask for more files, continuing the cycle until it's confident it can complete the task.

This interactive process ensures the AI gets exactly the information it needs without overwhelming its context window.