import subprocess
import sys
import os
import curses


def load_env(path=".env"):
    env = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip()
                if (v.startswith('"') and v.endswith('"')) or (
                    v.startswith("'") and v.endswith("'")
                ):
                    v = v[1:-1]
                env[k.strip()] = v
    return env


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV = load_env(os.path.join(SCRIPT_DIR, ".env"))


def env_get(key, default=None, required=False):
    if key in ENV and ENV[key] != "":
        return ENV[key]
    if required and default is None:
        print(f"Error: Missing required key '{key}' in .env", file=sys.stderr)
        sys.exit(1)
    return default


def detect_env():
    # Check for active conda env
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        return conda_prefix

    # Check for active venv / virtualenv
    virtual_env = os.environ.get("VIRTUAL_ENV", "")
    if virtual_env:
        return virtual_env

    print(
        "Error: No active conda or virtual environment detected.\n"
        "Activate an environment before running this script, or set 'env-path' in .env.",
        file=sys.stderr,
    )
    sys.exit(1)


def get_slurm_vars():
    return {
        "env-path": env_get("env-path", detect_env()),
        "account": env_get("account", required=True),
        "partition": env_get("partition", "gpu-exp"),
        "gres": env_get("gres", "gpu:1"),
        "cpus-per-task": env_get("cpus-per-task", "8"),
        "mem": env_get("mem", "64G"),
        "time": env_get("time", "48:00:00"),
        "cuda-module": env_get("cuda-module", "cuda/12.6.2"),
        "conda-module": env_get("conda-module", "miniconda3/24.1.2-py310"),
    }


def get_model_files():
    scripts_dir = "scripts"
    if not os.path.isdir(scripts_dir):
        print(f"Error: '{scripts_dir}/' directory not found.", file=sys.stderr)
        sys.exit(1)
    files = sorted(f for f in os.listdir(scripts_dir) if f.endswith(".py"))
    if not files:
        print(f"Error: No .py files found in '{scripts_dir}/'.", file=sys.stderr)
        sys.exit(1)
    return files


def select_files_tui(files, slurm_vars):
    def _run(stdscr):
        curses.curs_set(0)
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_CYAN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)

        selected = [False] * len(files)
        cursor = 0
        scroll_offset = 0

        # Batch variables to display (exclude internal-only keys)
        display_vars = {
            k: v for k, v in slurm_vars.items()
            if k not in ("env-path", "cuda-module", "conda-module")
        }

        while True:
            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx()

            # -- Header --
            row = 0
            stdscr.attron(curses.A_BOLD)
            stdscr.addstr(row, 0, "QuickBatch — SLURM Job Submitter")
            stdscr.attroff(curses.A_BOLD)
            row += 1
            stdscr.addstr(row, 0, "─" * min(max_x - 1, 70))
            row += 1

            # -- Batch variables panel --
            stdscr.attron(curses.color_pair(3) | curses.A_BOLD)
            stdscr.addstr(row, 0, " Batch Configuration:")
            stdscr.attroff(curses.color_pair(3) | curses.A_BOLD)
            row += 1

            col1_items = list(display_vars.items())[: len(display_vars) // 2 + 1]
            col2_items = list(display_vars.items())[len(display_vars) // 2 + 1 :]
            col2_offset = 38

            for i, (k, v) in enumerate(col1_items):
                label = f"  {k}: "
                stdscr.attron(curses.A_DIM)
                stdscr.addstr(row + i, 0, label[:max_x - 1])
                stdscr.attroff(curses.A_DIM)
                stdscr.addstr(row + i, len(label), v[: max_x - len(label) - 1])

            for i, (k, v) in enumerate(col2_items):
                label = f"  {k}: "
                x = col2_offset
                if x + len(label) + len(v) < max_x:
                    stdscr.attron(curses.A_DIM)
                    stdscr.addstr(row + i, x, label)
                    stdscr.attroff(curses.A_DIM)
                    stdscr.addstr(row + i, x + len(label), v)

            row += max(len(col1_items), len(col2_items)) + 1
            stdscr.addstr(row, 0, "─" * min(max_x - 1, 70))
            row += 1

            # -- Instructions --
            stdscr.addstr(
                row, 0,
                "↑/↓: navigate | SPACE: toggle | a: all | n: none | ENTER: submit | q: quit"
            )
            row += 1

            header_lines = row
            visible = max_y - header_lines - 2

            # -- File list --
            if cursor < scroll_offset:
                scroll_offset = cursor
            elif cursor >= scroll_offset + visible:
                scroll_offset = cursor - visible + 1

            for i in range(visible):
                idx = scroll_offset + i
                if idx >= len(files):
                    break

                r = header_lines + i
                check = "[x]" if selected[idx] else "[ ]"
                line = f" {check} {files[idx]}"

                if idx == cursor:
                    stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                    stdscr.addstr(r, 0, line[: max_x - 1])
                    stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
                elif selected[idx]:
                    stdscr.attron(curses.color_pair(1))
                    stdscr.addstr(r, 0, line[: max_x - 1])
                    stdscr.attroff(curses.color_pair(1))
                else:
                    stdscr.addstr(r, 0, line[: max_x - 1])

            count = sum(selected)
            status = f"  {count} file(s) selected"
            stdscr.addstr(max_y - 1, 0, status[: max_x - 1])
            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                cursor = max(0, cursor - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                cursor = min(len(files) - 1, cursor + 1)
            elif key == ord(" "):
                selected[cursor] = not selected[cursor]
                cursor = min(len(files) - 1, cursor + 1)
            elif key == ord("a"):
                for i in range(len(selected)):
                    selected[i] = True
            elif key == ord("n"):
                for i in range(len(selected)):
                    selected[i] = False
            elif key in (curses.KEY_ENTER, 10, 13):
                if any(selected):
                    return [files[i] for i, s in enumerate(selected) if s]
                stdscr.addstr(
                    max_y - 1, 0, "  ⚠ Select at least one file!" + " " * 20
                )
                stdscr.refresh()
                curses.napms(1000)
            elif key == ord("q"):
                return []

    return curses.wrapper(_run)


def submit_job(job_name, command, slurm_vars):
    os.makedirs(f"logs/{job_name}", exist_ok=True)

    slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={slurm_vars["time"]}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={slurm_vars["cpus-per-task"]}
#SBATCH --mem={slurm_vars["mem"]}
#SBATCH --account={slurm_vars["account"]}
#SBATCH --partition={slurm_vars["partition"]}
#SBATCH --gres={slurm_vars["gres"]}
#SBATCH --output=logs/{job_name}/%j.out
#SBATCH --error=logs/{job_name}/%j.err

module purge
module load {slurm_vars["conda-module"]}
module load {slurm_vars["cuda-module"]}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "{slurm_vars["env-path"]}"


python {command}
"""

    try:
        result = subprocess.run(
            ["sbatch"],
            input=slurm_script_content,
            text=True,
            capture_output=True,
            check=True,
        )
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job '{job_name}': {e.stderr}", file=sys.stderr)
    except FileNotFoundError:
        print(
            "Error: 'sbatch' command not found. Are you on a Slurm cluster?",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    slurm_vars = get_slurm_vars()
    files = get_model_files()
    selected = select_files_tui(files, slurm_vars)

    if not selected:
        print("No files selected. Exiting.")
        sys.exit(0)

    print(f"\nSubmitting {len(selected)} job(s)...\n")
    for filename in selected:
        job_name = filename.removesuffix(".py")
        command = f"scripts/{filename}"
        print(f"  → {job_name}: {command}")
        submit_job(job_name, command, slurm_vars)

    print("\nAll jobs submitted.")


if __name__ == "__main__":
    main()
