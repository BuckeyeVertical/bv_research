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
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                env[k.strip()] = v
    return env


ENV = load_env()


def env_get(key, default=None, required=False):
    if key in ENV and ENV[key] != "":
        return ENV[key]
    if required and default is None:
        print(f"Error: Missing required key '{key}' in .env", file=sys.stderr)
        sys.exit(1)
    return default


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


def select_files_tui(files):
    def _run(stdscr):
        curses.curs_set(0)
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_CYAN, -1)

        selected = [False] * len(files)
        cursor = 0
        scroll_offset = 0

        while True:
            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx()
            header_lines = 3
            visible = max_y - header_lines - 2

            stdscr.attron(curses.A_BOLD)
            stdscr.addstr(0, 0, "Select script(s) to submit to SLURM")
            stdscr.attroff(curses.A_BOLD)
            stdscr.addstr(1, 0, "↑/↓: navigate | SPACE: toggle | a: all | n: none | ENTER: submit | q: quit")
            stdscr.addstr(2, 0, "─" * min(max_x - 1, 70))

            if cursor < scroll_offset:
                scroll_offset = cursor
            elif cursor >= scroll_offset + visible:
                scroll_offset = cursor - visible + 1

            for i in range(visible):
                idx = scroll_offset + i
                if idx >= len(files):
                    break

                row = header_lines + i
                check = "[x]" if selected[idx] else "[ ]"
                line = f" {check} {files[idx]}"

                if idx == cursor:
                    stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                    stdscr.addstr(row, 0, line[:max_x - 1])
                    stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
                elif selected[idx]:
                    stdscr.attron(curses.color_pair(1))
                    stdscr.addstr(row, 0, line[:max_x - 1])
                    stdscr.attroff(curses.color_pair(1))
                else:
                    stdscr.addstr(row, 0, line[:max_x - 1])

            count = sum(selected)
            status = f"  {count} file(s) selected"
            stdscr.addstr(max_y - 1, 0, status[:max_x - 1])
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
                stdscr.addstr(max_y - 1, 0, "  ⚠ Select at least one file!" + " " * 20)
                stdscr.refresh()
                curses.napms(1000)
            elif key == ord("q"):
                return []

    return curses.wrapper(_run)


def submit_job(job_name, command):
    env_path = env_get("ENV_PATH", required=True)
    account = env_get("ACCOUNT", required=True)
    partition = env_get("PARTITION", "gpu-exp")
    gpus = env_get("GPUS", "1")
    cpus = env_get("CPUS", "8")
    mem = env_get("MEM", "64G")
    time = env_get("TIME", "48:00:00")
    cuda_module = env_get("CUDA_MODULE", "cuda/12.6.2")
    conda_module = env_get("CONDA_MODULE", "miniconda3/24.1.2-py310")


    os.makedirs(f"logs/{job_name}", exist_ok=True)

    slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --output=logs/{job_name}/%j.out
#SBATCH --error=logs/{job_name}/%j.err

module purge
module load {conda_module}
module load {cuda_module}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "{env_path}"


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
        print("Error: 'sbatch' command not found. Are you on a Slurm cluster?", file=sys.stderr)
        sys.exit(1)


def main():
    files = get_model_files()
    selected = select_files_tui(files)

    if not selected:
        print("No files selected. Exiting.")
        sys.exit(0)

    print(f"\nSubmitting {len(selected)} job(s)...\n")
    for filename in selected:
        job_name = filename.removesuffix(".py")
        command = f"scripts/{filename}"
        print(f"  → {job_name}: {command}")
        submit_job(job_name, command)

    print("\nAll jobs submitted.")


if __name__ == "__main__":
    main()