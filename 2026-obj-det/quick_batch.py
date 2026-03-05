import subprocess
import sys
import os
import curses

# --- SLURM defaults ---
DEFAULTS = {
    "time": "48:00:00",
    "nodes": "1",
    "ntasks_per_node": "1",
    "gpus_per_node": "1",
    "cpus_per_task": "8",
    "mem": "64gb",
    "account": "PAS2152",
    "partition": "",
    "cuda_module": "cuda/12.6.2",
    "scripts_dir": "scripts",
    "logs_dir": "logs",
}

# --- Active configuration (env vars override defaults) ---
cfg = {k: os.getenv(k, v) for k, v in DEFAULTS.items()}


def get_model_files():
    scripts_dir = cfg["scripts_dir"]
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

            stdscr.attron(curses.A_BOLD)
            stdscr.addstr(0, 0, "Select script(s) to submit to SLURM")
            stdscr.attroff(curses.A_BOLD)
            stdscr.addstr(1, 0, "↑/↓: navigate | SPACE: toggle | a: all | n: none | r: reset slurm settings | ENTER: submit | q: quit")
            stdscr.addstr(2, 0, "─" * min(max_x - 1, 70))

            settings = [
                f"time={cfg['time']}  nodes={cfg['nodes']}  ntasks={cfg['ntasks_per_node']}  gpus={cfg['gpus_per_node']}",
                f"cpus={cfg['cpus_per_task']}  mem={cfg['mem']}  account={cfg['account']}  cuda={cfg['cuda_module']}",
            ]
            if cfg["partition"]:
                settings[1] += f"  partition={cfg['partition']}"

            for i, line in enumerate(settings):
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(3 + i, 2, line[:max_x - 3])
                stdscr.attroff(curses.color_pair(1))

            header_lines = 3 + len(settings) + 1
            visible = max_y - header_lines - 2

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
            elif key == ord("r"):
                cfg.update(DEFAULTS)
                stdscr.addstr(max_y - 1, 0, "  ⟳ Reset to defaults" + " " * 20)
                stdscr.refresh()
                curses.napms(500)
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
    # Virtual environment detection
    if "VIRTUAL_ENV" in os.environ:
        activation_cmd = f"source {os.environ['VIRTUAL_ENV']}/bin/activate"
    elif "CONDA_DEFAULT_ENV" in os.environ:
        activation_cmd = f"source activate {os.environ['CONDA_DEFAULT_ENV']}"
    else:
        activation_cmd = "# No virtual environment detected"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --time={cfg['time']}",
        f"#SBATCH --nodes={cfg['nodes']}",
        f"#SBATCH --ntasks-per-node={cfg['ntasks_per_node']}",
        f"#SBATCH --gpus-per-node={cfg['gpus_per_node']}",
        f"#SBATCH --cpus-per-task={cfg['cpus_per_task']}",
        f"#SBATCH --mem={cfg['mem']}",
        f"#SBATCH --account={cfg['account']}",
        f"#SBATCH --output={cfg['logs_dir']}/{job_name}/%j/out.out",
        f"#SBATCH --error={cfg['logs_dir']}/{job_name}/%j/err.err",
        f"#SBATCH --partition={cfg['partition']}" if cfg["partition"] else "",
        "",
        f"module load {cfg['cuda_module']}",
        activation_cmd,
        f"python {command}",
    ]

    slurm_script_content = "\n".join(lines) + "\n"

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
        command = f"{cfg['scripts_dir']}/{filename}"
        print(f"  → {job_name}: {command}")
        submit_job(job_name, command)

    print("\nAll jobs submitted.")


if __name__ == "__main__":
    main()
