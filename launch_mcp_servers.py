"""Launch one MCP server per (worker, foundation model) slot.

Each worker gets its own pair of servers (one per FM) on consecutive ports
starting from ``mcp_base_port``. Press Ctrl+C to stop them all.
"""

import argparse
import os
import os.path as osp
import subprocess
import sys
import time
from io import TextIOWrapper

from eywa.utils.path import get_fm_port, log_dir


_FM_MODULE_NAME = "eywa.foundation_models"
_MCP_SERVERS = {"time_series": ["chronos"], "tabular": ["tabpfn"]}


def _launch_servers(num_workers: int, exp_log_dir: str):
    processes: dict[subprocess.Popen, str] = {}
    log_files: list[TextIOWrapper] = []

    for worker_id in range(num_workers):
        for domain, servers in _MCP_SERVERS.items():
            for server in servers:
                module_name = f"{_FM_MODULE_NAME}.{domain}.{server}"
                port = get_fm_port(server, worker_id)
                log_path = osp.join(exp_log_dir, f"{server}_w{worker_id}.log")
                log_file = open(log_path, "w", encoding="utf-8")
                log_files.append(log_file)
                process = subprocess.Popen(
                    ["python", "-m", module_name, "--port", str(port)],
                    stdout=log_file,
                    stderr=log_file,
                )
                processes[process] = (
                    f"worker{worker_id}_{domain}_{server}_port{port}"
                )
                print(
                    f"Launching {server} MCP server (worker {worker_id}) on port {port}; "
                    f"logs at {log_path}."
                )
    return processes, log_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", type=str, default="test",
                        help="Subfolder of logs/ for per-server log files.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of worker slots; each gets a chronos + tabpfn server.")
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError(f"--num_workers must be >= 1, got {args.num_workers}.")

    exp_log_dir = osp.join(log_dir, args.exp_name)
    os.makedirs(exp_log_dir, exist_ok=True)

    processes, log_files = _launch_servers(args.num_workers, exp_log_dir)

    try:
        while True:
            for process, name in processes.items():
                if process.poll() is not None:
                    print(f"Server {name} crashed.")
            time.sleep(1)
    except KeyboardInterrupt:
        for f in log_files:
            f.close()
        for process in processes:
            process.terminate()
        print("All servers stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
