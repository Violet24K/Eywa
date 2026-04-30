"""Filesystem and port conventions for the Eywa project."""

import os
import os.path as osp


_current_dir = os.path.dirname(os.path.abspath(__file__))
_eywa_dir = os.path.dirname(_current_dir)
root_dir = os.path.dirname(_eywa_dir)

benchmark_dir = osp.join(root_dir, "eywabench")
prompt_dir = osp.join(root_dir, "prompts")
exp_dir = osp.join(root_dir, "experiments")
log_dir = osp.join(root_dir, "logs")
config_dir = osp.join(_eywa_dir, "configs")


# Per-worker MCP layout: worker w uses ports [base + 2w, base + 2w + 1].
# Slot 0 -> chronos, slot 1 -> tabpfn.
mcp_base_port = 8000
_FM_PORT_SLOT = {"chronos": 0, "tabpfn": 1}


def get_fm_port(fm: str, worker_id: int = 0) -> int:
    """Return the MCP server port assigned to ``fm`` for ``worker_id``."""
    if fm not in _FM_PORT_SLOT:
        raise ValueError(f"Unknown foundation model: {fm!r}.")
    return mcp_base_port + 2 * worker_id + _FM_PORT_SLOT[fm]


# Backward-compatible aliases for worker 0.
chronos_mcp_server_port = get_fm_port("chronos", 0)
tabpfn_mcp_server_port = get_fm_port("tabpfn", 0)
