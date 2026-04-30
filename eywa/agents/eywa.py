"""EywaAgent: an LLM agent backed by a domain foundation model exposed via MCP."""

import socket

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from eywa.agents.base_agent import initialize_model
from eywa.utils.langchain_mcp import invoke_tool
from eywa.utils.path import get_fm_port


# Foundation models bundled with Eywa and the task types they support.
FM_SUPPORTED_TASKS: dict[str, list[str]] = {
    "chronos": ["time-series-forecasting"],
    "tabpfn": ["tabular-classification", "tabular-regression"],
}

# Tools that load task data into the MCP server's internal storage. These are
# called once at agent initialization time and should never be exposed to the
# planner LLM.
_INTERNAL_TOOL_NAMES: frozenset[str] = frozenset(
    {"load_time_series_data", "load_tabular_data"}
)


def _check_server_running(port: int, host: str = "localhost", timeout: float = 1.0) -> bool:
    """Return ``True`` if a TCP server is accepting connections on ``port``."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


class EywaAgent:
    """An LLM planner that can call a domain foundation model through MCP.

    Use :meth:`EywaAgent.create` to construct one — it asynchronously connects
    to the foundation model's MCP server and discovers the available tools.
    """

    def __init__(self, model_name: str, tools: list[BaseTool], fm_name: str):
        self.model = model_name
        self.fm_name = fm_name
        self.internal_tools = [t for t in tools if t.name in _INTERNAL_TOOL_NAMES]
        self.tools = [t for t in tools if t.name not in _INTERNAL_TOOL_NAMES]
        self.llm = initialize_model(model_name)
        self.agent = create_agent(model=self.llm, tools=self.tools)
        print(f"EywaAgent initialized with model {model_name}")

    @classmethod
    async def create(
        cls,
        model: str = "gpt-5-nano",
        fm: str | None = None,
        worker_id: int = 0,
    ) -> "EywaAgent":
        """Connect to ``fm``'s MCP server (for the given ``worker_id``) and build an agent."""
        if fm is None:
            raise ValueError("Foundation model is required.")
        port = get_fm_port(fm, worker_id)
        mcp_server_url = f"http://localhost:{port}/mcp"
        if not _check_server_running(port):
            raise RuntimeError(
                f"{fm} MCP server is not running at {mcp_server_url} (worker {worker_id})."
            )
        client = MultiServerMCPClient(
            {"EywaTimeSeries": {"transport": "http", "url": mcp_server_url}}
        )
        tools = await client.get_tools()
        return cls(model_name=model, tools=tools, fm_name=fm)

    async def invoke(self, prompt: str):
        """Run a single inference pass over ``prompt``."""
        print("Running Eywa inference.")
        return await self.agent.ainvoke({"messages": prompt})

    async def update_data(self, task: str, new_data: dict) -> None:
        """Push the current task sample into the foundation model's MCP storage.

        Args:
            task: Task type, e.g. ``time-series-forecasting`` or ``tabular-classification``.
            new_data: A dictionary form of the benchmark row (``task_sample.to_dict()``).
        """
        if task not in FM_SUPPORTED_TASKS[self.fm_name]:
            print(
                f"Task {task!r} is not supported by {self.fm_name!r}; "
                f"skipping data update."
            )
            return
        if not self.internal_tools:
            raise ValueError("No internal data-loading tools found from MCP server.")

        if task == "time-series-forecasting":
            await invoke_tool(
                self.internal_tools,
                "load_time_series_data",
                {"time_series_task_sample": new_data},
            )
        elif task.startswith("tabular-"):
            await invoke_tool(
                self.internal_tools,
                "load_tabular_data",
                {"tabular_task_sample": new_data},
            )
        else:
            raise ValueError(
                f"Unknown task type: {task!r}. "
                "Expected 'time-series-forecasting' or 'tabular-*'."
            )
