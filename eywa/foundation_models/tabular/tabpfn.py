"""TabPFN as an MCP server.

This module exposes TabPFN's classifier/regressor as a tool over the Model
Context Protocol so an Eywa LLM agent can call it with no parameters: the
data and task type are loaded into ``TabularStorage`` ahead of time via the
internal ``load_tabular_data`` tool.
"""

import argparse

from fastmcp import FastMCP
from tabpfn import TabPFNClassifier, TabPFNRegressor

from eywa.foundation_models.tabular.common import (
    TabularStorage,
    create_load_tabular_data_tool,
)
from eywa.utils.path import tabpfn_mcp_server_port


tabular_storage = TabularStorage()
mcp = FastMCP("tabpfn_mcp_server")
create_load_tabular_data_tool(mcp, tabular_storage)


@mcp.tool()
def tabpfn() -> str:
    """Run TabPFN inference on the previously loaded table.

    The function takes no arguments — task type, output size, and the
    feature matrix are read from ``TabularStorage``. Returns the predicted
    target column as a Python list serialized to a string.
    """
    output_size = int(tabular_storage.output_size)
    df = tabular_storage.dataframe
    X_train = df.iloc[:-output_size, :-1]
    y_train = df.iloc[:-output_size, -1]
    X_test = df.iloc[-output_size:, :-1]

    if tabular_storage.task_type == "tabular-classification":
        model = TabPFNClassifier()
    elif tabular_storage.task_type == "tabular-regression":
        model = TabPFNRegressor()
    else:
        raise ValueError(
            f"Unknown task type: {tabular_storage.task_type!r}. "
            "Expected 'tabular-classification' or 'tabular-regression'."
        )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test).tolist()
    return str(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=tabpfn_mcp_server_port)
    cli_args = parser.parse_args()
    mcp.run(transport="streamable-http", port=cli_args.port)
