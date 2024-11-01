import json
from typing import Dict, List
import os
import pandas as pd
from collections import defaultdict
import time
def get_kernel_type(name: str) -> str:
    if "ncclKernel" in name:
        return "COMMUNICATION"
    elif "Memcpy" in name or "Memset" in name:
        return "MEMORY"
    else:
        return "COMPUTATION"


def merge_kernel_intervals(kernel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all kernel intervals in the given dataframe such that there are no overlapping.
    """
    kernel_df.sort_values(by="ts", inplace=True)
    kernel_df["end"] = kernel_df["ts"] + kernel_df["dur"]
    # Operators within the same group need to be merged together to form a larger interval.
    kernel_df["group"] = (kernel_df["ts"] > kernel_df["end"].shift().cummax()).cumsum()
    kernel_df = (
        kernel_df.groupby("group", as_index=False)
        .agg({"ts": min, "end": max})
        .drop(["group"], axis=1)
        .sort_values(by="ts")
    )
    return kernel_df

def get_gpu_kernel_type_time(
    gpu_kernels: pd.DataFrame, kernel_type_to_analysis: List[str]
) -> pd.DataFrame:
    overlap_kernel_type_df = pd.DataFrame(
        {
            "status": pd.Series(dtype="str"),
            "time": pd.Series(dtype="int"),
        }
    )

    kernel_t_mapping: Dict[str, int] = defaultdict(int)
    for idx, kernel_type in enumerate(kernel_type_to_analysis):
        value = 1 << idx
        kernel_t_mapping[kernel_type] = value
        kernel_t_df = merge_kernel_intervals(
            gpu_kernels[gpu_kernels["kernel_type"].eq(kernel_type)].copy()
        )

        overlap_kernel_type_df = (
            pd.concat(
                [
                    overlap_kernel_type_df,
                    kernel_t_df.melt(var_name="status", value_name="time").replace(
                        {"ts": value, "end": -value}
                    ),
                ]
            )
            .sort_values(by="time")
            .reset_index(drop=True)
        )

    overlap_kernel_type_df["running"] = overlap_kernel_type_df["status"].cumsum()
    overlap_kernel_type_df["next_time"] = overlap_kernel_type_df["time"].shift(-1)
    unique_running = overlap_kernel_type_df["running"].unique()
    running_mapping: Dict[int, str] = defaultdict(str)
    for u_running in unique_running:
        if u_running > 0:
            for k_t, v_t in kernel_t_mapping.items():
                if u_running & v_t:
                    if u_running not in running_mapping:
                        running_mapping[u_running] = k_t
                    else:
                        running_mapping[
                            u_running
                        ] = f"{running_mapping[u_running]} overlapping {k_t}"

    overlap_kernel_type_df["kernel_type"] = ""
    overlap_kernel_type_df = overlap_kernel_type_df[
        overlap_kernel_type_df["running"] > 0
    ]
    for running in running_mapping:
        overlap_kernel_type_df.loc[
            overlap_kernel_type_df["running"].eq(running), "kernel_type"
        ] = running_mapping[running]
    overlap_kernel_type_df["dur"] = (
        overlap_kernel_type_df["next_time"] - overlap_kernel_type_df["time"]
    ).astype(int)

    overlap_kernel_type_df = overlap_kernel_type_df.groupby(by=["kernel_type"])[
        "dur"
    ].agg(["sum"])
    overlap_kernel_type_df.reset_index(inplace=True)

    return overlap_kernel_type_df

start = time.time()
# trace_dir = "/home/hzeng/prj/FAST/profile"
# trace_file = "/home/hzeng/prj/FAST/sm_p/nico3_2179330.1730340814603087095.pt.trace.json"
trace_file = "/home/hzeng/prj/FAST/profile/trace_rank0_step3_tp4_pp8_no_dis_op.json"
# trace_dir = "/home/hzeng/prj/FAST/sm_p"
with open(trace_file, "r") as fh2:
    trace_record = json.loads(fh2.read())
df: pd.DataFrame = pd.DataFrame()
df = pd.DataFrame(trace_record["traceEvents"])

args_to_keep = {
    "stream",
    # "correlation",
    # "Trace iteration",
    # "memory bandwidth (GB/s)",
}
for arg in args_to_keep:
    df[arg] = df["args"].apply(
        lambda row: row.get(arg, -1) if isinstance(row, dict) else -1
    )
kernel_type_df = pd.DataFrame(
        {
            "kernel_type": pd.Series(dtype="str"),
            "sum": pd.Series(dtype="int"),
        }
    )
kernel_type_to_analysis: List[str] = [
        "COMPUTATION",
        "MEMORY",
        "COMMUNICATION",
    ]


gpu_kernels = df[df["stream"].ne(-1)].copy()
gpu_kernels["kernel_type"] = gpu_kernels[["name"]].apply(
    lambda x: get_kernel_type(x["name"]), axis=1
)

# Create kernel type dataframe
kernel_type_df = pd.concat(
    [
        kernel_type_df,
        get_gpu_kernel_type_time(gpu_kernels, kernel_type_to_analysis),
    ],
    ignore_index=True,
)

kernel_type_df = kernel_type_df.groupby(by=["kernel_type"])["sum"].agg(["sum"])
kernel_type_df.reset_index(inplace=True)
kernel_type_df.sort_values(
    by=["sum"], ignore_index=True, inplace=True, ascending=False
)
kernel_type_df["percentage"] = (
    kernel_type_df["sum"] / kernel_type_df["sum"].sum()
) * 100
kernel_type_df = kernel_type_df.round({"percentage": 1})

end = time.time()
print(end - start)
print(kernel_type_df.to_string())