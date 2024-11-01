import pandas as pd
from collections import defaultdict
from typing import List, Dict
import os
import json
from typing import Union
import time

class KernelAnalyzer:
    def __init__(self, trace_path: Union[str, List[str]]):
        """
        Initialize the TraceProcessor with a path to a trace file or directory.

        :param trace_path: Path to a single trace file or a directory containing multiple trace files.
        """
        self.trace_path = trace_path
        self.df_list = []
        t0 = time.perf_counter()
        self.load_traces()
        t1 = time.perf_counter()
        print(f"Loaded traces in {t1-t0} seconds")
        self.kernel_type_to_analysis: List[str] = [
            "COMPUTATION",
            "MEMORY",
            "COMMUNICATION",
        ]

    def load_traces(self):
        """
        Load traces from the specified path.
        """
        if isinstance(self.trace_path, str):
            if os.path.isdir(self.trace_path):
                # Load all JSON files in the directory
                self.df_list = [self.load_trace_file(os.path.join(self.trace_path, file)) \
                    for file in os.listdir(self.trace_path) \
                    if file.endswith('.json')]
            elif os.path.isfile(self.trace_path):
                # Load a single JSON file
                self.df_list = [self.load_trace_file(self.trace_path)]
            else:
                raise ValueError("The provided path is neither a valid file nor a directory.")
        elif isinstance(self.trace_path, list):
            # Load multiple files from the list
            self.df_list = [self.load_trace_file(file) for file in self.trace_path]
        else:
            raise ValueError("The provided trace_path must be a string or a list of strings.")

    def load_trace_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a single trace file and process it.

        :param file_path: Path to the trace file.
        :return: Processed DataFrame.
        """
        with open(file_path, "r") as fh2:
            trace_record = json.loads(fh2.read())
        
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
        
        
        return df

    @staticmethod
    def get_kernel_type(name: str) -> str:
        # if "ncclKernel" in name:
        if "nccl" in name:
            return "COMMUNICATION"
        elif "Memcpy" in name or "Memset" in name:
            return "MEMORY"
        else:
            return "COMPUTATION"

    @staticmethod
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

    @staticmethod
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
            kernel_t_df = KernelAnalyzer.merge_kernel_intervals(
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
    
    def get_gpu_kernel_time(self, kernel_type_to_analysis: List[str] = None) -> pd.DataFrame:
        t0 = time.perf_counter()
        if kernel_type_to_analysis is None:
            kernel_type_to_analysis = self.kernel_type_to_analysis
        kernel_type_df = pd.DataFrame(
            {
                "kernel_type": pd.Series(dtype="str"),
                "sum": pd.Series(dtype="int"),
            }
        )
        for df in self.df_list:
            gpu_kernels = df[df["stream"].ne(-1)].copy()
            gpu_kernels["kernel_type"] = gpu_kernels[["name"]].apply(
                lambda x: KernelAnalyzer.get_kernel_type(x["name"]), axis=1
            )
            # Create kernel type dataframe
            kernel_type_df = pd.concat(
                [
                    kernel_type_df,
                    KernelAnalyzer.get_gpu_kernel_type_time(gpu_kernels, kernel_type_to_analysis),
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
        t1 = time.perf_counter()
        print(f"get_gpu_kernel_time takes {t1-t0} seconds")
        return kernel_type_df
def main():
    trace_dir = "/home/hzeng/prj/vllm/examples/profile/rank_0"
    analyzer = KernelAnalyzer(trace_dir)
    result = analyzer.get_gpu_kernel_time()
    print(result.to_string())
    compare_with_hta = True
    if(compare_with_hta):
        from hta.trace_analysis import TraceAnalysis
        t0 = time.perf_counter()
        analyzer = TraceAnalysis(trace_dir = trace_dir)
        kernel_type_metrics_df, kernel_metrics_df = analyzer.get_gpu_kernel_breakdown(visualize = False, 
                                                                              duration_ratio = 0.8,
                                                                            #   num_kernels = 5,
                                                                              include_memory_kernels = True)
        t1 = time.perf_counter()
        print(f"hta takes {t1-t0} seconds")
        print(kernel_type_metrics_df)
if __name__ == "__main__":
    main()