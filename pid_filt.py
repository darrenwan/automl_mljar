import pandas as pd
import os
from src.config import config
from src.missing.missing_analysis import   compute_missing_summary

from src.missing.data_loader import load_data

commit_dir = f"{config.DATA_FINAL_DIR}/交付"
pid = config.pid


def dropped_pid_get(suffix='3个医院train'):
    df = pd.read_excel(f"{commit_dir}/predict_all_Perform-{suffix}train.xlsx")
    df_fix = pd.read_excel(f"{commit_dir}/副本predict_all_Perform-{suffix}train(1修).xlsx")
    df_pid = df[pid].tolist()
    df_pid_fix = df_fix[pid].tolist()
    pid_dropped = list(set(df_pid) - set(df_pid_fix))
    return pid_dropped


def impute_pid_drop(suffix='3个医院'):
    pid_dropped = dropped_pid_get(suffix)
    mid_input_file = config.DATA_WIDE_FILE.replace(".xlsx", "_clean.xlsx")
    outf = mid_input_file.replace(".xlsx", f"_clean_{suffix}_pid_dropped.xlsx")

    df_clean = pd.read_excel(mid_input_file)
    df_clean = df_clean[~df_clean[pid].isin(pid_dropped)]
    df_clean.to_excel(outf, index=False)

    out_dir = os.path.split(mid_input_file)[0]
    miss_small, miss_big = compute_missing_summary(df_clean, features=df_clean.columns.tolist(), output_dir=out_dir)


if __name__ == "__main__":
    # impute_pid_drop(suffix='3个医院')
    impute_pid_drop(suffix='控江医院')
