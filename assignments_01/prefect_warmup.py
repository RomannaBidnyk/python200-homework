import numpy as np
import pandas as pd
from prefect import flow, task

arr = np.array(
    [12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0]
)


@task
def create_series(arr):
    return pd.Series(arr, name="values")


@task
def clean_data(series):
    return series.dropna()


@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0],
    }


@flow
def pipeline_flow():
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)

    for key, value in summary.items():
        print(f"{key.capitalize()}: {value}")

    return summary


if __name__ == "__main__":
    pipeline_flow()

"""
1. Why might Prefect be more overhead than it is worth here?
This pipeline is very small and runs locally on a tiny dataset.
Using Prefect here adds extra abstraction (tasks, flows, orchestration)
without providing real operational benefits.

2. Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple:
- Retries: Automatically retry failed tasks when external APIs or data sources fail.
- Scheduling: Run pipelines on a schedule (e.g., daily or hourly).
- Observability: Track runs, logs, and failures in a UI for debugging and monitoring.
- Alerts: Send notifications (Slack/email) when a pipeline fails.
- Caching: Reuse results from expensive tasks to avoid recomputation.
"""
