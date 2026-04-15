import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from prefect import flow, task, get_run_logger
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("HAPPINESS_DATA_DIR")
OUTPUT_DIR = os.getenv("HAPPINESS_OUTPUT_DIR")

if not DATA_DIR or not OUTPUT_DIR:
    raise EnvironmentError(
        "HAPPINESS_DATA_DIR and HAPPINESS_OUTPUT_DIR must be set in .env"
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)


@task(retries=3, retry_delay_seconds=2)
def load_and_merge_data():
    logger = get_run_logger()
    all_data = []
    for year in range(2015, 2025):
        path = os.path.join(DATA_DIR, f"world_happiness_{year}.csv")
        df = pd.read_csv(path, sep=";", decimal=",")
        if "Ladder score" in df.columns:
            df = df.rename(columns={"Ladder score": "Happiness score"})
        df["Year"] = year
        all_data.append(df)
    merged = pd.concat(all_data, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, "merged_happiness.csv")
    merged.to_csv(out_path, index=False)
    logger.info(f"Saved merged dataset to {out_path} ({len(merged)} rows)")
    return merged


@task
def descriptive_stats(df):
    logger = get_run_logger()
    scores = df["Happiness score"]
    logger.info(
        f"Mean: {scores.mean():.3f}, Median: {scores.median():.3f}, Std: {scores.std():.3f}"
    )

    by_year = df.groupby("Year")["Happiness score"].mean()
    for year, val in by_year.items():
        logger.info(f"Year {year}: {val:.3f}")

    by_region = (
        df.groupby("Regional indicator")["Happiness score"]
        .mean()
        .sort_values(ascending=False)
    )
    for region, val in by_region.items():
        logger.info(f"{region}: {val:.3f}")

    return by_region


@task
def visualizations(df):
    logger = get_run_logger()

    plt.figure()
    df["Happiness score"].hist(bins=20)
    plt.title("Happiness Score Distribution")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    path = os.path.join(OUTPUT_DIR, "happiness_histogram.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved {path}")

    plt.figure(figsize=(12, 5))
    df.boxplot(column="Happiness score", by="Year")
    plt.title("Happiness Score by Year")
    plt.suptitle("")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    path = os.path.join(OUTPUT_DIR, "happiness_by_year.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved {path}")

    plt.figure()
    plt.scatter(df["GDP per capita"], df["Happiness score"], alpha=0.4)
    plt.title("GDP per Capita vs Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    path = os.path.join(OUTPUT_DIR, "gdp_vs_happiness.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved {path}")

    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["Ranking", "Year"])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved {path}")


@task
def hypothesis_tests(df):
    logger = get_run_logger()

    scores_2019 = df[df["Year"] == 2019]["Happiness score"].dropna()
    scores_2020 = df[df["Year"] == 2020]["Happiness score"].dropna()
    t, p = stats.ttest_ind(scores_2019, scores_2020)
    logger.info(f"2019 vs 2020 -- t={t:.4f}, p={p:.4f}")
    logger.info(
        f"Mean 2019: {scores_2019.mean():.3f}, Mean 2020: {scores_2020.mean():.3f}"
    )
    if p < 0.05:
        direction = (
            "decreased" if scores_2020.mean() < scores_2019.mean() else "increased"
        )
        logger.info(
            f"Happiness significantly {direction} from 2019 to 2020 (p < 0.05)."
        )
    else:
        logger.info(
            "No significant change in happiness between 2019 and 2020 (p >= 0.05)."
        )

    we = df[df["Regional indicator"] == "Western Europe"]["Happiness score"].dropna()
    ssa = df[df["Regional indicator"] == "Sub-Saharan Africa"][
        "Happiness score"
    ].dropna()
    t2, p2 = stats.ttest_ind(we, ssa)
    logger.info(f"Western Europe vs Sub-Saharan Africa -- t={t2:.4f}, p={p2:.4f}")
    logger.info(f"Mean WE: {we.mean():.3f}, Mean SSA: {ssa.mean():.3f}")
    if p2 < 0.05:
        logger.info(
            "Western Europe is significantly happier than Sub-Saharan Africa (p < 0.05)."
        )

    return {
        "pandemic_p": p,
        "pandemic_direction": (
            "decreased" if scores_2020.mean() < scores_2019.mean() else "increased"
        ),
    }


@task
def correlations_and_bonferroni(df):
    logger = get_run_logger()
    explanatory = [
        "GDP per capita",
        "Social support",
        "Healthy life expectancy",
        "Freedom to make life choices",
        "Generosity",
        "Perceptions of corruption",
    ]
    n_tests = len(explanatory)
    adjusted_alpha = 0.05 / n_tests
    logger.info(
        f"Running {n_tests} tests. Adjusted alpha (Bonferroni): {adjusted_alpha:.4f}"
    )

    strongest_var = None
    strongest_corr = 0

    for var in explanatory:
        clean = df[[var, "Happiness score"]].dropna()
        corr, p = stats.pearsonr(clean[var], clean["Happiness score"])
        sig_orig = "yes" if p < 0.05 else "no"
        sig_adj = "yes" if p < adjusted_alpha else "no"
        logger.info(
            f"{var}: r={corr:.3f}, p={p:.4e} | sig@0.05={sig_orig}, sig@{adjusted_alpha:.4f}={sig_adj}"
        )
        if p < adjusted_alpha and abs(corr) > abs(strongest_corr):
            strongest_corr = corr
            strongest_var = var

    return {"strongest_var": strongest_var, "strongest_corr": strongest_corr}


@task
def summary_report(df, by_region, hypothesis_result, correlation_result):
    logger = get_run_logger()
    logger.info(
        f"Total countries: {df['Country'].nunique()}, Total years: {df['Year'].nunique()}"
    )
    logger.info(f"Top 3 regions: {', '.join(by_region.head(3).index.tolist())}")
    logger.info(f"Bottom 3 regions: {', '.join(by_region.tail(3).index.tolist())}")
    if hypothesis_result["pandemic_p"] < 0.05:
        logger.info(
            f"Happiness {hypothesis_result['pandemic_direction']} significantly between 2019 and 2020 (p < 0.05)."
        )
    else:
        logger.info("No significant change in happiness between 2019 and 2020.")
    logger.info(
        f"Strongest predictor of happiness (after Bonferroni): {correlation_result['strongest_var']} (r={correlation_result['strongest_corr']:.3f})"
    )


@flow
def happiness_pipeline():
    df = load_and_merge_data()
    by_region = descriptive_stats(df)
    visualizations(df)
    hyp_result = hypothesis_tests(df)
    corr_result = correlations_and_bonferroni(df)
    summary_report(df, by_region, hyp_result, corr_result)


if __name__ == "__main__":
    happiness_pipeline()
