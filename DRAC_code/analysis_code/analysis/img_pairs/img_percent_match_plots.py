import numpy as np
import pandas as pd
import csv
import rsatoolbox
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress
from statsmodels.stats.multitest import multipletests
import json
import traceback
import h5py
from sklearn.linear_model import HuberRegressor
from matplotlib.ticker import FormatStrFormatter
from PIL import Image
from matplotlib.ticker import PercentFormatter


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")

ROI_group = "all" # v1-v4_Cntrl or all
full_y = ""

if ROI_group == "v1-v4_Cntrl":
    ROIs = ["V1", "V2", "V3", "V4", "A1", "Pir"] 
elif ROI_group == "all":
    ROIs = ["V1", "V2", "V3", "V4",  "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC", "A1", "Pir"]
else:
    ROIs = []

img_variables = [('_Head_Direction', 'head direction'), ('_Species', 'species'), ('_Sub_Species', 'sub species'), ('_Branches', 'branches'), ('_Leaves', 'leaves'), ('_Grass', 'grass'), ('_Bg_Focused', 'bg focused'), ('_Beak_Open', 'beak open')]

x_u_lim = 10000 # Upper limit for x-axis on charts, or "no_x_lim"
results = {}

# Directory to save statistics
stats_dir = results_dir / "dissimilarity_stats"
stats_dir.mkdir(parents=True, exist_ok=True)

for im_var, im_var_name in img_variables:
    im_var_dir = results_dir / f"img_pair_plots/var_match_lines/{im_var}"
    im_var_dir.mkdir(parents=True, exist_ok=True)

    for version in ["top_5", "low_high", "top_5_avg_tie", "top_5_new"]:
        print(f"\n\n========== ({im_var_name}) Version: {version} ==========")
        results[version] = {} 
        percent_series_low = {}
        percent_series_high = {}

        stats_rows = []

        for ROI in ROIs:
            results[version][ROI] = {}

            roi_results_dir = results_dir / f"img_pair_DFs/{ROI}/{version}"
            low_exp_DF = pd.read_parquet(roi_results_dir / f'low_exp_DF.parquet')
            high_exp_DF = pd.read_parquet(roi_results_dir / f'high_exp_DF.parquet')

            # ensure consistent index ordering
            low_exp_DF = low_exp_DF.reset_index(drop=True)
            high_exp_DF = high_exp_DF.reset_index(drop=True)

            # truncate to x_u_lim
            if x_u_lim != "no_x_lim":
                low_exp_DF_trunc = low_exp_DF.head(x_u_lim)
                high_exp_DF_trunc = high_exp_DF.head(x_u_lim)
            else:
                low_exp_DF_trunc = low_exp_DF
                high_exp_DF_trunc = high_exp_DF

            # compute running percentage for plots
            n_low = len(low_exp_DF_trunc)
            eq_low = (low_exp_DF_trunc[f"img1{im_var}"] == low_exp_DF_trunc[f"img2{im_var}"]).astype(int)
            cum_low = eq_low.cumsum()
            idx = pd.RangeIndex(1, n_low + 1)
            percent_series_low[ROI] = (cum_low / idx) * 100

            n_high = len(high_exp_DF_trunc)
            eq_high = (high_exp_DF_trunc[f"img1{im_var}"] == high_exp_DF_trunc[f"img2{im_var}"]).astype(int)
            cum_high = eq_high.cumsum()
            idx = pd.RangeIndex(1, n_high + 1)
            percent_series_high[ROI] = (cum_high / idx) * 100

            # truncate for top 100 dissimilarities for existing mean/var calculations
            low_exp_DF_head = low_exp_DF_trunc.head(100)
            high_exp_DF_head = high_exp_DF_trunc.head(100)

            # Store mean/var for your results dict
            results[version][ROI]['mean_low_exp_dissimilarity'] = low_exp_DF_head['dissimilarity'].mean()
            results[version][ROI]['mean_high_exp_dissimilarity'] = high_exp_DF_head['dissimilarity'].mean()
            results[version][ROI]['var_low_exp_dissimilarity'] = low_exp_DF_head['dissimilarity'].var()
            results[version][ROI]['var_high_exp_dissimilarity'] = high_exp_DF_head['dissimilarity'].var()

            # --- Compute statistics for CSV using truncated data (up to x_u_lim) ---
            stats_row = {
                "ROI": ROI,
                "mean_low_exp": low_exp_DF_trunc['dissimilarity'].mean(),
                "std_low_exp": low_exp_DF_trunc['dissimilarity'].std(),
                "var_low_exp": low_exp_DF_trunc['dissimilarity'].var(),
                "mean_high_exp": high_exp_DF_trunc['dissimilarity'].mean(),
                "std_high_exp": high_exp_DF_trunc['dissimilarity'].std(),
                "var_high_exp": high_exp_DF_trunc['dissimilarity'].var()
            }
            stats_rows.append(stats_row)

        # Save CSV for this img_variable and version
        csv_path = stats_dir / f"{ROI_group}_dissimilarity_stats_{im_var}_{version}_{x_u_lim}.csv"
        pd.DataFrame(stats_rows).to_csv(csv_path, index=False)


        def plot_percent_series(series_dict, title, save_path=None, step=100, x_u_lim="no_x_lim", is_difference=False):
            # Create line plot for series in series_dict.
            plt.figure(figsize=(10, 6))

            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            linestyles = ['solid', 'dashed', 'dashdot']

            n_colors = len(color_cycle)
            n_styles = len(linestyles)

            for i, (roi, series) in enumerate(series_dict.items()):
                if series is None or series.empty:
                    continue
                # Subsample points to reduce plot resolution
                series_sub = series.iloc[::step]
                x = series_sub.index + 1
                color = color_cycle[i % n_colors]
                linestyle = linestyles[(i // n_colors) % n_styles]
                plt.plot(x, series_sub.values, label=roi, color=color, linestyle=linestyle, linewidth=1.8)

            # horizontal reference line
            if is_difference:
                plt.axhline(y=0, color='red', linestyle=':', linewidth=1.5)
            else:
                plt.axhline(y=50, color='red', linestyle=':', linewidth=1.5)

            plt.xlabel("Number of Dissimilarities Used in Percent Calculation (n)")

            if is_difference:
                plt.ylabel(f"Difference in percent same {im_var_name} (High - Low)")
                # set y-limits based on data range with margin if possible
                all_vals = pd.concat([s.dropna() for s in series_dict.values() if s is not None and not s.empty]) if series_dict else pd.Series(dtype=float)
                if not all_vals.empty:
                    vmin, vmax = all_vals.min(), all_vals.max()
                    margin = max(1.0, 0.1 * max(abs(vmin), abs(vmax)))
                    if full_y != "_y_full":
                        plt.ylim(-5, 15)
            else:
                plt.ylabel(f"Percent same {im_var_name}")
                plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
                if full_y != "_y_full":
                    plt.ylim(40, 70)

            plt.title(title)

            # Determine upper x-axis limit
            if x_u_lim != "no_x_lim":
                plt.xlim(0, x_u_lim)
            else:
                max_x = max((len(series) for series in series_dict.values() if series is not None and not series.empty), default=0)
                plt.xlim(0, max_x)

            plt.legend(title="ROI", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.grid(True, linestyle=":", linewidth=0.5)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300)
            plt.close()
        
        if version == "low_high":
            version_text = "All"
        elif version == "top_5_avg_tie":
            version_text = "Tied Avg"
        elif version == "top_5_new":
            version_text = "Top 5 New"
        else:
            version_text = "Top 5"

        # --- existing calls for low and high (unchanged) ---
        plot_percent_series(
            percent_series_low,
            f"Percent same {im_var_name} vs N Dissimilarities Used ({version_text} low group)",
            save_path=im_var_dir / f"{ROI_group}_percent_same{im_var}_low_{version}_{x_u_lim}{full_y}.png",
            step=100,
            x_u_lim=x_u_lim,
            is_difference=False
        )
        plot_percent_series(
            percent_series_high,
            f"Percent same {im_var_name} vs N Dissimilarities Used ({version_text} high group)",
            save_path=im_var_dir / f"{ROI_group}_percent_same{im_var}_high_{version}_{x_u_lim}{full_y}.png",
            step=100,
            x_u_lim=x_u_lim,
            is_difference=False
        )

        percent_series_diff = {}
        for ROI in ROIs:
            s_low = percent_series_low.get(ROI)
            s_high = percent_series_high.get(ROI)

            # align both series to the same index range (0..max_len-1)
            len_low = len(s_low) if (s_low is not None) else 0
            len_high = len(s_high) if (s_high is not None) else 0
            max_len = max(len_low, len_high)

            idx = pd.RangeIndex(0, max_len)
            s_low_re = s_low.reindex(idx) if s_low is not None else pd.Series(index=idx, dtype=float)
            s_high_re = s_high.reindex(idx) if s_high is not None else pd.Series(index=idx, dtype=float)
            # compute difference (NaN will propagate where data is missing)
            percent_series_diff[ROI] = s_high_re - s_low_re

        # Plot the difference series (High - Low)
        plot_percent_series(
            percent_series_diff,
            f"Difference (High - Low) percent same {im_var_name} vs N Dissimilarities Used ({version_text})",
            save_path=im_var_dir / f"{ROI_group}_percent_same{im_var}_difference_{version}_{x_u_lim}{full_y}.png",
            step=100,
            x_u_lim=x_u_lim,
            is_difference=True
        )


        df_low = pd.DataFrame(percent_series_low)
        df_high = pd.DataFrame(percent_series_high)

        # Compute mean across ROIs for each n
        avg_low = df_low.mean(axis=1, skipna=True)
        avg_high = df_high.mean(axis=1, skipna=True)

        # Plot averages only
        plt.figure(figsize=(10, 6))
        plt.plot(avg_low.index + 1, avg_low.values, label=f"{version_text} Low Expertise", linewidth=2.5)
        plt.plot(avg_high.index + 1, avg_high.values, label=f"{version_text} High Expertise", linewidth=2.5)

        plt.axhline(y=50, color='red', linestyle=':', linewidth=1.5)

        plt.xlabel("Number of Dissimiarities Used in Percent Calculation (n)")
        plt.ylabel(f"Percent same {im_var_name}")
        plt.title(f"Percent same {im_var_name} vs N Dissimilarities Used ({version_text})")
        plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
        if full_y != "_y_full":
            plt.ylim(40, 70)
        if x_u_lim != "no_x_lim":
            plt.xlim(0, x_u_lim)
        else:
            max_x = max(len(avg_low), len(avg_high))
            plt.xlim(0, max_x)
        plt.legend(title="Group")
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(im_var_dir / f"{ROI_group}_avg_percent_same{im_var}_high_and_low_{version}_{x_u_lim}{full_y}.png")