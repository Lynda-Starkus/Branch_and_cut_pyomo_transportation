# (c) 2017-2019 ETH Zurich, Automatic Control Lab, Joe Warrington, Dominik Ruchti

import pandas as pd
import numpy as np
import os
import platform
import matplotlib as mpl

# Configure matplotlib backend for non-interactive environments
if os.environ.get('DISPLAY', '') == '' and platform.system() == 'Linux':
    mpl.use('Agg')  # Use non-interactive Agg backend on Linux server

from matplotlib import pyplot as plt

# Set font family to serif
mpl.rcParams['font.family'] = 'serif'

def process_suffix(suffix: str, output_dir: str = 'output'):
    """
    Process CSV files with a given suffix, compute relative metrics,
    aggregate statistics, and optionally generate plots.

    Parameters:
    - suffix (str): The suffix to filter files.
    - output_dir (str): Directory where the CSV files are located.
    """
    # Import and concatenate results sheets
    suffix_len = len(suffix)
    filelist = [
        fname for fname in os.listdir(output_dir)
        if fname.startswith('stats') and fname.endswith(f"{suffix}.csv")
    ]

    if not filelist:
        print(f"Warning: No files found for suffix '{suffix}'. Ensure the files are in the '{output_dir}' directory.")
        return

    dfl = []
    for f in filelist:
        file_path = os.path.join(output_dir, f)
        df = pd.read_csv(file_path)

        try:
            # Ensure 'k' is treated consistently as numeric
            df['k_numeric'] = pd.to_numeric(df['k'], errors='coerce')
            baseline_row = df[df['k_numeric'] == 0]
            if baseline_row.empty:
                print(f"No baseline data (k_numeric == 0) in file: {file_path}")
                continue

            baseline_sr = baseline_row['SR'].iloc[0]
            baseline_cost = baseline_row['Cost'].iloc[0]
            df['SR_rel'] = df['SR'] - baseline_sr
            df['Cost_rel'] = df['Cost'] - baseline_cost
        except Exception as e:
            print(f"Error processing file: {file_path}")
            print(e)
            continue

        dfl.append(df)

    if not dfl:
        print(f"No valid data found for suffix '{suffix}'. Skipping.")
        return

    dc = pd.concat(dfl, ignore_index=True)
    print(f"Suffix {suffix}: {len(filelist)} files found, containing {len(dc)} records.")

    # Average results across network sizes, on the "Inst." column.
    iter_group = dc[dc['Integer'] == '0.0'].groupby(['N', 'V'])
    iter_times = iter_group[['t1', 't2']].mean()
    iter_times_std = iter_group[['t1', 't2']].std()
    rec_count = iter_group['k_numeric'].apply(lambda x: (x == 50).sum())

    print(rec_count)

    try:
        iter_csv = iter_times.merge(
            right=iter_times_std,
            on=['N', 'V'],
            suffixes=('_mean', '_std')
        )
        if iter_csv.empty:
            print(f"No valid iteration data to save for suffix '{suffix}'. Skipping.")
        else:
            iter_csv.round(6).to_csv(os.path.join(output_dir, f'iter_times{suffix}.csv'))
    except KeyError as e:
        print("Error merging iteration times and std deviations:")
        print(e)
        return

    # Group by ['N', 'V', 'k_numeric'] and compute mean and std for selected columns
    grouped = dc.groupby(['N', 'V', 'k_numeric'])[['Cost', 'SR', 'Cost_rel', 'SR_rel']]
    cost_sr_mean = grouped.mean().dropna()
    cost_sr_std = grouped.std().dropna()

    # Merge mean and std dataframes
    mean_std_df = cost_sr_mean.merge(
        right=cost_sr_std,
        on=['N', 'V', 'k_numeric'],
        suffixes=('_mean', '_std')
    )

    if mean_std_df.empty:
        print(f"No valid statistics to save for suffix '{suffix}'. Skipping.")
    else:
        mean_std_df.round(6).to_csv(os.path.join(output_dir, f'mean_std_stats{suffix}.csv'))

    save_plots = True  # Set to True to enable plot saving
    if save_plots:
        for (N, V), sub_df in mean_std_df.groupby(['N', 'V']):
            try:
                k_list = sub_df.index.get_level_values('k_numeric').round().astype(int).tolist()
                sr_mean_list = sub_df['SR_rel_mean'].tolist()
                sr_std_list = sub_df['SR_rel_std'].tolist()
                cost_mean_list = sub_df['Cost_rel_mean'].tolist()
                cost_std_list = sub_df['Cost_rel_std'].tolist()

                if suffix == '_integer':
                    # Correct for the fact that the integer solutions have an unneeded iteration 51.
                    k_list = k_list[:-1]
                    sr_mean_list = sr_mean_list[:-1]
                    sr_std_list = sr_std_list[:-1]
                    cost_mean_list = cost_mean_list[:-1]
                    cost_std_list = cost_std_list[:-1]

                # Plot Service Rate Increase
                plt.figure(figsize=(8, 4))
                plt.plot(k_list, sr_mean_list, 'k', label='Mean SR_rel')
                plt.plot(k_list, np.array(sr_mean_list) + np.array(sr_std_list), 'k--', label='SR_rel ± STD')
                plt.plot(k_list, np.array(sr_mean_list) - np.array(sr_std_list), 'k--')
                plt.xlim([min(k_list), max(k_list)])
                plt.xlabel('Iteration $k$')
                plt.ylabel('Service rate increase (%)')
                plt.title(f'{N} nodes, {V} RV' + ('' if V == 1 else 's'))
                plt.legend()
                filename_sr = os.path.join(output_dir, f'sr_stats_{N}_{V}{suffix}.pdf')
                plt.tight_layout()
                plt.savefig(filename_sr)
                plt.close()

                # Plot Cost Change
                plt.figure(figsize=(8, 4))
                plt.plot(k_list, cost_mean_list, 'k', label='Mean Cost_rel')
                plt.plot(k_list, np.array(cost_mean_list) + np.array(cost_std_list), 'k--', label='Cost_rel ± STD')
                plt.plot(k_list, np.array(cost_mean_list) - np.array(cost_std_list), 'k--')
                plt.xlim([min(k_list), max(k_list)])
                plt.xlabel('Iteration $k$')
                plt.ylabel('Cost change')
                plt.title(f'{N} nodes, {V} RV' + ('' if V == 1 else 's'))
                plt.legend()
                filename_cost = os.path.join(output_dir, f'cost_stats_{N}_{V}{suffix}.pdf')
                plt.tight_layout()
                plt.savefig(filename_cost)
                plt.show()
            except Exception as e:
                print(f"Failed to generate graphs for N={N}, V={V}, suffix={suffix}: {e}")
                continue

def main():
    """
    Main function to process all specified suffixes.
    """
    suffixes = ['_regular', '_integer', '_halfinteger', '_det2s_corrected', '_random']
    for suffix in suffixes:
        process_suffix(suffix)

if __name__ == "__main__":
    main()
