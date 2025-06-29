# chimera/analysis/plot_evaluation.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def plot_comparison(target_csv: Path, mimic_csv: Path, output_dir: Path):
    """
    Loads target and mimic telemetry data and generates comparison plots.
    """
    print(f"Loading target data from: {target_csv}")
    target_df = pd.read_csv(target_csv)
    print(f"Loading mimic data from: {mimic_csv}")
    mimic_df = pd.read_csv(mimic_csv)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify common telemetry columns to plot
    cols_to_plot = [col for col in mimic_df.columns if col in target_df.columns and col not in ['time', 'x', 'y', 'z']]
    
    # Trim dataframes to the same length for fair comparison
    min_len = min(len(target_df), len(mimic_df))
    target_df = target_df.iloc[:min_len]
    mimic_df = mimic_df.iloc[:min_len]

    # Set up time axis if available, otherwise use index
    time_axis = target_df['time'] if 'time' in target_df else target_df.index
    
    sns.set_theme(style="whitegrid")

    for col in cols_to_plot:
        plt.figure(figsize=(15, 7))
        plt.plot(time_axis, target_df[col], label=f'Target ({col})', color='green', linestyle='--')
        plt.plot(time_axis, mimic_df[col], label=f'Mimic ({col})', color='blue', alpha=0.8)
        
        plt.title(f'Target vs. Mimic Comparison: {col.replace("-", " ").title()}', fontsize=16)
        plt.xlabel('Time (s)' if 'time' in target_df else 'Timestep')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plot_filename = output_dir / f'comparison_{col}.png'
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved plot: {plot_filename}")

    # Trajectory Plot (X-Y)
    plt.figure(figsize=(10, 10))
    plt.plot(target_df['x'], target_df['y'], label='Target Trajectory', color='green', linestyle='--')
    plt.plot(mimic_df['x'], mimic_df['y'], label='Mimic Trajectory', color='blue', alpha=0.8)
    plt.title('Trajectory Comparison (X-Y Plane)', fontsize=16)
    plt.xlabel('X coordinate (m)')
    plt.ylabel('Y coordinate (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plot_filename = output_dir / 'comparison_trajectory.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved plot: {plot_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot evaluation results for Chimera.")
    parser.add_argument('--target_csv', type=str, required=True, help="Path to the target telemetry CSV.")
    parser.add_argument('--mimic_csv', type=str, required=True, help="Path to the mimic vehicle's evaluation CSV.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the plots.")
    
    args = parser.parse_args()
    
    plot_comparison(Path(args.target_csv), Path(args.mimic_csv), Path(args.output_dir))