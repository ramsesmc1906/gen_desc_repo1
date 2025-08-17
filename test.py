import pandas as pd
import matplotlib.pyplot as plt

def analyze_gpu_log(log_file):
    """
    Reads a GPU usage log file, calculates statistics, and plots the data.

    Args:
        log_file (str): The path to the CSV log file.
    """
    print("\n--- Analyzing GPU Usage ---")
    try:
        # Read the CSV log file into a pandas DataFrame
        df = pd.read_csv(log_file)
        
        # Convert the timestamp column to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f')
        
        # Calculate the elapsed time in seconds from the start of the log
        df['elapsed_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

        print("\nGPU Usage Statistics:")
        # Print descriptive statistics for GPU utilization and memory usage
        print(df[['gpu_utilization_percent', 'memory_used_mb']].describe())

        # --- Plotting the Data ---
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot GPU Utilization on the primary y-axis (ax1)
        color = 'tab:red'
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('GPU Utilization (%)', color=color)
        ax1.plot(df['elapsed_seconds'], df['gpu_utilization_percent'], color=color, label='GPU Utilization')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 105) # Set y-axis from 0 to 105 for percentage

        # Create a second y-axis (ax2) for Memory Usage
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Memory Used (MB)', color=color)
        ax2.plot(df['elapsed_seconds'], df['memory_used_mb'], color=color, label='Memory Used')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.suptitle('GPU Performance Analysis', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Display the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: The log file '{log_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")

# Example usage:
if __name__ == "__main__":
    # Change 'gpu_usage.log' to the name of your specific log file
    log_file_name = 'gpu_usage.log' 
    analyze_gpu_log(log_file_name)
