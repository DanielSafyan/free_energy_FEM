"""
Analysis functions for run-length encoded (RLE) ball position tracking data.
Updated to work with the new CSV format that logs block sequences instead of individual timesteps.
"""

import pandas as pd



def analyze_block_dwell_times_rle(csv_file_path="gameplay/ball_position_tracking.csv"):
    """
    Analyzes the run-length encoded CSV file to calculate block dwell time statistics.
    
    Args:
        csv_file_path (str): Path to the CSV file containing run-length encoded ball position data
        
    Returns:
        dict: Analysis results containing:
            - average_dwell_time: Average number of consecutive timesteps in same block
            - total_sequences: Total number of block sequences found
            - dwell_times: List of all individual dwell times
            - block_statistics: Per-block dwell time statistics
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        if df.empty:
            return {
                'error': 'CSV file is empty',
                'average_dwell_time': 0,
                'total_sequences': 0,
                'dwell_times': [],
                'block_statistics': {}
            }
        
        # Sort by start_timestep to ensure correct order
        df = df.sort_values('start_timestep')
        
        # Extract dwell times (durations) and block indices
        dwell_times = df['duration'].tolist()
        block_indices = df['block_index'].tolist()
        
        # Calculate per-block statistics
        block_dwell_times = {}
        for i, block_id in enumerate(block_indices):
            duration = dwell_times[i]
            if block_id not in block_dwell_times:
                block_dwell_times[block_id] = []
            block_dwell_times[block_id].append(duration)
        
        # Calculate overall statistics
        if dwell_times:
            average_dwell_time = sum(dwell_times) / len(dwell_times)
        else:
            average_dwell_time = 0
        
        # Calculate per-block statistics
        block_statistics = {}
        for block_id, times in block_dwell_times.items():
            if times:
                block_statistics[block_id] = {
                    'average_dwell_time': sum(times) / len(times),
                    'min_dwell_time': min(times),
                    'max_dwell_time': max(times),
                    'total_visits': len(times),
                    'total_timesteps': sum(times)
                }
        
        # Calculate total timesteps analyzed
        total_timesteps = sum(dwell_times)
        
        return {
            'average_dwell_time': average_dwell_time,
            'total_sequences': len(dwell_times),
            'dwell_times': dwell_times,
            'block_statistics': block_statistics,
            'total_timesteps_analyzed': total_timesteps
        }
        
    except FileNotFoundError:
        return {
            'error': f'CSV file not found: {csv_file_path}',
            'average_dwell_time': 0,
            'total_sequences': 0,
            'dwell_times': [],
            'block_statistics': {}
        }
    except Exception as e:
        return {
            'error': f'Error analyzing CSV file: {str(e)}',
            'average_dwell_time': 0,
            'total_sequences': 0,
            'dwell_times': [],
            'block_statistics': {}
        }


def print_dwell_time_analysis_rle(csv_file_path="gameplay/ball_position_tracking.csv"):
    """
    Convenience function to print a formatted analysis of block dwell times from RLE data.
    
    Args:
        csv_file_path (str): Path to the CSV file containing run-length encoded ball position data
    """
    results = analyze_block_dwell_times_rle(csv_file_path)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print("=== Ball Block Dwell Time Analysis (RLE Format) ===")
    print(f"Total timesteps analyzed: {results['total_timesteps_analyzed']}")
    print(f"Total block sequences: {results['total_sequences']}")
    print(f"Average consecutive timesteps in same block: {results['average_dwell_time']:.2f}")
    
    if results['dwell_times']:
        print(f"Shortest dwell time: {min(results['dwell_times'])} timesteps")
        print(f"Longest dwell time: {max(results['dwell_times'])} timesteps")
    
    print("\n=== Per-Block Statistics ===")
    for block_id, stats in results['block_statistics'].items():
        if block_id == -1:
            block_name = "Outside grid"
        else:
            block_name = f"Block {block_id}"
        
        print(f"{block_name}:")
        print(f"  Average dwell time: {stats['average_dwell_time']:.2f} timesteps")
        print(f"  Min/Max dwell time: {stats['min_dwell_time']}/{stats['max_dwell_time']} timesteps")
        print(f"  Total visits: {stats['total_visits']}")
        print(f"  Total time spent: {stats['total_timesteps']} timesteps")
        print()



if __name__ == "__main__":
    # Example usage
    csv_file = "gameplay/ball_position_tracking.csv"
    
    # Print analysis
    print_dwell_time_analysis_rle(csv_file)
    
    # Create visualizations
    visualize_block_statistics(csv_file)
    create_block_grid_visualization(csv_file)
