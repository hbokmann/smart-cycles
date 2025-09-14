# count_cycles.py
# Analyze production cycles from power consumption data
import pandas as pd
import os
from os.path import join, dirname
import plotly.express as px

# read, parse and process
file_name = "process_data_analyst_test.csv"
file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', file_name)
df = pd.read_csv(file_path, sep=";")

# Parse time
df["time"] = pd.to_datetime(df["time"])

# Only keep Power_consumption rows
df = df[df["metric"] == "Power_consumption"].copy()



# Define threshold (idle vs running)
threshold = 1.0 # adjust depending on data distribution


# Running = power > threshold
df["running"] = df["value"] > threshold


# Detect cycle start (transition from idle to running)
df["cycle_start"] = (df["running"] & ~df["running"].shift(fill_value=False))


# Detect cycle end (running → idle)
df["cycle_end"] = (~df["running"] & df["running"].shift(fill_value=False))


# Extract cycle start and end times
cycle_starts = df[df['cycle_start']][['time']].reset_index(drop=True)
cycle_ends   = df[df['cycle_end']][['time']].reset_index(drop=True)

# Align cycles (handle if the first cycle starts without a preceding idle)
if cycle_ends.index[0] < cycle_starts.index[0]:
    cycle_ends = cycle_ends.iloc[1:].reset_index(drop=True)
min_len = min(len(cycle_starts), len(cycle_ends))
cycle_starts = cycle_starts.iloc[:min_len]
cycle_ends   = cycle_ends.iloc[:min_len]

# Calculate durations
cycles = pd.DataFrame({
    'start_time': cycle_starts['time'],
    'end_time': cycle_ends['time']
})
cycles['duration_sec'] = (cycles['end_time'] - cycles['start_time']).dt.total_seconds()

# Calculate pause durations (time between end of one cycle and start of next)
cycles['next_start_time'] = cycles['start_time'].shift(-1)
cycles['pause_sec'] = (cycles['next_start_time'] - cycles['end_time']).dt.total_seconds()

print("Cycle durations and pauses:")
print(cycles)


# Count total cycles
cycle_count = df["cycle_start"].sum()
print(f"Total cycles: {cycle_count}")


# Assign shift and correct shift_date so night shift belongs to the previous day
def assign_shift_and_date(ts):
    hour = ts.hour
    if 6 <= hour < 14:
        return 'morning', ts.date()
    elif 14 <= hour < 22:
        return 'day', ts.date()
    else:
        # Night shift spans 22:00–06:00 → anchor to the starting day (previous day if before 6AM)
        shift_date = ts.date() if hour >= 22 else (ts - pd.Timedelta(days=1)).date()
        return 'night', shift_date

df[['shift', 'shift_date']] = df['time'].apply(assign_shift_and_date).apply(pd.Series)

# Determine full vs half shifts
def shift_is_full(shift_group):
    start_time = shift_group['time'].min()
    end_time = shift_group['time'].max()
    shift_name = shift_group['shift'].iloc[0]

    if shift_name == 'morning':  # 06:00–14:00
        return start_time.hour == 6 and end_time.hour == 13

    elif shift_name == 'day':  # 14:00–22:00
        return start_time.hour == 14 and end_time.hour == 21

    else:  # night shift = 22:00–06:00 (anchored to previous day)
        # We want to see data covering both sides of midnight
        # That means: data at/after 22:00, and data at/before 05:00 (next day)
        has_late = (start_time.hour == 22) and (end_time.hour == 5)
        # has_early = (start_time.hour == 5) or (end_time.hour == 6)
        return has_late

shift_groups = df.groupby(['shift_date', 'shift'])
shift_full_map = shift_groups.apply(lambda g: 'full' if shift_is_full(g) else 'half').reset_index()
shift_full_map.columns = ['shift_date', 'shift', 'shift_type']

# Merge back
df = df.merge(shift_full_map, on=['shift_date', 'shift'], how='left')

# Keep only full shifts
df_full = df[df['shift_type'] == 'full']

# Count cycles per individual full shift
cycles_per_individual_shift = (
    df_full[df_full['cycle_start']]
    .groupby(['shift_date', 'shift'])
    .size()
    .reset_index(name='cycles')
)

print("Cycles per full shift (individual):")
print(cycles_per_individual_shift)

# Compute average cycles per shift type
average_cycles_per_shift_type = (
    cycles_per_individual_shift.groupby('shift')['cycles']
    .mean()
    .reset_index(name='avg_cycles')
)

print("\nAverage cycles per full shift type:")
print(average_cycles_per_shift_type)

# Analyze variability in cycle durations and pauses

# Assume `cycles['duration_sec']` and `cycles['pause_sec']` are available

# Cycle duration variability
duration_mean = cycles['duration_sec'].mean()
duration_std  = cycles['duration_sec'].std()
duration_cv   = duration_std / duration_mean  # Coefficient of Variation

# Pause variability
pause_mean = cycles['pause_sec'].mean()
pause_std  = cycles['pause_sec'].std()
pause_cv   = pause_std / pause_mean

print(f"Cycle duration: mean={duration_mean:.2f}s, std={duration_std:.2f}s, CV={duration_cv:.2f}")
print(f"Pause duration: mean={pause_mean:.2f}s, std={pause_std:.2f}s, CV={pause_cv:.2f}")



# --- Shift-based cycle & pause analysis ---

# First, map each cycle to its shift
# Use the start_time of each cycle to determine shift
cycles[['shift', 'shift_date']] = cycles['start_time'].apply(assign_shift_and_date).apply(pd.Series)

# Merge with shift_type (full/half) from earlier
cycles = cycles.merge(shift_full_map, on=['shift_date', 'shift'], how='left')

# Keep only full shifts
cycles_full = cycles[cycles['shift_type'] == 'full']

# Compute average cycle & pause durations per shift type
shift_cycle_stats = (
    cycles_full.groupby('shift')
    .agg(
        avg_cycle_duration=('duration_sec', 'mean'),
        std_cycle_duration=('duration_sec', 'std'),
        avg_pause_duration=('pause_sec', 'mean'),
        std_pause_duration=('pause_sec', 'std'),
        count=('duration_sec', 'count')
    )
    .reset_index()
)

# Add Coefficient of Variation (CV)
shift_cycle_stats['cv_cycle_duration'] = (
    shift_cycle_stats['std_cycle_duration'] / shift_cycle_stats['avg_cycle_duration']
)
shift_cycle_stats['cv_pause_duration'] = (
    shift_cycle_stats['std_pause_duration'] / shift_cycle_stats['avg_pause_duration']
)

print("\nCycle & pause stats per full shift type:")
print(shift_cycle_stats)




# visualization

# Label cycles with cumulative sum of starts
df["cycle_id"] = df["cycle_start"].cumsum()

# Plot with Plotly
fig = px.line(df, x="time", y="value", color="cycle_id", title="Production Cycles")
fig.show()