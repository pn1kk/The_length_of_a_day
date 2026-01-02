import numpy as np
import matplotlib.pyplot as plt
import csv

def research_model(day_number):
    return 314 * np.cos(2 * np.pi * day_number / 365 - 69 * np.pi / 73) + 737

def load_data(filename):
    days = []
    durations = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            days.append(int(row['day_number']))
            durations.append(int(row['duration_minutes']))
    return np.array(days), np.array(durations)

days, actual_durations = load_data('data/ramenskoye_day_length.csv')

predicted_durations = research_model(days)

errors = actual_durations - predicted_durations
max_error = np.max(np.abs(errors))
mean_error = np.mean(np.abs(errors))

print(f"\nModel: y = 314 × cos(2πx/365 - 69π/73) + 737")
print(f"\nMaximum day length: {np.max(actual_durations)} min (day {days[np.argmax(actual_durations)]})")
print(f"Minimum day length: {np.min(actual_durations)} min (day {days[np.argmin(actual_durations)]})")
print(f"\nModel performance:")
print(f"Maximum error: {max_error:.1f} minutes")
print(f"Mean absolute error: {mean_error:.1f} minutes")

plt.figure(figsize=(10, 6))

plt.scatter(days, actual_durations, alpha=0.5, s=20, 
            label='Actual data (Ramenskoye)', color='blue')

smooth_days = np.linspace(1, 365, 1000)
smooth_predictions = research_model(smooth_days)
plt.plot(smooth_days, smooth_predictions, 'r-', linewidth=2, 
         label='Model: y = 314×cos(2πx/365 - 69π/73) + 737')

plt.xlabel('Day number of the year', fontsize=12)
plt.ylabel('Day length (minutes)', fontsize=12)
plt.title('Day Length vs Day Number in Ramenskoye\nPolina Nikulina, December 2023', 
          fontsize=14, pad=20)

plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('day_length_analysis.png', dpi=300)
print("\nPlot saved as 'day_length_analysis.png'")
plt.show()
