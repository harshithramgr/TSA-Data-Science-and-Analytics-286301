import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from google.colab import drive
drive.mount('/content/drive')

# --- 1. Initial Data Loading and Path Definitions ---

aviation_data_path = '/content/drive/My Drive/aviation data'
print(f"Aviation data path defined as: {aviation_data_path}")

# List all files and directories in the aviation_data_path
all_files = os.listdir(aviation_data_path)

# Filter for .csv files and construct their full paths
csv_files = [os.path.join(aviation_data_path, f) for f in all_files if f.endswith('.csv')]

print(f"Found {len(csv_files)} CSV files:")
for file in csv_files:
    print(file)

dataframes = {}

for file_path in csv_files:
    # Extract filename without extension to use as a dictionary key
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        dataframes[file_name] = df
        print(f"Successfully loaded '{file_path}' into DataFrame '{file_name}'.")
    except Exception as e:
        print(f"Error loading '{file_path}': {e}")

print(f"Loaded {len(dataframes)} DataFrames: {list(dataframes.keys())}")

# --- 2. Specific DataFrame Preparation (airliner impact) ---

df_airliner_impact = dataframes['airliner impact']

print("Original data types of df_airliner_impact:")
print(df_airliner_impact.dtypes)

# Convert 'Year' column to numeric if not already
if not pd.api.types.is_numeric_dtype(df_airliner_impact['Year']):
    df_airliner_impact['Year'] = pd.to_numeric(df_airliner_impact['Year'], errors='coerce')
    print("\n'Year' column converted to numeric.")
else:
    print("\n'Year' column is already numeric.")

print("\nUpdated data types of df_airliner_impact:")
print(df_airliner_impact.dtypes)

print("\nFirst 5 rows of df_airliner_impact:")
print(df_airliner_impact.head())

# --- 3. Aviation Trend Plots ---

# Passenger Demand Plot
plt.figure(figsize=(10, 6))
sns.regplot(x='Year', y='Passenger demand (billion passenger-km)', data=df_airliner_impact, scatter_kws={'alpha':0.6})
plt.xlabel('Year')
plt.ylabel('Passenger Demand (billion passenger-km)')
plt.title('Trend of Passenger Demand Over Years (with Linear Regression)')
plt.grid(True)
plt.show()

# Energy Intensity Plot
plt.figure(figsize=(10, 6))
sns.regplot(x='Year', y='Energy intensity (per passenger-km)', data=df_airliner_impact, scatter_kws={'alpha':0.6})
plt.xlabel('Year')
plt.ylabel('Energy Intensity (per passenger-km)')
plt.title('Trend of Energy Intensity Over Years (with Linear Regression)')
plt.grid(True)
plt.show()

# CO₂ per unit energy Plot
plt.figure(figsize=(10, 6))
sns.regplot(x='Year', y='CO₂ per unit energy (gCO₂eq per MJ)', data=df_airliner_impact, scatter_kws={'alpha':0.6})
plt.xlabel('Year')
plt.ylabel('CO₂ per unit energy (gCO₂eq per MJ)')
plt.title('Trend of CO₂ per Unit Energy Over Years (with Linear Regression)')
plt.grid(True)
plt.show()

# CO₂ emissions Plot
plt.figure(figsize=(10, 6))
sns.regplot(x='Year', y='CO₂ emissions (billion tonnes)', data=df_airliner_impact, scatter_kws={'alpha':0.6})
plt.xlabel('Year')
plt.ylabel('CO₂ Aviation Emissions (billion tonnes)')
plt.title('Trend of CO₂ Emissions via Aviation Over Years (with Linear Regression)')
plt.grid(True)
plt.show()

# --- 4. Specific DataFrame Preparation (arrivals by region) ---

df_arrivals_by_region = dataframes['arrivals by region']

print("Original data types of df_arrivals_by_region:")
print(df_arrivals_by_region.dtypes)

# Convert 'Year' column to numeric if not already
if not pd.api.types.is_numeric_dtype(df_arrivals_by_region['Year']):
    df_arrivals_by_region['Year'] = pd.to_numeric(df_arrivals_by_region['Year'], errors='coerce')
    print("\n'Year' column converted to numeric.")
else:
    print("\n'Year' column is already numeric.")

print("\nUpdated data types of df_arrivals_by_region:")
print(df_arrivals_by_region.dtypes)

print("\nFirst 5 rows of df_arrivals_by_region:")
print(df_arrivals_by_region.head())

# --- 5. Tourist Arrivals Plot ---

# Get unique regions
unique_regions = df_arrivals_by_region['Entity'].unique()

# Create the plot
plt.figure(figsize=(15, 8)) # Increased figure size for better readability with multiple lines

for region in unique_regions:
    region_data = df_arrivals_by_region[df_arrivals_by_region['Entity'] == region]
    plt.plot(region_data['Year'], region_data['International tourist arrivals by region of origin'], label=region)

plt.xlabel('Year')
plt.ylabel('International Tourist Arrivals')
plt.title('International Tourist Arrivals by Region Over Years')
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside the plot
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6. Combined Trends Data Preparation ---

# Select relevant columns from df_airliner_impact
df_airliner_trends = df_airliner_impact[['Year', 'Passenger demand (billion passenger-km)', 'CO₂ emissions (billion tonnes)']].copy()

# Group df_arrivals_by_region by 'Year' and sum 'International tourist arrivals'
df_tourist_arrivals_aggregated = df_arrivals_by_region.groupby('Year')['International tourist arrivals by region of origin'].sum().reset_index()
df_tourist_arrivals_aggregated.rename(columns={'International tourist arrivals by region of origin': 'International tourist arrivals'}, inplace=True);

# Merge the two DataFrames
df_combined_trends = pd.merge(df_airliner_trends, df_tourist_arrivals_aggregated, on='Year', how='inner')

print("First 5 rows of df_combined_trends:")
print(df_combined_trends.head())

print("\nData types of df_combined_trends:")
print(df_combined_trends.dtypes)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Define the columns to be normalized
columns_to_normalize = [
    'Passenger demand (billion passenger-km)',
    'CO₂ emissions (billion tonnes)',
    'International tourist arrivals'
]

# Apply normalization to the selected columns
df_combined_trends[columns_to_normalize] = scaler.fit_transform(df_combined_trends[columns_to_normalize])

print("First 5 rows of df_combined_trends after normalization:")
print(df_combined_trends.head())

print("\nDescriptive statistics of normalized columns:")
print(df_combined_trends[columns_to_normalize].describe())

# --- 7. Combined Trends Plot ---

# Create the multi-line plot for normalized trends
plt.figure(figsize=(12, 7))

plt.plot(df_combined_trends['Year'], df_combined_trends['Passenger demand (billion passenger-km)'], label='Normalized Passenger Demand')
plt.plot(df_combined_trends['Year'], df_combined_trends['International tourist arrivals'], label='Normalized International Tourist Arrivals')
plt.plot(df_combined_trends['Year'], df_combined_trends['CO₂ emissions (billion tonnes)'], label='Normalized Aviation CO₂ Emissions')

plt.xlabel('Year')
plt.ylabel('Normalized Value (0-1)')
plt.title('Normalized Trends of Passenger Demand, Tourist Arrivals, and Aviation CO₂ Emissions Over Years')
plt.legend()
plt.grid(True)
plt.show()

# --- 8. Temperature and Fire Data Loading (Corrected) ---

temperature_data_path = '/content/drive/My Drive/aviation data/surface_temps.csv'
fire_data_path = '/content/drive/My Drive/aviation data/total fires.xlsx'

print(f"Temperature data path defined as: {temperature_data_path}")
print(f"Fire incident data path defined as: {fire_data_path}")

try:
    df_temperature = pd.read_csv(temperature_data_path)
    print("\nSuccessfully loaded df_temperature:")
    print(df_temperature.head())
    print("\nData types of df_temperature:")
    print(df_temperature.dtypes)
except FileNotFoundError:
    print(f"\nError: Temperature data file not found at {temperature_data_path}. Please ensure the file exists and the path is correct.")
    df_temperature = None
except Exception as e:
    print(f"\nError loading temperature data: {e}")
    df_temperature = None

try:
    df_fire = pd.read_excel(fire_data_path)
    print("\nSuccessfully loaded df_fire:")
    print(df_fire.head())
    print("\nData types of df_fire:")
    print(df_fire.dtypes)
except FileNotFoundError:
    print(f"\nError: Fire incident data file not found at {fire_data_path}. Please ensure the file exists and the path is correct.")
    df_fire = None
except Exception as e:
    print(f"\nError loading fire incident data: {e}")
    df_fire = None

# --- 9. Temperature Data Preparation ---

# Identify non-year columns (id_vars) and year columns (value_vars)
id_vars = ['ObjectId', 'Country', 'ISO2', 'ISO3', 'Indicator', 'Unit', 'Source', 'CTS Code', 'CTS Name', 'CTS Full Descriptor']
year_columns = [col for col in df_temperature.columns if col not in id_vars]

# Melt the DataFrame from wide to long format
df_temp_long = df_temperature.melt(id_vars=id_vars,
                                   value_vars=year_columns,
                                   var_name='Year',
                                   value_name='Temperature')

# Convert 'Year' column to numeric
df_temp_long['Year'] = pd.to_numeric(df_temp_long['Year'], errors='coerce')

# Filter for 'World' country, if it exists
df_world_temperature = df_temp_long[df_temp_long['Country'] == 'World'].copy()

print("First 5 rows of df_world_temperature:")
print(df_world_temperature.head())

print("\nData types of df_world_temperature:")
print(df_world_temperature.dtypes)

# Create 'Decade' column in df_world_temperature
df_world_temperature['Decade'] = (df_world_temperature['Year'] // 10) * 10

print("First 5 rows of df_world_temperature with 'Decade' column:")
print(df_world_temperature.head())

print("\nUnique decades:")
print(df_world_temperature['Decade'].unique())

# --- 10. Temperature Plots ---

# Global Average Temperature Change Plot
plt.figure(figsize=(12, 6))
sns.regplot(x='Year', y='Temperature', data=df_world_temperature, scatter_kws={'alpha':0.6})
plt.xlabel('Year')
plt.ylabel('Average Temperature Change (Degree Celsius)')
plt.title('Global Average Temperature Change Over Years (with Linear Regression)')
plt.grid(True)
plt.show()

# Box plot for Global Average Temperature Change by Decade
plt.figure(figsize=(12, 7))
sns.boxplot(x='Decade', y='Temperature', data=df_world_temperature)
plt.xlabel('Decade')
plt.ylabel('Average Temperature Change (Degree Celsius)')
plt.title('Global Average Temperature Change Distribution by Decade')
plt.grid(True)
plt.show()

# --- 11. Fire Data Preparation ---

# Reload the df_fire DataFrame with the correct header
try:
    df_fire = pd.read_excel(fire_data_path, header=3)
    print("Successfully reloaded df_fire with correct header.")
except FileNotFoundError:
    print(f"\nError: Fire incident data file not found at {fire_data_path}. Please ensure the file exists and the path is correct.")
    df_fire = None
except Exception as e:
    print(f"\nError reloading fire incident data: {e}")
    df_fire = None

if df_fire is not None:
    # Convert 'Year', 'Fires', and 'Acres' columns to numeric
    df_fire['Year'] = pd.to_numeric(df_fire['Year'], errors='coerce')
    df_fire['Fires'] = pd.to_numeric(df_fire['Fires'], errors='coerce')
    df_fire['Acres'] = pd.to_numeric(df_fire['Acres'], errors='coerce')

    # Drop any rows that resulted in NaN after conversion
    df_fire.dropna(subset=['Year', 'Fires', 'Acres'], inplace=True)
    df_fire['Year'] = df_fire['Year'].astype(int)

    # Define thresholds for 'large fires' and 'small fires'
    large_fire_threshold = 1000  # Acres greater than 1000
    small_fire_threshold = 1000  # Acres less than or equal to 1000

    print(f"\nLarge fire threshold (Acres > {large_fire_threshold})")
    print(f"Small fire threshold (Acres <= {small_fire_threshold})")

    print("\nFirst 5 rows of cleaned df_fire:")
    print(df_fire.head())

    print("\nData types of cleaned df_fire:")
    print(df_fire.dtypes)

# Determine the minimum and maximum years
min_year = df_fire['Year'].min()
max_year = df_fire['Year'].max()
print(f"Minimum Year: {min_year}")
print(f"Maximum Year: {max_year}")

# Calculate split_year as the midpoint
split_year = int(min_year + (max_year - min_year) / 2)
print(f"Split Year for temporal periods: {split_year}")

# Calculate the median value of the 'Acres' column as acreage_threshold
acreage_threshold = df_fire['Acres'].median()
print(f"Acreage Threshold (median acres burned): {acreage_threshold:,.2f}")

# Create 'Time Period' column
df_fire['Time Period'] = df_fire['Year'].apply(lambda year: 'Older Period' if year <= split_year else 'Recent Period')

# Create 'Acres Category' column
df_fire['Acres Category'] = df_fire['Acres'].apply(lambda acres: 'High Acres Year' if acres > acreage_threshold else 'Low Acres Year')

print("\nFirst 5 rows of df_fire with new columns:")
print(df_fire.head())

print("\nData types of df_fire:")
print(df_fire.dtypes)

# --- 12. Fire Trend Plots ---

# Total Wildland Fires Plot
plt.figure(figsize=(12, 6))
sns.regplot(x='Year', y='Fires', data=df_fire, scatter_kws={'alpha':0.6})
plt.xlabel('Year')
plt.ylabel('Total Wildland Fires')
plt.title('Trend of Total Wildland Fires Over Years (with Linear Regression)')
plt.grid(True)
plt.show()

# Total Acres Burned Plot
plt.figure(figsize=(12, 6))
sns.regplot(x='Year', y='Acres', data=df_fire, scatter_kws={'alpha':0.6})
plt.xlabel('Year')
plt.ylabel('Total Acres Burned')
plt.title('Trend of Total Acres Burned Over Years (with Linear Regression)')
plt.grid(True)
plt.show()

# Stacked Bar Chart for Total Acres Burned by Time Period and Category
stacked_acres = df_fire.groupby(['Time Period', 'Acres Category'])['Acres'].sum().unstack(fill_value=0)

print("Total Acres Burned by Time Period and Acres Category:")
print(stacked_acres)

plt.figure(figsize=(10, 7))
stacked_acres.plot(kind='bar', stacked=True, figsize=(10, 7), rot=0)

plt.xlabel('Time Period (Years)')
plt.ylabel('Total Acres Burned')
plt.title('Total Acres Burned in High vs. Low Acres Years by Time Period')
x_labels = [
    f"Older Period ({min_year}-{split_year})",
    f"Recent Period ({split_year + 1}-{max_year})"
]
plt.xticks(ticks=[0, 1], labels=x_labels)
plt.legend(title='Acres Category')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# --- 13. Correlation Heatmap ---

# Select the normalized columns for correlation analysis
correlation_data = df_combined_trends[[
    'Passenger demand (billion passenger-km)',
    'CO₂ emissions (billion tonnes)',
    'International tourist arrivals'
]]

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

print("Correlation Matrix:")
print(correlation_matrix)

# Generate a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Normalized Aviation and Tourist Trends')
plt.show()

# --- 14. Additional Diagnostic/Summary ---

print("First 5 rows of df_temperature:")
print(df_temperature.head())

print("\nInformation about df_temperature:")
df_temperature.info()

print("Unique values in 'Indicator' column:")
print(df_temperature['Indicator'].unique())

print("\nUnique values in 'CTS Name' column:")
print(df_temperature['CTS Name'].unique())

print("\nUnique values in 'Country' column (first 20 unique values):")
print(df_temperature['Country'].unique()[:20])

