# stuff_model.py

import polars as pl
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMRegressor
from sklearn.preprocessing import RobustScaler


df = pl.read_csv("mlb_pitch_data_2020_2024.csv")

# Load the run values data from CSV
df_run_values = pl.read_csv("adj_run_values2.csv")  

# Define a dictionary to group pitch outcomes together
des_dict = {
    'Ball': 'ball',
    'In play, run(s)': 'hit_into_play',
    'In play, out(s)': 'hit_into_play',
    'In play, no out': 'hit_into_play',
    'Called Strike': 'called_strike',
    'Foul': 'foul',
    'Swinging Strike': 'swinging_strike',
    'Blocked Ball': 'ball',
    'Swinging Strike (Blocked)': 'swinging_strike',
    'Foul Tip': 'swinging_strike',
    'Foul Bunt': 'foul',
    'Hit By Pitch': 'hit_by_pitch',
    'Pitchout': 'ball',
    'Missed Bunt': 'swinging_strike',
    'Bunt Foul Tip': 'swinging_strike',
    'Foul Pitchout': 'foul',
    'Ball In Dirt': 'ball'
}

# Define a dictionary to group events together
event_dict = {
    'game_advisory': None,
    'single': 'single',
    'walk': 'walk',
    np.nan: None,
    'strikeout': 'strikeout',
    'field_out': 'field_out',
    'force_out': 'field_out',
    'double': 'double',
    'hit_by_pitch': 'hit_by_pitch',
    'home_run': 'home_run',
    'grounded_into_double_play': 'field_out',
    'fielders_choice_out': 'field_out',
    'fielders_choice': 'field_out',
    'field_error': None,
    'double_play': 'field_out',
    'sac_fly': 'field_out',
    'strikeout_double_play': None,
    'triple': 'triple',
    'caught_stealing_2b': None,
    'sac_bunt': 'field_out',
    'catcher_interf': None,
    'caught_stealing_3b': None,
    'sac_fly_double_play': 'field_out',
    'triple_play': 'field_out',
    'other_out': 'field_out',
    'pickoff_3b': None,
    'caught_stealing_home': None,
    'pickoff_1b': None,
    'pickoff_2b': None,
    'wild_pitch': None,
    'stolen_base_2b': None,
    'pickoff_caught_stealing_3b': None,
    'pickoff_caught_stealing_2b': None,
    'sac_bunt_double_play': None,
    'passed_ball': None,
    'pickoff_caught_stealing_home': None
}

# Join the run values data with the main dataframe based on event type, balls, and strikes
df = df.join(df_run_values, 
            left_on=['event_type', 'balls', 'strikes', 'pitcher_hand', 'batter_hand'],
            right_on=['event', 'balls', 'strikes', 'pitcher_hand', 'batter_hand'], 
            how='left')

# Replace play descriptions with the grouped outcomes from des_dict
df = df.with_columns(pl.col("play_description").replace_strict(des_dict, default=None))

# Join the run values data again based on the play description, balls, and strikes
df = df.join(df_run_values, 
            left_on=['play_description', 'balls', 'strikes', 'pitcher_hand', 'batter_hand'],
            right_on=['event', 'balls', 'strikes', 'pitcher_hand', 'batter_hand'], 
            how='left',
            suffix='_des')

# Assign the target column based on the delta run expectation
df = df.with_columns(
    pl.when(pl.col("delta_run_exp").is_null())
    .then(pl.col("delta_run_exp_des"))
    .otherwise(pl.col("delta_run_exp"))
    .alias("target")
)

def feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    # Extract the year from the game_date column
    df = df.with_columns(
        pl.col('game_date').str.slice(0, 4).alias('year')
    )

    # Mirror horizontal break for left-handed pitchers
    df = df.with_columns(
        pl.when(pl.col('pitcher_hand') == 'L')
        .then(-pl.col('ax'))
        .otherwise(pl.col('ax'))
        .alias('ax')
    )

    # Mirror horizontal release point for left-handed pitchers
    df = df.with_columns(
        pl.when(pl.col('pitcher_hand') == 'L')
        .then(pl.col('x0'))
        .otherwise(-pl.col('x0'))
        .alias('x0')
    )

    # Define the pitch types to be considered
    pitch_types = ['SI', 'FF', 'FC']

    # Filter the DataFrame to include only the specified pitch types
    df_filtered = df.filter(pl.col('pitch_type').is_in(pitch_types))

    # Group by pitcher_id and year, then aggregate to calculate average speed and usage percentage
    df_agg = df_filtered.group_by(['pitcher_id', 'year', 'pitch_type']).agg([ 
        pl.col('start_speed').mean().alias('avg_fastball_speed'),
        pl.col('az').mean().alias('avg_fastball_az'),
        pl.col('ax').mean().alias('avg_fastball_ax'),
        pl.len().alias('count')
    ])

    # Sort the aggregated data by count and average fastball speed
    df_agg = df_agg.sort(['count', 'avg_fastball_speed'], descending=[True, True])
    df_agg = df_agg.unique(subset=['pitcher_id', 'year'], keep='first')

    # Join the aggregated data with the main DataFrame
    df = df.join(df_agg, on=['pitcher_id', 'year'])

    # If no fastball, use the fastest pitch for avg_fastball_speed
    df = df.with_columns(
        pl.when(pl.col('avg_fastball_speed').is_null())
        .then(pl.col('start_speed').max().over('pitcher_id'))
        .otherwise(pl.col('avg_fastball_speed'))
        .alias('avg_fastball_speed')
    )

    # If no fastball, use the fastest pitch for avg_fastball_az
    df = df.with_columns(
        pl.when(pl.col('avg_fastball_az').is_null())
        .then(pl.col('az').max().over('pitcher_id'))
        .otherwise(pl.col('avg_fastball_az'))
        .alias('avg_fastball_az')
    )

    # If no fastball, use the fastest pitch for avg_fastball_ax
    df = df.with_columns(
        pl.when(pl.col('avg_fastball_ax').is_null())
        .then(pl.col('ax').max().over('ax'))
        .otherwise(pl.col('avg_fastball_ax'))
        .alias('avg_fastball_ax')
    )

    # Calculate pitch differentials
    df = df.with_columns(
        (pl.col('start_speed') - pl.col('avg_fastball_speed')).alias('speed_diff'),
        (pl.col('az') - pl.col('avg_fastball_az')).alias('az_diff'),
        (pl.col('ax') - pl.col('avg_fastball_ax')).abs().alias('ax_diff')
    )

    # Cast the year column to integer type
    df = df.with_columns(
        pl.col('year').cast(pl.Int64)
    )

    return df

df = feature_engineering(df.clone())

# Filter the dataframe to include only the years 2020, 2021, 2022, and 2023
df_train = df.filter(pl.col('year').is_in([2020, 2021, 2022, 2023, 2024]))

# Define the features to be used for training
features = ['start_speed',
            'spin_rate',
            'extension',
            'az',
            'ax',
            'x0',
            'z0',
            'speed_diff',
            'az_diff',
            'ax_diff']

# Define the target variable
target = 'target'

# Drop rows with null values in the specified features and target column
df_train = df_train.drop_nulls(subset=features + [target])

# Extract features and target from the training dataframe
X = df_train[features]
y = df_train['target']

# Create a pipeline with RobustScaler and LGBMRegressor
model = make_pipeline(
    RobustScaler(),
    LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        force_row_wise=True
    )
)

# Fit the model to the training data
model.fit(X, y)

# Save the trained model using joblib
joblib.dump(model, 'tjstuff_model.pkl')

print("Model trained and saved successfully!")



