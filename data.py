import pybaseball


def get_raw_data(start_year, end_year=2024, min_AB=100):
    """Fetch raw batting statistics from pybaseball"""
    pybaseball.cache.enable()
    return pybaseball.batting_stats(start_year, end_year, qual=min_AB)


def get_prepared_data(start_year=2015):
    START_YEAR = start_year
    cols = [
        "Season",
        "Name",
        "Age",
        "G",
        "AB",
        "PA",
        "H",
        "1B",
        "2B",
        "3B",
        "HR",
        "R",
        "RBI",
        "BB",
        "IBB",
        "SO",
        "AVG",
        "SB",
        "WAR",
    ]
    all_data = get_raw_data(START_YEAR - 1).sort_values(["Name", "Season"])
    to_model = all_data[cols]
    to_model["next_WAR"] = to_model.groupby("Name")["WAR"].shift(
        -1
    )  # Prediction target
    # Last year WAR
    to_model["last_WAR"] = to_model.groupby("Name")["WAR"].shift(1).fillna(0.2)
    model_cols = [
        "Season",
        "Name",
        "Age",
        "PA",
        "2B",
        "HR",
        "BB",
        "AVG",
        "WAR",
        "next_WAR",
        "last_WAR",
    ]
    data = to_model[model_cols]
    return data
