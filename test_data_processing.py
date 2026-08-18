"""Tests for the climate project data-processing functions."""

import math

import pandas as pd

from data_processing import calculate_slope
from data_processing import clean_nasa_data
from data_processing import clean_weather_data
from data_processing import get_world_co2
from data_processing import merge_country_co2_weather
from data_processing import merge_temperature_co2
from data_processing import reshape_co2_data
from data_processing import summarize_weather_by_month
from data_processing import (
    test_temperature_period_difference as run_temperature_test
)


def test_clean_nasa_data():
    """Tests numeric conversion and missing NASA measurements."""
    data = pd.DataFrame(
        {
            "Year": ["2000", "2001"],
            "J-D": ["0.25", "***"],
            "DJF": ["0.10", "0.20"],
        }
    )
    cleaned = clean_nasa_data(data)

    assert cleaned["Year"].tolist() == [2000, 2001]
    assert cleaned.loc[0, "J-D"] == 0.25
    assert math.isnan(cleaned.loc[1, "J-D"])


def test_clean_weather_data():
    """Tests date parsing, derived columns, and invalid-row removal."""
    data = pd.DataFrame(
        {
            "country": ["A", "B"],
            "location_name": ["One", "Two"],
            "last_updated": ["2024-05-10 12:00", "invalid"],
            "temperature_celsius": [20.0, 30.0],
            "humidity": [50, 60],
            "precip_mm": [1.0, 2.0],
            "wind_kph": [10.0, 15.0],
        }
    )
    cleaned = clean_weather_data(data)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "year"] == 2024
    assert cleaned.loc[0, "month"] == 5


def test_reshape_co2_data():
    """Tests indicator filtering and wide-to-long reshaping."""
    data = pd.DataFrame(
        {
            "REF_AREA": ["WLD", "WLD"],
            "REF_AREA_LABEL": ["World", "World"],
            "INDICATOR": ["OWID_CB_CO2", "OTHER"],
            "2000": [100.0, 999.0],
            "2001": [110.0, 999.0],
        }
    )
    reshaped = reshape_co2_data(data, 2000, 2001)

    assert len(reshaped) == 2
    assert reshaped["CO2_Emissions"].tolist() == [100.0, 110.0]
    assert reshaped["INDICATOR"].eq("OWID_CB_CO2").all()


def test_merge_temperature_co2():
    """Tests that only overlapping years remain after a merge."""
    nasa = pd.DataFrame(
        {
            "Year": [2000, 2001, 2002],
            "J-D": [0.1, 0.2, 0.3],
        }
    )
    co2 = pd.DataFrame(
        {
            "REF_AREA_LABEL": ["World", "World"],
            "Year": [2001, 2002],
            "CO2_Emissions": [100.0, 110.0],
        }
    )
    world = get_world_co2(co2)
    merged = merge_temperature_co2(nasa, world)

    assert merged["Year"].tolist() == [2001, 2002]
    assert merged["J-D"].tolist() == [0.2, 0.3]


def test_merge_country_co2_weather():
    """Tests combining country emissions with weather averages."""
    co2 = pd.DataFrame(
        {
            "REF_AREA_LABEL": ["China", "United States"],
            "Year": [2021, 2021],
            "CO2_Emissions": [100.0, 50.0],
        }
    )
    weather = pd.DataFrame(
        {
            "country": [
                "China",
                "China",
                "United States of America",
            ],
            "temperature_celsius": [20.0, 30.0, 10.0],
            "humidity": [50.0, 70.0, 40.0],
            "precip_mm": [1.0, 3.0, 2.0],
            "wind_kph": [10.0, 20.0, 15.0],
        }
    )
    countries = ["China", "United States"]
    merged = merge_country_co2_weather(
        co2,
        weather,
        countries,
    )

    assert len(merged) == 2
    china = merged[merged["REF_AREA_LABEL"] == "China"]
    assert china.iloc[0]["temperature_celsius"] == 25.0


def test_summarize_weather_by_month():
    """Tests monthly averaging with known values."""
    weather = pd.DataFrame(
        {
            "year": [2024, 2024],
            "month": [5, 5],
            "temperature_celsius": [10.0, 20.0],
            "humidity": [40.0, 60.0],
            "precip_mm": [0.0, 2.0],
            "wind_kph": [5.0, 15.0],
        }
    )
    summary = summarize_weather_by_month(weather)

    assert summary.loc[0, "temperature_celsius"] == 15.0
    assert summary.loc[0, "humidity"] == 50.0
    assert summary.loc[0, "precip_mm"] == 1.0
    assert summary.loc[0, "wind_kph"] == 10.0


def test_calculate_slope():
    """Tests a slope calculation with an exact linear pattern."""
    data = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [2, 4, 6],
        }
    )
    slope = calculate_slope(data, "x", "y")

    assert math.isclose(slope, 2.0)


def test_temperature_period_difference():
    """Tests the temperature-period significance calculation."""
    data = pd.DataFrame(
        {
            "Year": [
                1950,
                1951,
                1952,
                1953,
                2000,
                2001,
                2002,
                2003,
            ],
            "J-D": [
                0.0,
                0.1,
                -0.1,
                0.0,
                2.0,
                2.1,
                1.9,
                2.0,
            ],
        }
    )
    early_mean, recent_mean, statistic, p_value = (
        run_temperature_test(data)
    )

    assert math.isclose(early_mean, 0.0, abs_tol=1e-9)
    assert math.isclose(recent_mean, 2.0, abs_tol=1e-9)
    assert statistic > 0
    assert p_value < 0.05


def main():
    """Runs every data-processing test."""
    test_clean_nasa_data()
    test_clean_weather_data()
    test_reshape_co2_data()
    test_merge_temperature_co2()
    test_merge_country_co2_weather()
    test_summarize_weather_by_month()
    test_calculate_slope()
    test_temperature_period_difference()
    print("All tests passed!")


if __name__ == "__main__":
    main()
