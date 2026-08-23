"""
data/dealership_data.py
-------------------------
Simulated dealership data backing the MCP tools. In a real system this
would be a database (the dealership platform this mirrors -- Cash4Wheels --
uses Postgres); here it's an in-memory dict so the whole project runs with
zero setup and the tool logic is what's under test, not a DB connection.
"""

from datetime import datetime, timedelta

# VIN -> vehicle record. VINs are fake (not valid check-digit VINs) but
# formatted realistically.
VEHICLE_DB = {
    "1FTFW1ET5DFC10312": {
        "vin": "1FTFW1ET5DFC10312", "make": "Ford", "model": "F-150",
        "year": 2021, "mileage": 34200, "trim": "XLT", "condition": "Good",
    },
    "3GNAXUEV5NL123456": {
        "vin": "3GNAXUEV5NL123456", "make": "Chevrolet", "model": "Equinox",
        "year": 2022, "mileage": 18900, "trim": "LT", "condition": "Excellent",
    },
    "5YJ3E1EA8JF012345": {
        "vin": "5YJ3E1EA8JF012345", "make": "Tesla", "model": "Model 3",
        "year": 2020, "mileage": 41250, "trim": "Long Range", "condition": "Good",
    },
    "1HGCV1F34LA045678": {
        "vin": "1HGCV1F34LA045678", "make": "Honda", "model": "Accord",
        "year": 2023, "mileage": 8100, "trim": "Sport", "condition": "Excellent",
    },
    "2C4RC1BG0FR567890": {
        "vin": "2C4RC1BG0FR567890", "make": "Chrysler", "model": "Town & Country",
        "year": 2015, "mileage": 112400, "trim": "Touring", "condition": "Fair",
    },
}

# Listing DB: listing_id -> current listing state
LISTING_DB = {
    "L-1001": {"listing_id": "L-1001", "vin": "1FTFW1ET5DFC10312", "price": 31900,
               "status": "active", "days_on_lot": 12, "dealer_id": "D-01"},
    "L-1002": {"listing_id": "L-1002", "vin": "3GNAXUEV5NL123456", "price": 26500,
               "status": "active", "days_on_lot": 5, "dealer_id": "D-01"},
    "L-1003": {"listing_id": "L-1003", "vin": "5YJ3E1EA8JF012345", "price": 27900,
               "status": "active", "days_on_lot": 41, "dealer_id": "D-01"},
    "L-1004": {"listing_id": "L-1004", "vin": "1HGCV1F34LA045678", "price": 28400,
               "status": "active", "days_on_lot": 3, "dealer_id": "D-02"},
    "L-1005": {"listing_id": "L-1005", "vin": "2C4RC1BG0FR567890", "price": 9200,
               "status": "active", "days_on_lot": 67, "dealer_id": "D-02"},
}

# Simplified market comp table, keyed by (make, model, year_bucket) ->
# a baseline price per mile-adjusted estimate. Real system would call a
# service like MarketCheck or Black Book; this is a deterministic stand-in.
MARKET_COMP_TABLE = {
    ("Ford", "F-150"): {"base_price": 38000, "depreciation_per_year": 2400, "depreciation_per_1k_miles": 180},
    ("Chevrolet", "Equinox"): {"base_price": 29500, "depreciation_per_year": 1900, "depreciation_per_1k_miles": 140},
    ("Tesla", "Model 3"): {"base_price": 42000, "depreciation_per_year": 3100, "depreciation_per_1k_miles": 160},
    ("Honda", "Accord"): {"base_price": 30500, "depreciation_per_year": 1700, "depreciation_per_1k_miles": 130},
    ("Chrysler", "Town & Country"): {"base_price": 24000, "depreciation_per_year": 1500, "depreciation_per_1k_miles": 90},
}

# Compliance rules a listing must satisfy. Deliberately includes a couple
# of listings that fail these, so the compliance agent has real work to do.
MAX_DAYS_ON_LOT_WARNING = 45
MIN_PRICE_FLOOR_FRACTION_OF_MARKET = 0.55  # price shouldn't be below 55% of estimated market value (data-entry-error signal)
