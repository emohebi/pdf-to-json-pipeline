"""
UOM (Unit of Measure) normalisation mapping.

Maps raw UOM strings from invoices/timesheets to standardised codes.
Lookup is case-insensitive.  Values not found here are left as-is.
Junk values (OCR artefacts, summary labels) are mapped to None.
"""

# Canonical UOM codes used in the normalised output
# Key = lowercased raw value, Value = normalised code (or None for junk)
UOM_MAP: dict[str, str | None] = {
    # Hours
    "hours": "HR",
    "hour": "HR",
    "hr": "HR",
    "hrs": "HR",
    "hur": "HR",
    "per hour": "HR",
    "p/h": "HR",
    "per hr": "HR",
    "/hr": "HR",
    "/hour": "HR",
    "hourly": "HR",
    # Days
    "days": "DAY",
    "day": "DAY",
    "dy": "DAY",
    "dys": "DAY",
    "daily": "DAY",
    "per day": "DAY",
    "p/d": "DAY",
    # Each
    "ea": "EA",
    "each": "EA",
    "unit": "EA",
    "units": "EA",
    "pc": "EA",
    "pcs": "EA",
    "piece": "EA",
    "pieces": "EA",
    "item": "EA",
    "no": "EA",
    "no.": "EA",
    # Lot / Lump sum
    "lot": "LOT",
    "lo": "LOT",
    "ls": "LOT",
    "lump sum": "LOT",
    "lump": "LOT",
    "lumpsum": "LOT",
    # Weeks
    "week": "WK",
    "weeks": "WK",
    "wk": "WK",
    "wks": "WK",
    "per week": "WK",
    # Months
    "month": "MTH",
    "months": "MTH",
    "mth": "MTH",
    "mths": "MTH",
    "per month": "MTH",
    "monthly": "MTH",
    "mo": "MTH",
    # Weight
    "tonne": "T",
    "tonnes": "T",
    "ton": "T",
    "tons": "T",
    "t": "T",
    "tl": "T",
    "mt": "T",
    "kg": "KG",
    "kgs": "KG",
    "kilogram": "KG",
    "kilograms": "KG",
    "lb": "LB",
    "lbs": "LB",
    # Volume
    "litre": "L",
    "litres": "L",
    "liter": "L",
    "liters": "L",
    "l": "L",
    "lt": "L",
    "ltr": "L",
    "ml": "ML",
    "kl": "KL",
    "kilolitre": "KL",
    # Length
    "metre": "M",
    "metres": "M",
    "meter": "M",
    "meters": "M",
    "m": "M",
    "km": "KM",
    "mm": "MM",
    # Area / Volume
    "m2": "M2",
    "sqm": "M2",
    "sq m": "M2",
    "m3": "M3",
    "cum": "M3",
    # Rolls
    "roll": "ROLL",
    "rolls": "ROLL",
    "rol": "ROLL",
    # Shifts
    "shift": "SHIFT",
    "shifts": "SHIFT",
    # Junk / OCR artefacts — clear to None
    "0": None,
    "o": None,
    "0 . 0": None,
    "0.0": None,
    "subtotal": None,
    "total": None,
    "fifo": None,
    "cab surface": None,
    "n/a": None,
    "-": None,
    "": None,
}
