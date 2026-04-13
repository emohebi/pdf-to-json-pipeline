import pandas as pd
from datetime import timedelta
from pathlib import Path

# ── Configuration ──
INPUT_FILE = Path(r"C:\Users\hasssa\OneDrive - BHP\Documents\Code\pdf-to-json-pipeline\Concat - 9100000695\all.xlsx")       # <-- change to your file path
OUTPUT_FILE = INPUT_FILE.parent / "weekly.xlsx"     # <-- change to your desired output path
OUTPUT_FILE_NON_WEEKLY = INPUT_FILE.parent / "non_weekly.xlsx"     # <-- change to your desired output path
DATE_FMT = "%d/%m/%Y"
non_weekly_snapshot = True
# ── Load data ──
df = pd.read_excel(INPUT_FILE, dtype=str)

# Parse dates
df["VALID_FROM_dt"] = pd.to_datetime(df["VALID_FROM"], format=DATE_FMT, dayfirst=True)
df["VALID_TO_dt"] = pd.NaT

# ── Sort by ITEM_DESCRIPTION asc, VALID_FROM asc ──
df.sort_values(["ITEM_DESCRIPTION", "VALID_FROM_dt"], ascending=[True, True], inplace=True)
df.reset_index(drop=True, inplace=True)
 
# ── Fill VALID_TO: for each item, current row's VALID_TO = next row's VALID_FROM - 1 day ──
for item, group in df.groupby("ITEM_DESCRIPTION"):
    idxs = group.index.tolist()
    for i in range(len(idxs) - 1):
        next_from = df.loc[idxs[i + 1], "VALID_FROM_dt"]
        if pd.isna(df.loc[idxs[i], "VALID_TO"]):
            df.loc[idxs[i], "VALID_TO_dt"] = (next_from - timedelta(days=1)).strftime(DATE_FMT)
        else:
            df.loc[idxs[i], "VALID_TO_dt"] = pd.to_datetime(df.loc[idxs[i], "VALID_TO"], format=DATE_FMT, dayfirst=True)

today_timestamp = pd.Timestamp.today()

df.loc[pd.isna(df['VALID_TO_dt']), 'VALID_TO_dt'] = today_timestamp.strftime(DATE_FMT)
if non_weekly_snapshot:
    df['VALID_TO'] = df['VALID_TO_dt']
    # ── Cross-contract duplication ──
    # For each CONTRACT_ID, duplicate all rows from every other CONTRACT_ID
    # (replacing CONTRACT_ID with the current one)
    unique_contracts = df["CONTRACT_ID"].unique()
    all_parts = []
    
    for cid in unique_contracts:
        # Take ALL rows (from every contract), set CONTRACT_ID to this one
        block = df.copy()
        block["CONTRACT_ID"] = cid
        all_parts.append(block)
    
    df = pd.concat(all_parts, ignore_index=True)
    df.drop(columns=["VALID_FROM_dt", "VALID_TO_dt"], inplace=True)
    df.sort_values(["CONTRACT_ID", "ITEM_DESCRIPTION", "VALID_FROM"], inplace=True)
    df.to_excel(OUTPUT_FILE_NON_WEEKLY, index=False)
else:
    # ── Weekly expansion ──
    expanded_rows = []
    for _, row in df.iterrows():
        start = row["VALID_FROM_dt"]
        end = row["VALID_TO_dt"]
    
        if pd.isna(end):
            # Last occurrence for this item — keep as-is, no expansion
            new_row = row.copy()
            new_row["VALID_FROM"] = start.strftime(DATE_FMT)
            new_row["VALID_TO"] = ""
            expanded_rows.append(new_row)
        else:
            # Generate one row per week within [start, end]
            week_start = start
            while week_start <= end:
                week_end = min(week_start + timedelta(days=6), end)
                new_row = row.copy()
                new_row["VALID_FROM"] = week_start.strftime(DATE_FMT)
                new_row["VALID_TO"] = week_end.strftime(DATE_FMT)
                expanded_rows.append(new_row)
                week_start += timedelta(days=7)
    
    result = pd.DataFrame(expanded_rows)
    result.drop(columns=["VALID_FROM_dt", "VALID_TO_dt"], inplace=True)
    
    print(f"After weekly expansion: {len(result)} rows.")
    
    # ── Cross-contract duplication ──
    # For each CONTRACT_ID, duplicate all rows from every other CONTRACT_ID
    # (replacing CONTRACT_ID with the current one)
    unique_contracts = result["CONTRACT_ID"].unique()
    all_parts = []
    
    for cid in unique_contracts:
        # Take ALL rows (from every contract), set CONTRACT_ID to this one
        block = result.copy()
        block["CONTRACT_ID"] = cid
        all_parts.append(block)
    
    final = pd.concat(all_parts, ignore_index=True)
    
    # Sort for readability
    final.sort_values(["CONTRACT_ID", "ITEM_DESCRIPTION", "VALID_FROM"], inplace=True)
    final.reset_index(drop=True, inplace=True)
    
    # ── Save ──
    final.to_excel(OUTPUT_FILE, index=False)
    
    print(f"Cross-contract duplication: {len(unique_contracts)} contracts x {len(result)} rows = {len(final)} rows.")
    print(f"Output saved to: {OUTPUT_FILE}")
