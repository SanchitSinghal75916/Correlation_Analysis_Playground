
import io
import pandas as pd


def excel_bytes_from_sheets(sheets: dict) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    buffer.seek(0)
    return buffer.read()
