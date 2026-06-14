"""Download Binance EOHSummary option-chain archives.

Lists every daily zip on data.binance.vision under
data/option/daily/EOHSummary/<SYMBOL>/, downloads in parallel, unzips in memory,
and concatenates to one parquet per symbol in data/external/binance_eoh/.

Schema (per row, per option, per hour):
    date, hour, symbol, underlying, type, strike, open, high, low, close,
    volume_contracts, volume_usdt,
    best_bid_price, best_ask_price, best_bid_qty, best_ask_qty,
    best_buy_iv, best_sell_iv, mark_price, mark_iv,
    delta, gamma, vega, theta,
    openinterest_contracts, openinterest_usdt
"""

from __future__ import annotations

import argparse
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx
import pandas as pd

BASE = "https://data.binance.vision"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "binance_eoh"


def list_zip_keys(client: httpx.Client, symbol: str) -> list[str]:
    r = client.get(
        S3,
        params={
            "prefix": f"data/option/daily/EOHSummary/{symbol}/",
            "delimiter": "/",
        },
        timeout=30.0,
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    return sorted(c.text for c in root.iter(NS + "Key") if c.text.endswith(".zip"))


def fetch_zip(client: httpx.Client, key: str) -> pd.DataFrame:
    r = client.get(f"{BASE}/{key}", timeout=60.0)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            return pd.read_csv(f)


def fetch_symbol(symbol: str, workers: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(http2=False) as client:
        keys = list_zip_keys(client, symbol)
        print(f"{symbol}: {len(keys)} daily zips queued")
        frames: list[pd.DataFrame] = []
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch_zip, client, k): k for k in keys}
            for fut in as_completed(futs):
                key = futs[fut]
                try:
                    frames.append(fut.result())
                except Exception as e:
                    print(f"  FAIL {key}: {e}")
                done += 1
                if done % 20 == 0 or done == len(keys):
                    print(f"  {symbol}: {done}/{len(keys)}")
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "hour", "symbol"])
    out_path = out_dir / f"{symbol}_eoh.parquet"
    df.to_parquet(out_path, index=False)
    print(f"{symbol}: {len(df):,} rows, {df['date'].nunique()} days  →  {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args()
    for sym in args.symbols:
        fetch_symbol(sym, args.workers, args.out)


if __name__ == "__main__":
    main()
