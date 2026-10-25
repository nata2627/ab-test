from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/criteo-research-uplift-v2.1.csv.gz"
PARQUET = ROOT / "data/criteo.parquet"

FEATURES = [f"f{i}" for i in range(12)]
LABELS = ["treatment", "conversion", "visit", "exposure"]
SCHEMA = {**{c: pa.float32() for c in FEATURES}, **{c: pa.int8() for c in LABELS}}


def csv_to_parquet(src: Path = RAW, dst: Path = PARQUET) -> int:
    """Потоковая конвертация: в памяти один блок csv, а не весь файл."""
    reader = pv.open_csv(
        src,
        read_options=pv.ReadOptions(block_size=64 << 20),
        convert_options=pv.ConvertOptions(column_types=SCHEMA),
    )
    rows = 0
    with pq.ParquetWriter(dst, reader.schema, compression="zstd") as writer:
        for batch in reader:
            writer.write_batch(batch)
            rows += batch.num_rows
    return rows


def load(columns: list[str]):
    return pq.read_table(PARQUET, columns=columns).to_pandas()


if __name__ == "__main__":
    print(csv_to_parquet())
