import os
import sys
import argparse
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_drought_0528_ms import get_0528_ms_paths
from datasets.dataset_drought import read_envi_band

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--bad_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    good_rows = []
    bad_rows = []

    keys = ["nir", "red", "blue", "green", "rededge"]

    for _, row in df.iterrows():
        sid = row["id"]
        label = row["label"]
        paths = get_0528_ms_paths(sid, args.data_root)

        ok = True
        for key in keys:
            hdr_path = paths[key]
            dat_path = hdr_path[:-4] + ".dat"

            try:
                _ = read_envi_band(hdr_path, band_idx=0)
            except Exception as e:
                ok = False
                bad_rows.append({
                    "id": sid,
                    "label": label,
                    "key": key,
                    "hdr_path": hdr_path,
                    "hdr_exists": os.path.exists(hdr_path),
                    "dat_path": dat_path,
                    "dat_exists": os.path.exists(dat_path),
                    "dat_size": os.path.getsize(dat_path) if os.path.exists(dat_path) else -1,
                    "error": repr(e),
                })
                break

        if ok:
            good_rows.append(row.to_dict())

    good_df = pd.DataFrame(good_rows)
    bad_df = pd.DataFrame(bad_rows)

    good_df.to_csv(args.out_csv, index=False)
    bad_df.to_csv(args.bad_csv, index=False)

    print("=" * 80)
    print("0528 readable filter done")
    print("=" * 80)
    print(f"original samples: {len(df)}")
    print(f"good samples    : {len(good_df)}")
    print(f"bad samples     : {len(bad_df)}")
    print(f"saved good csv  : {args.out_csv}")
    print(f"saved bad csv   : {args.bad_csv}")

    if len(good_df) > 0:
        print("\nGood label distribution:")
        print(good_df["label"].value_counts().sort_index())

    if len(bad_df) > 0:
        print("\nBad sample preview:")
        print(bad_df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
