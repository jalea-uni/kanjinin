import os
import json

def create_map(root_dir, etl_dirs, out_file='label_map.json'):
    # Scans ETL directories and lists unique code directories (e.g., '0x8702')
    codes = set()
    for etl in etl_dirs:
        etl_path = os.path.join(root_dir, etl)
        if not os.path.isdir(etl_path):
            continue
        for code_dir in os.listdir(etl_path):
            # Ensure valid hex code
            try:
                int(code_dir, 16)
                codes.add(code_dir.upper())
            except ValueError:
                continue
    # Sort codes by integer value
    sorted_codes = sorted(codes, key=lambda x: int(x, 16))
    # Map index -> code string
    label_map = {str(idx): code for idx, code in enumerate(sorted_codes)}
    with open(out_file, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"Label map written to {out_file} with {len(sorted_codes)} entries.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Root directory of ETL datasets')
    parser.add_argument('--etl-dirs', default="ETL1,ETL2,ETL3,ETL4,ETL5,ETL6,ETL7,ETL8G,ETL9G",
                        help='Comma-separated ETL directory names')
    parser.add_argument('--out', default='label_map.json', help='Output JSON file')
    args = parser.parse_args()
    etl_dirs = args.etl_dirs.split(',')
    create_map(args.root, etl_dirs, args.out)