import pandas as pd
import json 
import csv

path_train_df = r"C:\Users\Admin\Documents\Tuan_Huy\UIT_DS_Challenge\data\vihallu-train.csv"
path_test_df = r"C:\Users\Admin\Documents\Tuan_Huy\UIT_DS_Challenge\data\vihallu-public-test.csv"

# Bảng ánh xạ
label_map = {"extrinsic": 0, "intrinsic": 1, "no": 2}

def csv_to_json(csv_file, json_file):
    data = []

    # Đọc file CSV
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Nếu có cột "predict_label" thì đổi thành "labels"
            if "predict_label" in row:
                row["labels"] = row.pop("predict_label")
            elif "label" in row:
                row["labels"] = row.pop("label")

            # Nếu có nhãn thì map sang số
            if "labels" in row:
                if row["labels"] in label_map:
                    row["labels"] = label_map[row["labels"]]
                else:
                    row["labels"] = -1

            data.append(row)

    # Ghi ra file JSON
    with open(json_file, mode='w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Ví dụ sử dụng
csv_to_json(path_train_df, "data/train.json")
csv_to_json(path_test_df, "data/test.json")