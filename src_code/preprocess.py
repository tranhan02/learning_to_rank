# from parse import get_train_args
import pandas as pd
import numpy as np


def get_data(data_path): #tạo 1 dictionary dfs
    dfs = {
        "train": pd.read_csv(f"{data_path}/train.txt", delimiter=" "), #f để chèn giá trị biến data_path vào đường dẫn đọc tệp tin của các tệp train.txt, vali.txt và test.txt.
        "vali": pd.read_csv(f"{data_path}/vali.txt", delimiter=" "),
        "test": pd.read_csv(f"{data_path}/test.txt", delimiter=" "),
    }

#xóa cột tất cả giá trị đều là NaN
# df.isna() là một DataFrame mới, trong đó mỗi phần tử là True nếu tại vị trí tương ứng trong DataFrame df là NaN, ngược lại là False.
# df.isna().all() là một Series chứa True hoặc False tương ứng với mỗi cột của DataFrame df, True nếu tất cả các phần tử của cột đó đều là NaN, ngược lại là False.
# df.columns[df.isna().all()] là một Index chứa tên các cột trong DataFrame df mà tất cả các phần tử của cột đó đều là NaN.
# .tolist() là phương thức của Index để chuyển Index sang dạng list.
# columns=df.columns[df.isna().all()].tolist() dùng để gán danh sách tên các cột mà chỉ chứa giá trị NaN cho thuộc tính columns của DataFrame df.
# inplace=True là tham số để chỉ định việc thay đổi này được áp dụng trực tiếp trên DataFrame df mà không cần tạo ra một DataFrame mới.
    for df in dfs.values():
        df.columns = np.arange(len(df.columns))
        df.drop(
            columns=df.columns[df.isna().all()].tolist(), inplace=True
        )  

    split = {}

    split["X_train"] = dfs["train"].iloc[:, 1:]
    split["X_val"] = dfs["vali"].iloc[:, 1:]
    split["X_test"] = dfs["test"].iloc[:, 1:]

    y_train = dfs["train"].iloc[:, 0]

    y_val = dfs["vali"].iloc[:, 0]
    y_test = dfs["test"].iloc[:, 0]

    g = split["X_train"].groupby(by=1)
    size = g.size()
    group_train = size.to_list()

    g = split["X_val"].groupby(by=1)
    size = g.size()
    group_vali = size.to_list()

    for name, df in split.items():
        # Loại bỏ số + ":": 1 là chia 1 lần, -1 là lấy phần tử thứ hai
        df = df.applymap(lambda x: x.split(":", 1)[-1] if isinstance(x, str) else x)
        df = df.astype(float)
        df = df.drop(columns=1)
        df.columns = [i for i in range(1, 137)]

        split[name] = df

    return (
        split["X_train"],
        split["X_test"],
        split["X_val"],
        y_train,
        y_test,
        y_val,
        group_vali,
        group_train,
    )
