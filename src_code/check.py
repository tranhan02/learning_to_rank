import os
from parse import get_test_args
import pandas as pd
import numpy as np
import optuna.integration.lightgbm as lgb
import parse

 

def main():
    args = get_test_args()
    if os.path.exists(args["model_path"]):
        print("File exists")
    else:
        print("File does not exist")

if __name__ == "__main__":
    main()