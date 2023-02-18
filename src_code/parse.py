import argparse


def get_train_args():

    parser = argparse.ArgumentParser() #tạo ra trình phân tính để định nghĩa các tham số
    parser.add_argument("data_path", type=str)
    parser.add_argument("--num_leaves", type=int, default=66) #66
    parser.add_argument("--learning_rate", type=float, default=0.2447211446488824) #0.2447211446488824
    parser.add_argument("--reg_lambda", type=float, default=2.3763155117571912) #2.3763155117571912
    parser.add_argument("--reg_alpha", type=float, default=0.8) 
    parser.add_argument("--output_file_name", type=str, default="model5.txt")
    args = parser.parse_args() #đưa các tham số thành một object args dạng namespace
    return vars(args)
# num_leaves: số lượng lá của cây, một siêu tham số quan trọng nhất để kiểm soát độ phức tạp của mô hình.
# Nó được tối ưu để giúp giảm thiểu tình trạng quá khớp (overfitting).
# learning_rate: tỷ lệ học của mô hình, tức là độ lớn của bước cập nhật trong quá trình tối ưu hóa. 
# Giá trị này thường được giảm dần theo thời gian để giúp tối ưu mô hình một cách ổn định hơn.
# reg_lambda: giá trị độ lớn của tham số L2 regularization, được sử dụng để kiểm soát quá trình tối ưu hóa để tránh tình trạng quá khớp. 
# Nó được tối ưu để giúp tăng độ chính xác và tính tổng quát của mô hình.

def get_tune_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    args = parser.parse_args()

    return vars(args)


def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()

    return vars(args)


def get_deploy_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()

    return vars(args)
#include <bits/stdc++.h> 
using namespace std; 
  
int n, k, m; 
int a[10]; 
int cnt = 0; 
  
void Try(int i, int sum) { 
    if (i == n + 1) { 
        if (sum == m && k == 0) { 
            cnt++; 
        } 
        return; 
    } 

    Try(i + 1, sum); //không chọn phần tử thứ i

    if (k > 0 && sum + a[i] <= m) { //chọn phần tử thứ i nếu k > 0 và tổng các phần tử đã chọn <= m.  
        k--; //giá trị còn lại của k giảm đi 1.  

        Try(i + 1, sum + a[i]); //tiêp tuc quay lui và chon phan tu tiep theo.

        k++; //trương hơp không chon phan tu thu i nen gia tri cua k tang len 1.  

    } 

    return; 
} 

  
int main() {     

	cin >> n >> k >> m; //Nhap vao so luong phan tu trong day so va 2 so duong K va M.  

	for (int i = 1; i <= n; i++) { //Nhap vao day so.  

		cin >> a[i];  

	}    

	Try(1, 0); //Goi ham quay lui de tinh toan ket qua.    

	cout << cnt << endl; //In ra ket qua.    

	return 0;    
}