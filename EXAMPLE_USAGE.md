# Cách chạy AutoML Workflow với dữ liệu Iris

## Chuẩn bị

1. Đảm bảo bạn đã có file `iris_data.csv` trong thư mục `data/`
2. Đảm bảo các thư viện cần thiết đã được cài đặt:
   - pandas
   - scikit-learn
   - numpy

## Cách chạy

### Phương pháp 1: Chạy trực tiếp script đơn giản (Đã test và hoạt động)

```bash
cd automl-agent
python src/simple_main.py
```

### Phương pháp 2: Chạy qua script ví dụ (Đã test và hoạt động)

```bash
cd automl-agent
python src/run_example_simple.py
```

Cả hai script sẽ tự động:
1. Tải dữ liệu từ `data/iris_data.csv`
2. Tiền xử lý dữ liệu (chia train/test)
3. Huấn luyện cả hai mô hình: Random Forest và SVM
4. Đánh giá hiệu suất trên tập test
5. Hiển thị báo cáo kết quả

### Phương pháp 3: Chạy với LLM (Cần cài đặt thư viện và cấu hình LLM)

```bash
cd automl-agent
python src/run_example.py
```

*Lưu ý: Phương pháp này yêu cầu cài đặt đầy đủ các thư viện LangChain và LangGraph, cũng như cấu hình LLM.*

## Kết quả mong đợi

Khi chạy thành công, bạn sẽ thấy output tương tự như sau:

```
Running AutoML pipeline with Iris dataset (Simple Version)
============================================================
Using data file: C:\Users\ASUS\Learning-LangGraph\automl-agent\data\iris_data.csv
============================================================
Running AutoML pipeline for file: C:\Users\ASUS\Learning-LangGraph\automl-agent\data\iris_data.csv
Loading and preprocessing data...
Data loaded with shape: (150, 5)
Data preprocessing completed.
Training models...
Model training completed.
Evaluating models...
Model evaluation completed.

==================================================
AutoML Pipeline Completed Successfully!
==================================================
Data Processing:
  - Original data shape: (150, 4)
  - Training data shape: (120, 4)
  - Test data shape: (30, 4)

Model Training Results:
  - Random Forest: Training Accuracy = 1.0000
  - SVM: Training Accuracy = 0.9750

Model Evaluation Results:
  - Random Forest: Test Accuracy = 1.0000
  - SVM: Test Accuracy = 1.0000

Pipeline completed successfully!

AutoML pipeline completed successfully!
```

## Giải thích kết quả

- **Data Processing**: Thông tin về kích thước dữ liệu trước và sau khi chia train/test
- **Model Training Results**: Độ chính xác trên tập huấn luyện
- **Model Evaluation Results**: Độ chính xác trên tập kiểm tra
- Với bộ dữ liệu iris, cả hai mô hình đều đạt độ chính xác 100% trên tập test

## Tùy chỉnh

Bạn có thể tùy chỉnh các tham số trong script:
- Thay đổi tỷ lệ chia train/test
- Thay đổi hyperparameters của các mô hình
- Thêm các mô hình mới
- Thay đổi metric đánh giá

## Khắc phục sự cố

Nếu gặp lỗi khi chạy các script:

1. **Kiểm tra sự tồn tại của file dữ liệu**:
   - Đảm bảo file `iris_data.csv` tồn tại trong thư mục `data/`

2. **Kiểm tra thư viện**:
   - Đảm bảo các thư viện cần thiết đã được cài đặt

3. **Kiểm tra đường dẫn**:
   - Đảm bảo bạn đang chạy script từ thư mục `automl-agent`

4. **Với phiên bản LLM**:
   - Cần cài đặt đầy đủ thư viện LangChain, LangGraph
   - Cần cấu hình LLM trong file `src/config/llm_config.py`