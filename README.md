# ETS: Finance Forecasting & Regime Transfer MLOps Platform

Một nền tảng MLOps chuyên dụng để dự báo tài chính (Finance Forecasting) và phân tích sự chuyển dịch trạng thái thị trường (Regime Transfer), được xây dựng với kiến trúc hiện đại, tập trung vào việc quản lý vòng đời Machine Learning: từ xử lý dữ liệu, huấn luyện mô hình, kiểm thử Walk-Forward đến phân tích và so sánh kết quả.

![ETS Dashboard Demo](https://via.placeholder.com/1200x600.png?text=ETS+MLOps+Dashboard+Demo)

## ✨ Tính năng nổi bật

- **Quản lý Dữ liệu (Datasets)**: Tải lên, lưu trữ và theo dõi các tập dữ liệu tài chính (vd: VN30). Tự động trích xuất đặc trưng (Feature Engineering) và cấu trúc dữ liệu.
- **Quản lý Thử nghiệm (Experiments)**: Tạo và theo dõi các thử nghiệm, liên kết với từng bộ dữ liệu cụ thể.
- **Mô hình hóa Đa dạng (Modeling)**:
  - **LightGBM**: Base Model (tiền huấn luyện), Transfer Model (tinh chỉnh theo Regime) và Ensemble Model (kết hợp).
  - **Baselines**: Hỗ trợ các baseline chuẩn như XGBoost, Static LGB.
  - **Deep Learning**: Tích hợp các mô hình dựa trên chuỗi thời gian như LSTM và Transformer (yêu cầu PyTorch).
- **Kiểm thử Walk-Forward**: Thực hiện backtest với cơ chế cửa sổ trượt (Expanding Window) tránh data leakage, cho phép đánh giá hiệu suất thực tế trên chuỗi thời gian.
- **Dynamic Hyperparameter Tuning**: Cấu hình và tinh chỉnh trực tiếp các siêu tham số (như `learning_rate`, `max_depth`) từ giao diện người dùng trước mỗi lần chạy (Run).
- **So sánh & Phân tích (Run Comparison)**: Trực quan hóa kết quả bằng Plotly. Hỗ trợ biểu đồ Cột (Bar Chart) cho các chỉ số tốt nhất và biểu đồ Phân tán (Scatter Plot) để phân tích tương quan giữa siêu tham số và độ chính xác.

## 🛠 Công nghệ sử dụng

- **Backend**: Python 3.9+, FastAPI, SQLAlchemy (SQLite), Pydantic. Xử lý hàng đợi bất đồng bộ với BackgroundTasks.
- **Machine Learning**: LightGBM, XGBoost, Scikit-learn, PyTorch, Pandas, Numpy.
- **Frontend**: React.js (CDN), CSS thuần (Modern UI/UX), Plotly.js để vẽ biểu đồ.

## 🚀 Hướng dẫn cài đặt và chạy ứng dụng

### 1. Yêu cầu hệ thống
- Python 3.9 hoặc mới hơn.
- Trình duyệt web hiện đại (Chrome, Edge, Firefox).

### 2. Cài đặt Backend

Clone repository về máy:
```bash
git clone https://github.com/your-username/ets-finance-mlops.git
cd ets-finance-mlops
```

Tạo môi trường ảo và cài đặt thư viện:
```bash
python -m venv .venv
# Kích hoạt môi trường (Windows)
.venv\Scripts\activate
# Hoặc trên Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

*(Tuỳ chọn) Nếu bạn muốn chạy các mô hình Deep Learning (LSTM, Transformer), hãy cài đặt thêm PyTorch tương thích với hệ thống của bạn.*

### 3. Khởi động Server

Chạy lệnh sau từ thư mục gốc của dự án:
```bash
# Thiết lập PYTHONPATH và khởi động FastAPI server
$env:PYTHONPATH="backend"
python -m ets.api.main
```

Server sẽ khởi chạy tại: `http://localhost:8000`

### 4. Truy cập Giao diện Web

Mở trình duyệt và truy cập vào địa chỉ: [http://localhost:8000/](http://localhost:8000/)

- **Đăng nhập / Đăng ký**: Tạo một tài khoản cục bộ để bắt đầu.
- **Datasets**: Vào tab Datasets để tải lên bộ dữ liệu CSV (ví dụ: dữ liệu giá chứng khoán VN30).
- **Experiments**: Tạo Experiment mới, gán Dataset vừa tải lên.
- **Runs**: Trong Experiment, bạn có thể chỉnh sửa `LR` (Learning Rate) và `Depth` (Max Depth), sau đó nhấn **Trigger Run** để bắt đầu quá trình huấn luyện và đánh giá Walk-Forward.

## 📁 Cấu trúc thư mục

```
📦 qlib-main
 ┣ 📂 backend
 ┃ ┣ 📂 db                 # Định nghĩa Models và Database Session (SQLAlchemy)
 ┃ ┣ 📂 ets                # Core API, Event Bus, Tracking Service, REST Routes
 ┃ ┣ 📂 plugins            # Các plugin thực thi Pipeline (FinanceForecastingPipeline)
 ┃ ┗ 📂 vn_regime_transfer # Logic Machine Learning, Walk-forward validation, Models
 ┣ 📂 frontend
 ┃ ┗ 📜 index.html         # Giao diện Web SPA React.js
 ┣ 📜 ets.db               # Cơ sở dữ liệu SQLite (tự động tạo)
 ┗ 📜 README.md
```

## 🤝 Đóng góp

Mọi đóng góp (Issues, Pull Requests) đều được chào đón! Vui lòng mở một Issue để thảo luận về tính năng bạn muốn thêm vào hoặc lỗi bạn gặp phải trước khi tạo Pull Request.

## 📄 Giấy phép (License)

Dự án này được phân phối dưới giấy phép [MIT License](LICENSE).
