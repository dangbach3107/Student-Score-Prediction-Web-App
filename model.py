import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
df = pd.read_csv("data.csv", sep=";")

# In ra 5 dòng đầu để xem thử
print("📊 Dữ liệu mẫu:")
print(df.head())

# Thống kê tổng quan
print("\n📈 Thống kê mô tả:")
print(df.describe())

# Vẽ biểu đồ mối quan hệ giữa Midterm và FinalScore
plt.scatter(df["Midterm"], df["FinalScore"])
plt.xlabel("Midterm Score")
plt.ylabel("Final Score")
plt.title("Mối quan hệ giữa Midterm và Final")
plt.grid(True)
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Chia dữ liệu thành X (features) và y (label)
X = df[["Midterm", "Attendance", "StudyHours", "Assignments"]]
y = df["FinalScore"]

# 2. Chia tiếp thành train/test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Khởi tạo và huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Dự đoán
y_pred = model.predict(X_test)

# 5. In kết quả
print("\n🔮 Dự đoán thử:")
for i in range(len(y_test)):
    print(f"Thực tế: {y_test.values[i]:.2f} - Dự đoán: {y_pred[i]:.2f}")

# 6. In hệ số mô hình (optional)
print("\n📐 Hệ số mô hình:")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.4f}")
