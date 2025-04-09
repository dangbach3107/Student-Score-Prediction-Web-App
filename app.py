import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
df = pd.read_csv("data.csv", sep=";")

# Chuẩn bị dữ liệu
X = df[["Midterm", "Attendance", "StudyHours", "Assignments"]]
y = df["FinalScore"]

# Huấn luyện model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("🎓 Dự Đoán Điểm Cuối Kỳ")

st.write("Nhập thông tin của bạn:")

midterm = st.slider("Điểm giữa kỳ", 0.0, 10.0, 7.0)
attendance = st.slider("Tỉ lệ đi học (%)", 0, 100, 90)
study_hours = st.slider("Giờ ôn bài", 0, 20, 10)
assignments = st.slider("Số bài tập đã làm", 0, 10, 4)

# Dự đoán khi nhấn nút
if st.button("🔮 Dự đoán điểm"):
    input_data = pd.DataFrame([[midterm, attendance, study_hours, assignments]],
                              columns=["Midterm", "Attendance", "StudyHours", "Assignments"])
    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Dự đoán điểm cuối kỳ: **{prediction:.2f}**")
