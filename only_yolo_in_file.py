import os
import shutil
import cv2
from ultralytics import YOLO

# 定義模型和資料夾路徑
pth_path = r"/home/cluster/ultralytics/runs/detect/train_pu/weights/best.pt"
test_path = r"/home/cluster/Downloads/dwgtopdf_image/工程圖_test"
output_path = r"/home/cluster/Downloads/dwgtopdf_image/工程圖_test_output"  # 新的輸出資料夾

# 初始化 YOLO 模型
model = YOLO(pth_path)

# 檢查輸出資料夾是否存在，不存在則創建
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 遞迴遍歷所有子資料夾和檔案
for root, dirs, files in os.walk(test_path):
    for file in files:
        # 檢查是否為圖片檔案（可以根據需要擴展）
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            # 構建原始檔案和目標檔案的完整路徑
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, test_path)  # 相對路徑
            target_dir = os.path.join(output_path, relative_path)  # 新資料夾路徑

            # 如果目標資料夾不存在，則創建
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # 使用 YOLO 進行檢測
            results = model(file_path)  # 檢測圖片

            # 將檢測結果圖像提取出來
            result_img = results[0].plot()  # 把檢測結果畫到原始圖片上

            # 顯示檢測結果圖片
            cv2.imshow('Detection Result', result_img)

            # 等待按鍵來關閉顯示窗口 (等待1000ms)
            cv2.waitKey(1000)

            # 儲存檢測後的圖片到新資料夾
            output_image_path = os.path.join(target_dir, file)
            cv2.imwrite(output_image_path, result_img)

            print(f"Processed {file_path} -> {output_image_path}")

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()

