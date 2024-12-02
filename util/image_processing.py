# image_processing.py

import os
import cv2
from ultralytics import YOLO

# 副程式：載入模型
def load_model(pth_path: str):
    """載入 YOLO 模型"""
    return YOLO(pth_path)

# 副程式：檢查並創建資料夾
def create_directory(directory: str):
    """檢查目標資料夾是否存在，不存在則創建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

# 副程式：處理圖片並保存檢測結果（僅儲存圖片）
def process_and_save_image(model, file_path: str, output_path: str,test_path: str):
    """處理圖片，進行檢測並儲存結果圖片"""
    # 使用 YOLO 進行檢測
    results = model(file_path)

    # 讀取圖片並獲取尺寸
    img = cv2.imread(file_path)
    image_h, image_w, _ = img.shape

    # 取得檢測結果並畫在圖片上
    result_img = results[0].plot()

    # 儲存檢測後的圖片
    relative_path = os.path.relpath(os.path.dirname(file_path), test_path)
    target_dir = os.path.join(output_path, relative_path)
    create_directory(target_dir)  # 確保資料夾存在
    output_image_path = os.path.join(target_dir, os.path.basename(file_path))
    cv2.imwrite(output_image_path, result_img)

    print(f"Processed {file_path} -> {output_image_path}")

# 副程式：處理圖片並保存檢測結果（儲存圖片和標籤）
def process_and_save_image_with_labels(model, file_path: str, output_path: str, label_output_path: str, test_path: str):
    """處理圖片，進行檢測並儲存圖片和標籤"""
    # 使用 YOLO 進行檢測
    results = model(file_path)

    # 讀取圖片並獲取尺寸
    img = cv2.imread(file_path)
    image_h, image_w, _ = img.shape

    # 取得檢測結果並畫在圖片上
    result_img = results[0].plot()

    # 儲存檢測後的圖片
    relative_path = os.path.relpath(os.path.dirname(file_path), test_path)
    target_dir = os.path.join(output_path, relative_path)
    create_directory(target_dir)  # 確保資料夾存在
    output_image_path = os.path.join(target_dir, os.path.basename(file_path))
    cv2.imwrite(output_image_path, result_img)

    # 儲存標籤檔案
    label_dir = os.path.join(label_output_path, relative_path)
    create_directory(label_dir)

    label_file_path = os.path.join(label_dir, os.path.basename(file_path).replace(file_path.split('.')[-1], 'txt'))
    with open(label_file_path, 'w') as f:
        for obj in results[0].boxes:
            cls_id = int(obj.cls)
            xmin, ymin, xmax, ymax = obj.xyxy[0]

            # 計算標準化的 YOLO 座標
            x_center = (xmin + xmax) / 2 / image_w
            y_center = (ymin + ymax) / 2 / image_h
            width = (xmax - xmin) / image_w
            height = (ymax - ymin) / image_h

            # 寫入標籤
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

    print(f"Processed {file_path} -> {output_image_path} -> {label_file_path}")

