import os
import cv2
from ultralytics import YOLO

# 定義模型和資料夾路徑
pth_path = r"/home/cluster/ultralytics/runs/detect/train_pu/weights/best.pt"
test_path = r"/home/cluster/Downloads/dwgtopdf_image/工程圖_test"
output_path = r"/home/cluster/Downloads/dwgtopdf_image/工程圖_test_output"  # 新的輸出資料夾
label_output_path = r"/home/cluster/Downloads/dwgtopdf_image/工程圖_test_labels"  # 儲存標籤的資料夾

# 初始化 YOLO 模型
model = YOLO(pth_path)

# 檢查輸出資料夾是否存在，不存在則創建
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 檢查標籤資料夾是否存在，不存在則創建
if not os.path.exists(label_output_path):
    os.makedirs(label_output_path)

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

            # 提取檢測結果
            img = cv2.imread(file_path)  # 讀取圖片
            image_h, image_w, _ = img.shape  # 取得圖片的尺寸

            # 儲存檢測後的圖片到新資料夾
            result_img = results[0].plot()  # 繪製檢測結果在圖片上
            output_image_path = os.path.join(target_dir, file)
            cv2.imwrite(output_image_path, result_img)

            # 生成對應的 YOLO 標籤檔案 (.label)
            label_dir = os.path.join(label_output_path, relative_path)  # 在標籤資料夾中創建相對路徑
            if not os.path.exists(label_dir):  # 如果標籤資料夾中的相對路徑不存在，則創建
                os.makedirs(label_dir)

            label_file_path = os.path.join(label_dir, file.replace(file.split('.')[-1], 'txt'))  # 將圖片檔案名替換為 label

            # 儲存標籤
            with open(label_file_path, 'w') as f:
                for obj in results[0].boxes:
                    # 獲取檢測到的物體的類別 ID 和邊界框座標
                    cls_id = int(obj.cls)  # 類別 ID
                    xmin, ymin, xmax, ymax = obj.xyxy[0]  # 取得邊界框的 xmin, ymin, xmax, ymax
                    

                    # 使用提供的公式計算標準化的座標
                    x_center = (xmin + xmax) / 2 / image_w  # 中心 x 座標
                    y_center = (ymin + ymax) / 2 / image_h  # 中心 y 座標
                    width = (xmax - xmin) * 1.0 / image_w
                    height = (ymax - ymin) * 1.0 / image_h

                    # 寫入 YOLO 標籤格式
                    f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

            print(f"Processed {file_path} -> {output_image_path} -> {label_file_path}")

