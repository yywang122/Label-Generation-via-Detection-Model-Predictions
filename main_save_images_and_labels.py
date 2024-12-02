# main_save_images_and_labels.py

import os
import cv2
from util.image_processing import load_model, create_directory, process_and_save_image_with_labels

# 主程式 2：儲存檢測結果的圖片和標籤
def main():
    # 設定路徑
    pth_path = r"/home/cluster/ultralytics/runs/detect/train_pu/weights/best.pt"
    test_path = r"/home/cluster/Downloads/dwgtopdf_image/工程圖_test"
    output_path = r"/home/cluster/Downloads/dwgtopdf_image/工程圖_test_output2"
    label_output_path = r"/home/cluster/Downloads/dwgtopdf_image/工程圖_test_labels2"

    # 載入 YOLO 模型
    model = load_model(pth_path)

    # 創建必要的資料夾
    create_directory(output_path)
    create_directory(label_output_path)

    # 遍歷所有圖片並處理
    for root, dirs, files in os.walk(test_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                file_path = os.path.join(root, file)
                process_and_save_image_with_labels(model, file_path, output_path, label_output_path,test_path)

    cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗

if __name__ == "__main__":
    main()

