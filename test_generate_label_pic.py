import cv2

# 类别名称对应的列表
class_names = ['major_title', 'minor_title', 'article', 'item', 'trash']

def draw_boxes(image_path, label_file, output_image_path):
    # 读取图片
    image = cv2.imread(image_path)
    
    # 打开YOLO的标注文件
    with open(label_file, 'r') as f:
        lines = f.readlines()
        
    # 获取图片的宽度和高度
    h, w, _ = image.shape
    
    # 遍历每一行标注
    for line in lines:
        # YOLO格式：class x_center y_center width height
        parts = line.strip().split()
        class_id = int(parts[0])  # 获取类别id
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        box_width = float(parts[3]) * w
        box_height = float(parts[4]) * h
        
        # 计算左上角和右下角的坐标
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)
        
        # 绘制矩形框
        color = (0, 255, 0)  # 矩形框颜色，可以根据类别不同指定不同的颜色
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # 添加类别名称
        label = class_names[class_id]  # 使用类别名称
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # 白色文字
        thickness = 1
        
        # 获取文字的宽高
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_width, text_height = text_size
        
        # 绘制背景矩形框来提高文字可读性
        cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        
        # 在框上方显示类别名称
        cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
    
    # 保存修改后的图片
    cv2.imwrite(output_image_path, image)
    print(f"Image saved to {output_image_path}")

# 使用示例
#image_path = '/home/cluster/Desktop/yolo_dataset_0905/train/images/A5_地坪平面圖(20240118)_1F (施工範圍)_page_1.jpg'  # 图片路径
#label_file = '/home/cluster/Desktop/yolo_dataset_0905/train/labels/A5_地坪平面圖(20240118)_1F (施工範圍)_page_1.txt'  # YOLO格式的标注文件路径
image_path = '/home/cluster/Downloads/dwgtopdf_image/工程圖_test/2023/7S122GC1/00.__A1_20230606__竣工圖_A1-1.jpg'  # 图片路径
label_file = '/home/cluster/Downloads/dwgtopdf_image/工程圖_test_labels/2023/7S122GC1/00.__A1_20230606__竣工圖_A1-1.txt'  # YOLO格式的标注文件路径
output_image_path = './00.__A1_20230606__竣工圖_A1-1_output_image.jpg'  # 输出的图片路径
draw_boxes(image_path, label_file,output_image_path)

