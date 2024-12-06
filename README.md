#     Label-Generation-via-Detection-Model-Predictions

This process outlines how to train a YOLO model using a small annotated dataset, then apply it to an unlabeled image set to expand the annotated dataset. This strategy helps to effectively enhance the model's detection capabilities and reduce the manual annotation workload.

## Steps

### Step 1: Prepare a Small Annotated Dataset to Train the YOLO Model

#### 1.1 [Data Preparation](https://www.makesense.ai/)
- Collect and prepare a small annotated dataset. The images should have corresponding object annotations (bounding boxes) and class labels.
- The annotation format must comply with YOLO’s required format. Each image should have an associated `.txt` annotation file containing the class label and the bounding box coordinates (object position in the image).

#### 1.2 Build Dataset
- Directory Preparation:
  Ensure your annotation files (*.txt) and image files (*.jpg) are paired by filename and follow one of these directory structures:
  Option 1: 
    ```
    original_directory/
      subdirectory1/
        a.txt
        a.jpg
        b.txt
        b.jpg ...
      subdirectory2/
        c.txt
        c.jpg
        d.txt
        d.jpg ...
    ```
  Option 2: 
  ```
    original_directory/
      a.txt
      a.jpg
      b.txt
      b.jpg ...
  ```
- Run the following command to process the files:
   `python build_datasty.py --ori_path {path_to_directory_mentioned_above} --save_path {rebuilt_dataset_saving_path} --train 0.7 --val 0.2`
  Notes: 
  - Renaming Duplicates: Files with the same name across subdirectories are renamed to avoid conflicts.
  - Removing Unpaired Files: Deletes any files without a corresponding pair.
  - Dataset Splitting (in this case): Divides the dataset into training (70%), validation (20%), and test (10%) subsets. 
  
        
### Step 2: YOLO Model Training
#### 2.1 Install and Configure [YOLO](https://github.com/ultralytics/ultralytics) Environment
- Choose a YOLO model version, such as YOLOv5 or YOLOv8, which typically offer better performance and ease of use.
- Set up the YOLO environment locally and install required dependencies such as PyTorch and other necessary libraries.
- Configure YOLO's configuration files, ensuring that the training and test datasets are correctly loaded.

#### 2.2 Model Training
- Train the YOLO model using the prepared small annotated dataset. This is done by specifying the dataset paths, number of classes, and training hyperparameters in YOLO's configuration files.
- The training process will generate an initial detection model, which may have lower performance due to the small dataset size.

### Step 3: Predict on Unlabeled Images

#### 3.1 Use the Initial Trained Model for Prediction
- Apply the trained YOLO model to predict on an unlabeled image set. The predictions will generate detection results, including the object classes and bounding box locations.

### Step 4: Manually Verify and Correct Some Annotations

#### 4.1 Manual Inspection and Correction
- Manually inspect the predicted results to ensure the bounding boxes and object labels are correct. This step is crucial in semi-supervised learning, as the predictions might contain errors.
- You can use tools to visualize the detection results and make necessary corrections. For example, image annotation tools like LabelImg  can be used to adjust or correct the predicted bounding boxes.

#### 4.2 Save Modified Annotations
- Save the corrected annotation results in the YOLO-required format (`.txt` files). These files will be used to expand the training dataset.
