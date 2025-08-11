# Brain Tumor Detection using a Convolutional Neural Network (CNN)

## Overview

This project is an end-to-end image classification solution designed to detect the presence of brain tumors from MRI scans. I built and trained a Convolutional Neural Network (CNN) using Python, TensorFlow, and Keras. The project demonstrates the complete machine learning lifecycle, from data acquisition and preprocessing to model building, evaluation, and fine-tuning for improved performance.

The final model achieved an accuracy of **[e.g., 97%]** and, more importantly, a recall of **[e.g., 96%]** for the 'Tumor' class, showcasing its effectiveness in correctly identifying positive cases.

## Key Achievements & Features

*   **Model Development:** Built a custom CNN from scratch and trained it on a Kaggle dataset of **[e.g., 253]** MRI images.
*   **Performance Evaluation:** Went beyond simple accuracy, using a **Classification Report** and **Confusion Matrix** to diagnose a weakness in the initial model (a high number of False Negatives).
*   **Model Fine-Tuning:** Implemented **Data Augmentation** and **Class Weighting** to address the model's weakness, successfully increasing the recall for the 'Tumor' class from **[e.g., 85%]** to **[e.g., 96%]**.
*   **Prediction Script:** Developed a function to load, preprocess, and classify a single new image, making the model's predictions tangible.

## Technologies Used

*   **Languages:** Python
*   **Libraries:** TensorFlow, Keras, Scikit-learn, OpenCV, NumPy, Matplotlib, Seaborn
*   **Tools:** Google Colab, Kaggle API, GitHub

## Dataset

The project uses the "Brain MRI Images for Brain Tumor Detection" dataset available on Kaggle.
[Link to Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

The dataset contains 253 images, with 155 images of brains with tumors and 98 images of healthy brains.

## Methodology

1.  **Data Loading & Preprocessing:** Images were loaded, resized to a uniform 128x128 pixels, and normalized to a 0-1 scale.
2.  **Initial Model:** A baseline CNN was constructed and trained.
3.  **Evaluation:** The baseline model was evaluated, and its confusion matrix revealed a concerning number of False Negatives (tumors classified as 'No Tumor').
4.  **Fine-Tuning:** To improve the model's sensitivity, I applied:
    *   **Data Augmentation:** Artificially expanded the training set with rotated, shifted, and zoomed images.
    *   **Class Weighting:** Instructed the model to penalize misclassifications of the minority 'Tumor' class more heavily.
5.  **Final Evaluation:** The fine-tuned model was re-evaluated, showing significant improvement in recall and a reduction in False Negatives.

## Results

The fine-tuning process was highly successful. The model's ability to correctly identify actual tumors improved dramatically, which is the most critical metric for a medical diagnosis task.

*(**Action for you:** Take screenshots of your final classification report and confusion matrix. Upload these image files to your GitHub repository, then update the image paths below.)*

**Final Confusion Matrix:**
![Final Confusion Matrix](path/to/your/confusion_matrix_image.png)

**Final Classification Report:**
![Final Classification Report](path/to/your/classification_report_image.png)


## How to Run This Project

1.  Download the `.ipynb` notebook file from this repository.
2.  Open the notebook in Google Colab (`File` -> `Upload notebook...`).
3.  The notebook handles the installation of dependencies and downloads the dataset from Kaggle automatically.
4.  Run the cells sequentially from top to bottom.

## Example Prediction

Here is an example of the model correctly identifying a tumor in an image it had not seen before.

*(**Action for you:** Take a screenshot of one of your single-image predictions, upload the image to your repository, and update the path below.)*
![Example Prediction](path/to/your/prediction_example.png)
