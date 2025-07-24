# Leukemia Classification and Segmentation from Microscopic Images
This project presents a deep learning pipeline for the early diagnosis of Acute Lymphoblastic Leukemia (ALL) from peripheral blood smear images. It features a comparative study of various CNN architectures to build a robust and accurate system for both classifying and segmenting leukemic cells.

https://github.com/user-attachments/assets/02fdfeae-970f-4dd7-a90e-1cc654c95a4d

## Situation
Acute Lymphoblastic Leukemia (ALL) is the most common childhood cancer, but a significant gap in survival rates exists between high-income and low-to-middle-income countries (LMICs) like Indonesia. In many LMICs, access to advanced, rapid, and cost-effective diagnostic tools is limited, leading to diagnostic delays and poorer patient outcomes. Traditional diagnosis via manual inspection of blood smears is time-consuming and subject to human error, creating a critical need for an automated, reliable, and accessible solution.

## Objectives
The primary goal of this research is to develop an accurate, cost-effective, and clinically viable AI-assisted diagnostic tool to enhance the early detection of ALL.

The key objectives include:
1. Develop a high-accuracy classification model to distinguish between healthy and malignant cells by comparing state-of-the-art CNN architectures.
2. Create a precise segmentation model to accurately delineate the boundaries of leukemic cells, even in complex, overlapping scenarios.
3. Build an optimized end-to-end pipeline that integrates advanced data preprocessing, augmentation, and model training strategies suitable for deployment in resource-limited clinical environments.

## Approaches
To achieve these objectives, I developed a comprehensive pipeline using a dataset of 3,256 peripheral blood smear images.

#### Classification Task
- **Architectures Compared:** A systematic comparison was conducted between InceptionResNetV2 (a manually engineered architecture) and NASNetLarge (an architecture designed via Neural Architecture Search).
- **Data Handling:** Implemented a Single Global Dataset Creation (SGDC) strategy for data loading and splitting, which proved more effective than class-specific splitting. The dataset was balanced, and data augmentation was used to improve model robustness.
- **Outcome:** InceptionResNetV2 consistently outperformed NASNetLarge, demonstrating superior accuracy and stability for this specific medical imaging task.

#### Segmentation Task
Baseline Model: Started with the well-established U-Net architecture, which is highly effective for biomedical image segmentation.

Architectural Enhancements:
- Integrated a Feature Pyramid Network (FPN) into the U-Net encoder to improve multi-scale feature learning and better handle cells of varying sizes.
- Replaced standard ReLU activation functions with Gaussian Error Linear Unit (GELU) within the FPN blocks to improve gradient flow and feature integration.
- Outcome: The resulting hybrid U-Net-FPN with GELU model showed significant improvements in delineating cell boundaries, especially for overlapping cells.

## Impacts
The proposed deep learning pipeline demonstrates a robust, accurate, and scalable solution with significant potential for clinical application.

- **High Accuracy:** The InceptionResNetV2 classification model achieved 98.16% accuracy. The enhanced U-Net-FPN segmentation model reached 99.17% accuracy and a dice score of 92.05%.
- **State-of-the-Art Performance:** Both models outperformed results from several prominent prior studies, validating the effectiveness of our architectural choices and training methodologies.
- **Clinical Relevance:** This work provides a foundation for a cost-effective and reliable diagnostic tool that can be deployed in resource-limited settings like Indonesia. By enabling earlier and more accurate detection of ALL, this system has the potential to bridge existing diagnostic gaps and ultimately improve patient outcomes.
- **Future Work:** The next step is to deploy these models into a practical, web-based application to make the technology accessible to clinicians.
