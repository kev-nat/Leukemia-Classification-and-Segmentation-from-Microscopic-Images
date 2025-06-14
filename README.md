# Leukemia Classification and Segmentation from Microscopic Images

https://github.com/user-attachments/assets/02fdfeae-970f-4dd7-a90e-1cc654c95a4d

## Dataset
<p align="justify"> 

The dataset used in this study was obtained from the [Kaggle repository](https://www.kaggle.com/datasets/mehradaria/leukemia), specifically focusing on the classification and segmentation of Acute Lymphoblastic Leukemia (ALL) and its subtypes. The dataset comprises 3,256 microscopic images derived from 89 patients under investigation for ALL at the bone marrow laboratory of Taleqani Hospital in Tehran, Iran. Among these patients, 25 were diagnosed as healthy with benign conditions (hematogones), while 64 were confirmed to have various ALL subtypes.  
</p>

<div align="center">
  <h4>An Abstract Overview of the Classification Dataset Used in the Proposed Research</h4>
</div>

<div align="center">
<table>
  <thead>
    <tr style="border-top: 2px solid black; border-bottom: 1px solid black;">
      <th style="text-align: left; padding: 8px;">Type</th>
      <th style="text-align: left; padding: 8px;">Subtypes</th>
      <th style="text-align: left; padding: 8px;">No. of Samples</th>
      <th style="text-align: left; padding: 8px;">No. of Patients</th>
      <th style="text-align: left; padding: 8px;">Image Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; padding: 8px;">Benign</td>
      <td style="text-align: left; padding: 8px;">Hematogones</td>
      <td style="text-align: left; padding: 8px;">504</td>
      <td style="text-align: left; padding: 8px;">25</td>
      <td rowspan="5" style="text-align: left; padding: 8px;">224 x 224 px</td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: left; padding: 8px;">Malignant</td>
      <td style="text-align: left; padding: 8px;">Early Pre-B ALL</td>
      <td style="text-align: left; padding: 8px;">985</td>
      <td style="text-align: left; padding: 8px;">20</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px;">Pre-B ALL</td>
      <td style="text-align: left; padding: 8px;">963</td>
      <td style="text-align: left; padding: 8px;">21</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px;">Pro-B ALL</td>
      <td style="text-align: left; padding: 8px;">804</td>
      <td style="text-align: left; padding: 8px;">23</td>
    </tr>
    <tr style="border-top: 1px solid black; border-bottom: 2px solid black;">
      <td style="text-align: left; padding: 8px; font-weight: bold;">Total</td>
      <td></td>
      <td style="text-align: left; padding: 8px;">3256</td>
      <td style="text-align: left; padding: 8px;">89</td>
    </tr>
  </tbody>
</table>
</div>

<p align="justify"> The blood smear samples were prepared and stained following standard laboratory protocols by qualified laboratory personnel. The dataset is systematically categorized into two main classifications: benign and malignant. The benign category encompasses hematogones, which are physiologically normal B-lymphocyte precursors typically present in healthy bone marrow. While these cells share morphological similarities with lymphoblasts found in ALL, they are benign in nature and typically resolve spontaneously without chemotherapeutic intervention. The malignant category is further subdivided into three distinct lymphoblast subtypes: Early Pre-B, Pre-B, and Pro-B ALL. Representative images from each category are presented below. </p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/904289b9-c81e-4429-92bf-f89ff607fdc5" alt="image" />
</p>

## Classification Task
<p align="justify"> The classification phase of this study employs a comparative analysis between 2 sophisticated deep learning architectures: InceptionResNetV2 and NASNetLarge. This comparative approach is particularly significant as it explores the relative effectiveness of human-engineered architectures (InceptionResNetV2) versus neural architecture search-based models (NASNetLarge) in the context of medical image classification. The investigation specifically focuses on assessing these architectures' capabilities in terms of classification accuracy, computational efficiency, and scalability potential. The model demonstrating superior performance across the established metrics, with particular emphasis on maintaining consistent high accuracy across all classes, will be selected for detailed analysis and implementation. This methodological approach ensures a robust and empirically sound basis for model selection in the context of ALL classification. </p>

### Data Preprocessing (Data Augmentation & Balancing Imbalanced Classes)
<p align="justify">
Data augmentation plays a crucial role in enhancing the robustness and generalization capabilities of deep learning models, particularly in medical image analysis applications. This technique involves the systematic generation of varied training samples through carefully selected transformations while preserving the essential semantic information within the images. In this study, the training dataset underwent five distinct types of augmentation transformations: brightness adjustment, contrast adjustment, rotation, JPEG noise injection, and random horizontal and vertical flips as it shown in the table below. These transformations were specifically selected to introduce controlled variability that simulates real-world imaging conditions and potential artifacts, thereby improving the model's ability to generalize across diverse clinical scenarios.
</p>

<div align="center">
  <h4>Comparison of Data Loading Approaches for Model Training</h4>
</div>

<div align="center">
<table>
  <thead>
    <tr style="border-top: 2px solid black; border-bottom: 1px solid black;">
      <th style="text-align: left; padding: 8px;">Augmentation</th>
      <th style="text-align: left; padding: 8px;">Range/Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; padding: 8px;">Brightness</td>
      <td style="text-align: left; padding: 8px;">[-10%, +10%]</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px;">Contrast</td>
      <td style="text-align: left; padding: 8px;">[-10%, +10%]</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px;">Rotation</td>
      <td style="text-align: left; padding: 8px;">[-20°, +20°]</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px;">JPEG noise</td>
      <td style="text-align: left; padding: 8px;">[50, 100]</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px;">Flip</td>
      <td style="text-align: left; padding: 8px;">Horizontal, Vertical</td>
    </tr>
  </tbody>
</table>
</div>
  
<p align="justify">
Importantly, all applied transformations were designed to maintain the clinical interpretability of the images, ensuring that the augmented samples remain consistent with standard medical diagnostic criteria used by physicians and specialists in clinical settings. The Peripheral Blood Smear (PBS) images in the dataset maintain their original classification into two primary categories: malignant and benign (hematogones). 
</p>

<p align="justify">
The malignant B-cell Acute Lymphoblastic Leukemia (B-ALL) samples comprise three main subtypes: Early Pre-B ALL, Pre-B ALL, and Pro-B ALL. These samples contain various cellular components, with particular emphasis on blast cells, which may present as either cancerous lymphoblasts or normal lymphocytes. Figure below illustrates a random selection of unsegmented images following the preprocessing phase.
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/97a78315-505f-4d23-80c3-688b10cc224e" alt="image" />
</p>

<p align="justify">
Class imbalance represents a significant challenge in the dataset, necessitating appropriate balancing techniques to mitigate potential model bias and ensure effective generalization across all classes in the classification task. To address this imbalance, a systematic approach was implemented to augment the representation of each class to achieve 1,000 images per category, resulting in a balanced dataset of 4,000 images in total. This balancing strategy was specifically designed to prevent model bias towards overrepresented classes and enhance the model's discriminative capabilities across both malignant and benign categories. The balanced distribution ensures equal representation during the training phase, leading to more reliable and unbiased classification performance.
</p>

### Different Data Loading Approaches for Model Training
<p align="justify">
The classification task necessitates a systematic approach to data organization, involving the careful loading, preprocessing, and partitioning of the dataset into three distinct subsets: training, validation, and test sets. Two methodological approaches were considered for this process: Class-Specific Data Preparation and Balanced Splitting (Option 1), and Single Global Dataset Creation with Direct Splitting (Option 2). The selection of the optimal approach is guided by critical considerations of class distribution maintenance and data utilization efficiency. Each methodology presents distinct advantages and limitations, particularly in terms of maintaining class balance across all data partitions while ensuring efficient data handling. This structured approach to data preparation is fundamental to establishing a robust foundation for the subsequent model training and evaluation phases.
</p>

<div align="center">
  <h4>Comparison of Different Data Loading Approaches for Model Training</h4>
</div>

<div align="center">
<table>
  <thead>
    <tr style="border-top: 2px solid black; border-bottom: 1px solid black;">
      <th style="text-align: left; padding: 8px;">Aspects</th>
      <th style="text-align: left; padding: 8px;">Class-Specific Data Preparation and Balanced Splitting</th>
      <th style="text-align: left; padding: 8px;">Single Global Dataset Creation with Direct Splitting</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; padding: 8px;">Class-Level Management</td>
      <td style="text-align: left; padding: 8px;">Manages classes independently, allowing for better control and easier adjustments.</td>
      <td style="text-align: left; padding: 8px;">All images and labels are stored globally, simplifying dataset preparation.</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px;">Class Balance in Splitting</td>
      <td style="text-align: left; padding: 8px;">Ensures balanced splitting across training, validation, and test sets (700, 150, 150 images per class).</td>
      <td style="text-align: left; padding: 8px;">No explicit control over class balance, but uses 70%, 15%, 15% split of the global dataset.</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px;">Shuffling</td>
      <td style="text-align: left; padding: 8px;">Shuffles within each class and globally to ensure randomness.</td>
      <td style="text-align: left; padding: 8px;">Shuffles globally after combining the dataset.</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px;">Memory Usage</td>
      <td style="text-align: left; padding: 8px;">Higher memory usage due to class-specific lists.</td>
      <td style="text-align: left; padding: 8px;">More memory efficient as images are combined in one list.</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px;">Code Simplicity</td>
      <td style="text-align: left; padding: 8px;">More complex due to manual splitting and class-specific management.</td>
      <td style="text-align: left; padding: 8px;">Simpler, but may result in imbalanced splits due to randomness.</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px;">Flexibility</td>
      <td style="text-align: left; padding: 8px;">More flexible for class-specific operations, but less adaptable to dataset changes.</td>
      <td style="text-align: left; padding: 8px;">Easier to scale and modify but lacks class-specific adjustments.</td>
    </tr>
  </tbody>
</table>
</div>

### Results and Analysis
<p align="justify">
All these tables shows 48 experiments comparing InceptionResNetV2 and NASNetLarge architectures for classification. The experiments were performed using a Kaggle notebook with P100 GPU, maintaining consistent parameters across all trials: batch size of 32, 60 epochs, and Adam optimizer. The primary variables investigated were learning rate, dropout rate, and data loading configurations.
</p>

<div align="center">
  <h4>Experimental Results of the InceptionResNetV2 Model using Data Loading Option 1</h4>
</div>

<div align="center">
<table>
  <thead>
    <tr>
      <th colspan="2" style="text-align: center">Parameters</th>
      <th colspan="3" style="text-align: center">Classification Report</th>
      <th colspan="2" style="text-align: center">Training Score (%)</th>
      <th colspan="2" style="text-align: center">Validation Score (%)</th>
      <th colspan="2" style="text-align: center">Test Score (%)</th>
    </tr>
    <tr>
      <th style="text-align: center">Learning Rate</th>
      <th style="text-align: center">Dropout</th>
      <th style="text-align: center">Precision</th>
      <th style="text-align: center">Recall</th>
      <th style="text-align: center">F1-Score</th>
      <th style="text-align: center">Accuracy</th>
      <th style="text-align: center">Loss</th>
      <th style="text-align: center">Accuracy</th>
      <th style="text-align: center">Loss</th>
      <th style="text-align: center">Accuracy</th>
      <th style="text-align: center">Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" style="text-align: center">0.1</td>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.06</td>
      <td style="text-align: center">0.25</td>
      <td style="text-align: center">0.10</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
    </tr>
    <tr>
      <td style="text-align: center">0.2</td>
      <td style="text-align: center">0.06</td>
      <td style="text-align: center">0.25</td>
      <td style="text-align: center">0.10</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
    </tr>
    <tr>
      <td style="text-align: center">0</td>
      <td style="text-align: center">0.06</td>
      <td style="text-align: center">0.25</td>
      <td style="text-align: center">0.10</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center">0.01</td>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.06</td>
      <td style="text-align: center">0.25</td>
      <td style="text-align: center">0.10</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
    </tr>
    <tr>
      <td style="text-align: center">0.2</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">97.57</td>
      <td style="text-align: center">7.33</td>
      <td style="text-align: center">97.00</td>
      <td style="text-align: center">7.40</td>
      <td style="text-align: center">96.67</td>
      <td style="text-align: center">8.94</td>
    </tr>
    <tr>
      <td style="text-align: center">0</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">99.96</td>
      <td style="text-align: center">0.56</td>
      <td style="text-align: center">98.00</td>
      <td style="text-align: center">4.44</td>
      <td style="text-align: center">97.67</td>
      <td style="text-align: center">4.79</td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center">0.001</td>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">99.18</td>
      <td style="text-align: center">2.90</td>
      <td style="text-align: center">98.00</td>
      <td style="text-align: center">5.53</td>
      <td style="text-align: center">97.67</td>
      <td style="text-align: center">5.56</td>
    </tr>
    <tr>
      <td style="text-align: center">0.2</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">99.82</td>
      <td style="text-align: center">0.92</td>
      <td style="text-align: center">98.33</td>
      <td style="text-align: center">3.75</td>
      <td style="text-align: center">97.33</td>
      <td style="text-align: center">4.71</td>
    </tr>
    <tr>
      <td style="text-align: center">0</td>
      <td style="text-align: center">0.99</td>
      <td style="text-align: center">0.99</td>
      <td style="text-align: center">0.99</td>
      <td style="text-align: center">99.86</td>
      <td style="text-align: center">1.15</td>
      <td style="text-align: center">98.67</td>
      <td style="text-align: center">4.00</td>
      <td style="text-align: center">97.50</td>
      <td style="text-align: center">5.40</td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center"><b>0.0001</b></td>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">98.25</td>
      <td style="text-align: center">5.75</td>
      <td style="text-align: center">97.33</td>
      <td style="text-align: center">7.28</td>
      <td style="text-align: center">97.50</td>
      <td style="text-align: center">7.05</td>
    </tr>
    <tr>
      <td style="text-align: center">0.2</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">98.79</td>
      <td style="text-align: center">4.39</td>
      <td style="text-align: center">97.67</td>
      <td style="text-align: center">6.25</td>
      <td style="text-align: center">97.83</td>
      <td style="text-align: center">6.51</td>
    </tr>
    <tr>
      <td style="text-align: center"><b>0</b></td>
      <td style="text-align: center"><b>0.99</b></td>
      <td style="text-align: center"><b>0.99</b></td>
      <td style="text-align: center"><b>0.99</b></td>
      <td style="text-align: center"><b>99.43</b></td>
      <td style="text-align: center"><b>3.41</b></td>
      <td style="text-align: center"><b>99.00</b></td>
      <td style="text-align: center"><b>3.65</b></td>
      <td style="text-align: center"><b>97.83</b></td>
      <td style="text-align: center"><b>4.94</b></td>
    </tr>
  </tbody>
</table>
</div>

<div align="center">
  <h4>Experimental Results of the NASNetLarge Model using Data Loading Option 1</h4>
</div>

<table>
  <thead>
    <tr>
      <th colspan="2" style="text-align: center">Parameters</th>
      <th colspan="3" style="text-align: center">Classification Report</th>
      <th colspan="2" style="text-align: center">Training Score (%)</th>
      <th colspan="2" style="text-align: center">Validation Score (%)</th>
      <th colspan="2" style="text-align: center">Test Score (%)</th>
    </tr>
    <tr>
      <th style="text-align: center">Learning Rate</th>
      <th style="text-align: center">Dropout</th>
      <th style="text-align: center">Precision</th>
      <th style="text-align: center">Recall</th>
      <th style="text-align: center">F1-Score</th>
      <th style="text-align: center">Accuracy</th>
      <th style="text-align: center">Loss</th>
      <th style="text-align: center">Accuracy</th>
      <th style="text-align: center">Loss</th>
      <th style="text-align: center">Accuracy</th>
      <th style="text-align: center">Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" style="text-align: center">0.1</td>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.06</td>
      <td style="text-align: center">0.25</td>
      <td style="text-align: center">0.10</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
    </tr>
    <tr>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.06</td>
      <td style="text-align: center">0.25</td>
      <td style="text-align: center">0.10</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
    </tr>
    <tr>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.06</td>
      <td style="text-align: center">0.25</td>
      <td style="text-align: center">0.10</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
      <td style="text-align: center">25.00</td>
      <td style="text-align: center">1208.86</td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center">0.01</td>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">97.57</td>
      <td style="text-align: center">6.59</td>
      <td style="text-align: center">95.67</td>
      <td style="text-align: center">11.25</td>
      <td style="text-align: center">95.83</td>
      <td style="text-align: center">12.82</td>
    </tr>
    <tr>
      <td style="text-align: center">0.2</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">97.79</td>
      <td style="text-align: center">6.05</td>
      <td style="text-align: center">95.83</td>
      <td style="text-align: center">11.79</td>
      <td style="text-align: center">96.33</td>
      <td style="text-align: center">9.67</td>
    </tr>
    <tr>
      <td style="text-align: center">0</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">99.96</td>
      <td style="text-align: center">0.41</td>
      <td style="text-align: center">97.67</td>
      <td style="text-align: center">12.53</td>
      <td style="text-align: center">96.17</td>
      <td style="text-align: center">11.50</td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center">0.001</td>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">99.11</td>
      <td style="text-align: center">3.22</td>
      <td style="text-align: center">96.50</td>
      <td style="text-align: center">9.33</td>
      <td style="text-align: center">96.50</td>
      <td style="text-align: center">8.42</td>
    </tr>
    <tr>
      <td style="text-align: center">0.2</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">99.79</td>
      <td style="text-align: center">1.14</td>
      <td style="text-align: center">97.17</td>
      <td style="text-align: center">6.86</td>
      <td style="text-align: center">96.67</td>
      <td style="text-align: center">7.36</td>
    </tr>
    <tr>
      <td style="text-align: center">0</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">99.93</td>
      <td style="text-align: center">1.07</td>
      <td style="text-align: center">98.17</td>
      <td style="text-align: center">6.90</td>
      <td style="text-align: center">97.17</td>
      <td style="text-align: center">9.47</td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center"><b>0.0001</b></td>
      <td style="text-align: center"><b>0.4</b></td>
      <td style="text-align: center"><b>0.98</b></td>
      <td style="text-align: center"><b>0.98</b></td>
      <td style="text-align: center"><b>0.98</b></td>
      <td style="text-align: center"><b>99.93</b></td>
      <td style="text-align: center"><b>1.07</b></td>
      <td style="text-align: center"><b>98.17</b></td>
      <td style="text-align: center"><b>6.90</b></td>
      <td style="text-align: center"><b>97.17</b></td>
      <td style="text-align: center"><b>9.47</b></td>
    </tr>
    <tr>
      <td style="text-align: center">0.2</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">99.46</td>
      <td style="text-align: center">2.93</td>
      <td style="text-align: center">96.50</td>
      <td style="text-align: center">9.04</td>
      <td style="text-align: center">96.17</td>
      <td style="text-align: center">8.12</td>
    </tr>
    <tr>
      <td style="text-align: center">0</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">99.68</td>
      <td style="text-align: center">2.39</td>
      <td style="text-align: center">97.17</td>
      <td style="text-align: center">8.58</td>
      <td style="text-align: center">96.67</td>
      <td style="text-align: center">8.41</td>
    </tr>
  </tbody>
</table>

<div align="center">
  <h4>Experimental Results of the InceptionResNetV2 Model using Data Loading Option 2</h4>
</div>

<table>
  <thead>
    <tr>
      <th colspan="2" style="text-align: center">Parameters</th>
      <th colspan="3" style="text-align: center">Classification Report</th>
      <th colspan="2" style="text-align: center">Training Score (%)</th>
      <th colspan="2" style="text-align: center">Validation Score (%)</th>
      <th colspan="2" style="text-align: center">Test Score (%)</th>
    </tr>
    <tr>
      <th style="text-align: center">Learning Rate</th>
      <th style="text-align: center">Dropout</th>
      <th style="text-align: center">Precision</th>
      <th style="text-align: center">Recall</th>
      <th style="text-align: center">F1-Score</th>
      <th style="text-align: center">Accuracy</th>
      <th style="text-align: center">Loss</th>
      <th style="text-align: center">Accuracy</th>
      <th style="text-align: center">Loss</th>
      <th style="text-align: center">Accuracy</th>
      <th style="text-align: center">Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" style="text-align: center">0.1</td>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.06</td>
      <td style="text-align: center">0.25</td>
      <td style="text-align: center">0.10</td>
      <td style="text-align: center">24.64</td>
      <td style="text-align: center">1214.61	</td>
      <td style="text-align: center">24.83	</td>
      <td style="text-align: center">1211.54	</td>
      <td style="text-align: center">26.83	</td>
      <td style="text-align: center">1179.31</td>
    </tr>
    <tr>
      <td style="text-align: center">0.2</td>
      <td style="text-align: center">0.06</td>
      <td style="text-align: center">0.25</td>
      <td style="text-align: center">0.10</td>
      <td style="text-align: center">25.18	</td>
      <td style="text-align: center">1205.98	</td>
      <td style="text-align: center">24.83	</td>
      <td style="text-align: center">1211.54	</td>
      <td style="text-align: center">24.33	</td>
      <td style="text-align: center">1219.60</td>
    </tr>
    <tr>
      <td style="text-align: center">0</td>
      <td style="text-align: center">0.06</td>
      <td style="text-align: center">0.25</td>
      <td style="text-align: center">0.10</td>
      <td style="text-align: center">24.64	</td>
      <td style="text-align: center">1214.61	</td>
      <td style="text-align: center">24.83	</td>
      <td style="text-align: center">1211.54	</td>
      <td style="text-align: center">26.83	</td>
      <td style="text-align: center">1179.31</td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center">0.01</td>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.07</td>
      <td style="text-align: center">0.25</td>
      <td style="text-align: center">0.10</td>
      <td style="text-align: center">24.89	</td>
      <td style="text-align: center">1210.58	</td>
      <td style="text-align: center">26.67	</td>
      <td style="text-align: center">1181.99	</td>
      <td style="text-align: center">23.83	</td>
      <td style="text-align: center">1227.66</td>
    </tr>
    <tr>
      <td style="text-align: center">0.2</td>
      <td style="text-align: center">0.06	</td>
      <td style="text-align: center">0.25	</td>
      <td style="text-align: center">0.10	</td>
      <td style="text-align: center">25.29	</td>
      <td style="text-align: center">1204.25	</td>
      <td style="text-align: center">23.67	</td>
      <td style="text-align: center">1230.35	</td>
      <td style="text-align: center">25.00	</td>
      <td style="text-align: center">1208.86</td>
    </tr>
    <tr>
      <td style="text-align: center">0</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">99.75	</td>
      <td style="text-align: center">0.93	</td>
      <td style="text-align: center">98.00	</td>
      <td style="text-align: center">5.61	</td>
      <td style="text-align: center">98.17	</td>
      <td style="text-align: center">5.97</td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center">0.001</td>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">98.89	</td>
      <td style="text-align: center">3.55	</td>
      <td style="text-align: center">97.67	</td>
      <td style="text-align: center">6.04	</td>
      <td style="text-align: center">97.33	</td>
      <td style="text-align: center">6.03</td>
    </tr>
    <tr>
      <td style="text-align: center">0.2</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">99.50	</td>
      <td style="text-align: center">2.10	</td>
      <td style="text-align: center">98.17	</td>
      <td style="text-align: center">4.91	</td>
      <td style="text-align: center">97.67	</td>
      <td style="text-align: center">4.93</td>
    </tr>
    <tr>
      <td style="text-align: center">0</td>
      <td style="text-align: center">0.97	</td>
      <td style="text-align: center">0.97	</td>
      <td style="text-align: center">0.97	</td>
      <td style="text-align: center">99.57	</td>
      <td style="text-align: center">2.23	</td>
      <td style="text-align: center">97.50	</td>
      <td style="text-align: center">5.96	</td>
      <td style="text-align: center">98.17	</td>
      <td style="text-align: center">4.79</td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center"><b>0.0001</b></td>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.98	</td>
      <td style="text-align: center">0.98	</td>
      <td style="text-align: center">0.98	</td>
      <td style="text-align: center">98.50	</td>
      <td style="text-align: center">4.75	</td>
      <td style="text-align: center">97.67	</td>
      <td style="text-align: center">6.61	</td>
      <td style="text-align: center">97.17	</td>
      <td style="text-align: center">6.99</td>
    </tr>
    <tr>
      <td style="text-align: center">0.2</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">99.04	</td>
      <td style="text-align: center">3.59	</td>
      <td style="text-align: center">97.83	</td>
      <td style="text-align: center">6.32	</td>
      <td style="text-align: center">97.67	</td>
      <td style="text-align: center">5.66</td>
    </tr>
    <tr>
      <td style="text-align: center"><b>0</b></td>
      <td style="text-align: center"><b>0.98	</b></td>
      <td style="text-align: center"><b>0.98	</b></td>
      <td style="text-align: center"><b>0.98	</b></td>
      <td style="text-align: center"><b>99.86	</b></td>
      <td style="text-align: center"><b>1.81	</b></td>
      <td style="text-align: center"><b>97.50	</b></td>
      <td style="text-align: center"><b>6.00	</b></td>
      <td style="text-align: center"><b>98.50	</b></td>
      <td style="text-align: center"><b>4.73</b></td>
    </tr>
  </tbody>
</table>

<div align="center">
  <h4>Experimental Results of the NASNetLarge Model using Data Loading Option 2</h4>
</div>

<table>
  <thead>
    <tr>
      <th colspan="2" style="text-align: center">Parameters</th>
      <th colspan="3" style="text-align: center">Classification Report</th>
      <th colspan="2" style="text-align: center">Training Score (%)</th>
      <th colspan="2" style="text-align: center">Validation Score (%)</th>
      <th colspan="2" style="text-align: center">Test Score (%)</th>
    </tr>
    <tr>
      <th style="text-align: center">Learning Rate</th>
      <th style="text-align: center">Dropout</th>
      <th style="text-align: center">Precision</th>
      <th style="text-align: center">Recall</th>
      <th style="text-align: center">F1-Score</th>
      <th style="text-align: center">Accuracy</th>
      <th style="text-align: center">Loss</th>
      <th style="text-align: center">Accuracy</th>
      <th style="text-align: center">Loss</th>
      <th style="text-align: center">Accuracy</th>
      <th style="text-align: center">Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" style="text-align: center">0.1</td>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.22	</td>
      <td style="text-align: center">0.39	</td>
      <td style="text-align: center">0.28	</td>
      <td style="text-align: center">40.57	</td>
      <td style="text-align: center">957.46	</td>
      <td style="text-align: center">40.33	</td>
      <td style="text-align: center">961.71	</td>
      <td style="text-align: center">37.33	</td>
      <td style="text-align: center">1010.07</td>
    </tr>
    <tr>
      <td style="text-align: center">0.2</td>
      <td style="text-align: center">0.06</td>
      <td style="text-align: center">0.25</td>
      <td style="text-align: center">0.10</td>
      <td style="text-align: center">24.64	</td>
      <td style="text-align: center">1214.61	</td>
      <td style="text-align: center">24.83	</td>
      <td style="text-align: center">1211.54	</td>
      <td style="text-align: center">26.83	</td>
      <td style="text-align: center">1179.31</td>
    </tr>
    <tr>
      <td style="text-align: center">0</td>
      <td style="text-align: center">0.06</td>
      <td style="text-align: center">0.25</td>
      <td style="text-align: center">0.10</td>
      <td style="text-align: center">25.18	</td>
      <td style="text-align: center">1205.98	</td>
      <td style="text-align: center">24.83	</td>
      <td style="text-align: center">1211.54	</td>
      <td style="text-align: center">24.33	</td>
      <td style="text-align: center">1219.60</td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center">0.01</td>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.95	</td>
      <td style="text-align: center">0.95	</td>
      <td style="text-align: center">0.95	</td>
      <td style="text-align: center">98.29	</td>
      <td style="text-align: center">5.12	</td>
      <td style="text-align: center">95.00	</td>
      <td style="text-align: center">11.52	</td>
      <td style="text-align: center">94.50	</td>
      <td style="text-align: center">16.05</td>
    </tr>
    <tr>
      <td style="text-align: center">0.2</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">99.71	</td>
      <td style="text-align: center">1.18	</td>
      <td style="text-align: center">95.67	</td>
      <td style="text-align: center">14.44	</td>
      <td style="text-align: center">95.17	</td>
      <td style="text-align: center">19.37</td>
    </tr>
    <tr>
      <td style="text-align: center">0</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">99.71	</td>
      <td style="text-align: center">1.21	</td>
      <td style="text-align: center">96.50	</td>
      <td style="text-align: center">15.07	</td>
      <td style="text-align: center">95.33	</td>
      <td style="text-align: center">14.79</td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center"><b>0.001</b></td>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">99.50	</td>
      <td style="text-align: center">1.95	</td>
      <td style="text-align: center">96.50	</td>
      <td style="text-align: center">11.71	</td>
      <td style="text-align: center">95.50	</td>
      <td style="text-align: center">11.63</td>
    </tr>
    <tr>
      <td style="text-align: center"><b>0.2</b></td>
      <td style="text-align: center"><b>0.97</b></td>
      <td style="text-align: center"><b>0.97</b></td>
      <td style="text-align: center"><b>0.97</b></td>
      <td style="text-align: center"><b>99.75</b></td>
      <td style="text-align: center"><b>1.67</b></td>
      <td style="text-align: center"><b>96.83</b></td>
      <td style="text-align: center"><b>11.92</b></td>
      <td style="text-align: center"><b>96.67</b></td>
      <td style="text-align: center"><b>11.66</b></td>
    </tr>
    <tr>
      <td style="text-align: center">0</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">99.57	</td>
      <td style="text-align: center">1.94	</td>
      <td style="text-align: center">96.17	</td>
      <td style="text-align: center">12.79	</td>
      <td style="text-align: center">95.33	</td>
      <td style="text-align: center">12.44</td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center">0.0001</td>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.95</td>
      <td style="text-align: center">0.95</td>
      <td style="text-align: center">0.95</td>
      <td style="text-align: center">98.50	</td>
      <td style="text-align: center">4.73	</td>
      <td style="text-align: center">95.17	</td>
      <td style="text-align: center">13.58	</td>
      <td style="text-align: center">94.67	</td>
      <td style="text-align: center">12.89</td>
    </tr>
    <tr>
      <td style="text-align: center">0.2</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">98.68	</td>
      <td style="text-align: center">4.39	</td>
      <td style="text-align: center">96.00	</td>
      <td style="text-align: center">13.07	</td>
      <td style="text-align: center">94.83	</td>
      <td style="text-align: center">12.18</td>
    </tr>
    <tr>
      <td style="text-align: center">0</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">99.50</td>
      <td style="text-align: center">2.67</td>
      <td style="text-align: center">96.50</td>
      <td style="text-align: center">11.60</td>
      <td style="text-align: center">95.00</td>
      <td style="text-align: center">11.58</td>
    </tr>
  </tbody>
</table>

<p align="justify">
All these table shows that the experimental results demonstrate that InceptionResNetV2 consistently outperformed NASNetLarge in classification accuracy. Three critical parameters emerged as significant determinants of model performance: dropout rate, learning rate and data loading configurations. While both architectures achieved high accuracy, the traditional InceptionResNetV2 architecture exhibited superior performance compared to the modern NASNetLarge architecture on the given dataset.
</p>

<div align="center">
  <h4>Summary of Model Performance for InceptionResNetV2 and NASNetLarge Across Two Data Loading Options: Row 1 Represents InceptionResNetV2 with Data Loading Option 2, While Row 2 Represents NASNetLarge with Data Loading Option 1.</h4>
</div>

<table>
  <thead>
    <tr>
      <th colspan="2" style="text-align: center">Parameters</th>
      <th colspan="3" style="text-align: center">Classification Report</th>
      <th colspan="2" style="text-align: center">Training Score (%)</th>
      <th colspan="2" style="text-align: center">Validation Score (%)</th>
      <th colspan="2" style="text-align: center">Test Score (%)</th>
    </tr>
    <tr>
      <th style="text-align: center">Learning Rate</th>
      <th style="text-align: center">Dropout</th>
      <th style="text-align: center">Precision</th>
      <th style="text-align: center">Recall</th>
      <th style="text-align: center">F1-Score</th>
      <th style="text-align: center">Accuracy</th>
      <th style="text-align: center">Loss</th>
      <th style="text-align: center">Accuracy</th>
      <th style="text-align: center">Loss</th>
      <th style="text-align: center">Accuracy</th>
      <th style="text-align: center">Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" style="text-align: center">0.0001</td>
      <td style="text-align: center"><b>0</b></td>
      <td style="text-align: center"><b>0.98</b></td>
      <td style="text-align: center"><b>0.98</b></td>
      <td style="text-align: center"><b>0.98</b></td>
      <td style="text-align: center"><b>99.86</b></td>
      <td style="text-align: center"><b>1.81</b></td>
      <td style="text-align: center"><b>97.50</b></td>
      <td style="text-align: center"><b>6.00</b></td>
      <td style="text-align: center"><b>98.50</b></td>
      <td style="text-align: center"><b>4.73</b></td>
    </tr>
    <tr>
      <td style="text-align: center">0.4</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">0.98</td>
      <td style="text-align: center">99.93</td>
      <td style="text-align: center">1.07</td>
      <td style="text-align: center">98.17</td>
      <td style="text-align: center">6.90</td>
      <td style="text-align: center">97.17</td>
      <td style="text-align: center">9.47</td>
    </tr>
  </tbody>
</table>

<div align="center">
  <h4>InceptionResNetV2 Accuracy vs Loss graph with 0.0001 Learning Rate and 0 Dropout</h4>
</div>

<p align="center">
  <img src="https://github.com/user-attachments/assets/f1396a18-d8c7-4f0a-a254-d69b02704b63" width="400" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/ec3cafe7-dd1a-4afb-b6f1-eb040b9348dd" width="400" style="margin: 0 10px;" />
</p>

<p align="justify">
Initial experiments with InceptionResNetV2 as it shown in a figure above promising accuracy but exhibited overfitting tendencies, particularly when implemented without dropout (dropout rate = 0). To address this, the model was fine-tuned by adjusting the dropout rate to 0.1 while maintaining the original learning rate. The optimization process yielded significant improvements in model stability, culminating in an impressive accuracy of 98.16%. This enhanced performance is comprehensively documented through multiple evaluation metrics, as illustrated in a figure below, which displays the relationship between accuracy and loss across training epochs. 
</p>

<h4 align="center">Final Optimized InceptionResNetV2 Model Accuracy vs Loss graph with a Learning Rate of 0.0001 and a Dropout Rate of 0.1</h4>
<p align="center">
  <img src="https://github.com/user-attachments/assets/1e071f66-ec69-4fd4-af2a-c823e7e046d6" width="400" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/83da6fc8-052c-43f0-8cba-b060c210c73b" width="400" style="margin: 0 10px;" />
</p>

<div align="center">
  <h4>Final Optimized InceptionResNetV2 Classification Report and Confusion Matrix</h4>
</div>

<div align="center">
<table>
  <thead>
    </tr>
    <tr style="border-top: 2px solid black; border-bottom: 1px solid black;">
      <th style="text-align: right; padding: 8px;"></th>
      <th style="text-align: center; padding: 8px;">precision</th>
      <th style="text-align: center; padding: 8px;">recall</th>
      <th style="text-align: center; padding: 8px;">f1-score</th>
      <th style="text-align: center; padding: 8px;">support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: right; padding: 8px;">Benign</td>
      <td style="text-align: center; padding: 8px;">0.99</td>
      <td style="text-align: center; padding: 8px;">0.95</td>
      <td style="text-align: center; padding: 8px;">0.97</td>
      <td style="text-align: center; padding: 8px;">149</td>
    </tr>
    <tr>
      <td style="text-align: right; padding: 8px;">Early</td>
      <td style="text-align: center; padding: 8px;">0.97</td>
      <td style="text-align: center; padding: 8px;">0.98</td>
      <td style="text-align: center; padding: 8px;">0.97</td>
      <td style="text-align: center; padding: 8px;">142</td>
    </tr>
    <tr>
      <td style="text-align: right; padding: 8px;">Pre</td>
      <td style="text-align: center; padding: 8px;">0.99</td>
      <td style="text-align: center; padding: 8px;">0.99</td>
      <td style="text-align: center; padding: 8px;">0.99</td>
      <td style="text-align: center; padding: 8px;">149</td>
    </tr>
    <tr>
      <td style="text-align: right; padding: 8px;">Pro</td>
      <td style="text-align: center; padding: 8px;">0.98</td>
      <td style="text-align: center; padding: 8px;">1.00</td>
      <td style="text-align: center; padding: 8px;">0.99</td>
      <td style="text-align: center; padding: 8px;">160</td>
    </tr>
    <tr>
      <td style="text-align: right; padding: 8px;">accuracy</td>
      <td style="text-align: center; padding: 8px;"></td>
      <td style="text-align: center; padding: 8px;"></td>
      <td style="text-align: center; padding: 8px;">0.98</td>
      <td style="text-align: center; padding: 8px;">600</td>
    </tr>
    <tr>
      <td style="text-align: right; padding: 8px;">macro avg</td>
      <td style="text-align: center; padding: 8px;">0.98</td>
      <td style="text-align: center; padding: 8px;">0.98</td>
      <td style="text-align: center; padding: 8px;">0.98</td>
      <td style="text-align: center; padding: 8px;">600</td>
    </tr>
    <tr style="border-bottom: 2px solid black;">
      <td style="text-align: right; padding: 8px;">weighted avg</td>
      <td style="text-align: center; padding: 8px;">0.98</td>
      <td style="text-align: center; padding: 8px;">0.98</td>
      <td style="text-align: center; padding: 8px;">0.98</td>
      <td style="text-align: center; padding: 8px;">600</td>
    </tr>
  </tbody>
</table>

<table>
  <tr>
    <td colspan="2" rowspan="2"></td>
    <td colspan="4" align="center"><b>Predicted Classification</b></td>
  </tr>
  <tr>
    <td align="center"><b>Benign</b></td>
    <td align="center"><b>Early</b></td>
    <td align="center"><b>Pre</b></td>
    <td align="center"><b>Pro</b></td>
  </tr>
  <tr>
    <td rowspan="4"><b>Actual<br>Classification</b></td>
    <td align="center"><b>Benign</b></td>
    <td align="center">142</td>
    <td align="center">5</td>
    <td align="center">0</td>
    <td align="center">2</td>
  </tr>
  <tr>
    <td align="center"><b>Early</b></td>
    <td align="center">2</td>
    <td align="center">139</td>
    <td align="center">1</td>
    <td align="center">0</td>
  </tr>
  <tr>
    <td align="center"><b>Pre</b></td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">148</td>
    <td align="center">1</td>
  </tr>
  <tr>
    <td align="center"><b>Pro</b></td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">160</td>
  </tr>
</table>
</div>

<h4 align="center">Final Optimized InceptionResNetV2 ROC Curves</h4>
<p align="center">
  <img src="https://github.com/user-attachments/assets/eebb1c6a-e5fe-4d8b-8b25-49a19b2cf1a3" width="400" style="margin: 0 10px;" />
</p>

<p align="justify">
The detailed performance analysis is further substantiated by the classification report presented in a table above and the confusion matrix, which provides a thorough breakdown of the model's classification performance across different classes. Additionally, the model's discrimination capabilities are visualized through ROC curves, offering a complete perspective of its predictive power. These collective metrics demonstrate the robustness and effectiveness of the optimized model in accurately classifying the target variables. 
</p>

#### Impact of Data Loading and Key Parameters
<p align="justify"> The experimental results highlighted optimized data loading strategy and two crucial parameters affecting model performance:
</p>

- <p align="justify">Experimental results demonstrate that Single Global Dataset Creation (Option 2) outperforms Class-Specific Data Preparation (Option 1) in deep learning model training. This superiority stems from three key mechanisms: enhanced randomization, computational optimization, and statistical integrity. SGDC's global shuffling mechanism ensures optimal dataset randomization, generating diverse batch compositions that yield robust gradient updates and improved model generalization. The unified processing approach eliminates redundant operations inherent in class-specific structures, while maintaining natural statistical distributions across training, validation, and test splits. This streamlined methodology not only enhances computational efficiency but also minimizes implementation artifacts that could impair training dynamics, ultimately providing a more robust foundation for deep learning model development.</p>
  
- <p align="justify">The implementation of dropout proved essential for model regularization, with models showing consistent overfitting tendencies when dropout was omitted (rate = 0). Dropout implementation contributed to model robustness through several key mechanisms. It prevented overfitting by reducing neuron codependency, while simultaneously enhancing generalization capability through forced independent learning. Furthermore, the implementation improved model resilience through distributed feature learning, ensuring more robust feature extraction across the network. </p>
  
- <p align="justify">The learning rate significantly influenced model convergence and stability. Higher learning rates led to training instability, while excessively low rates resulted in slower convergence and potential local minima traps. The relationship between learning rate and batch size emerged as an important consideration, with larger batch sizes generally accommodating higher learning rates due to gradient stability. </p>

#### Architectural Performance Analysis
<p align="justify"> The superior performance of InceptionResNetV2 over NASNetLarge can be attributed to several factors:
</p>

- <p align="justify"> The current dataset size (4,000 images, 1,000 per class) appears optimally suited for InceptionResNetV2's architecture. However, these results might differ with substantially larger datasets (>7,000 images).</p>
  
- <p align="justify">NASNetLarge's reinforcement learning-based architecture, while sophisticated, may be prone to overfitting on smaller datasets due to its extensive parameter count. Conversely, InceptionResNetV2's combination of Inception modules and residual connections provides efficient feature extraction while maintaining effective training dynamics.</p>
  
- <p align="justify">The high accuracy achieved with a significant proportion of non-trainable parameters shows effective transfer learning implementation. InceptionResNetV2's architecture appears to better leverage pre-trained features for the specific requirements of leukemia classification.</p>

<p align="justify"> The results demonstrate that well-tuned traditional architectures can outperform modern alternatives, particularly when working with moderate-sized datasets. </p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/3207775d-fb87-4a4a-a65a-3f4fae625f61" alt="Image 1" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/385a3015-1f57-46c0-b319-31725782b877" alt="Image 2" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/4bc904dc-9aec-4129-8f49-6ff054fc4d34" alt="Image 3" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/258476cc-7511-45d0-b044-5b715597780a" alt="Image 4" width="200" style="margin: 0 10px;" />
</p>

## Segmentation Task
<p align="justify">
Image segmentation methodologies can be broadly categorized into two fundamental approaches: traditional image processing techniques, exemplified by HSV thresholding, and deep learning-based methods, represented by architectures such as U-Net. These approaches differ significantly in their underlying principles and capabilities for image analysis.
</p>

<p align="justify">
HSV thresholding represents a classical image processing approach that operates by transforming images into the HSV (Hue, Saturation, Value) color space and applying specific threshold parameters to isolate desired color ranges. This methodology demonstrates particular efficacy in controlled environments where target objects exhibit consistent color characteristics, as the HSV color space effectively decouples color information (Hue) from illumination components (Saturation and Value). However, this approach exhibits inherent limitations in handling environmental variations such as illumination changes, color inconsistencies, or image noise. Furthermore, its reliance on color-based discrimination restricts its capability to incorporate other critical features such as morphological characteristics, textural properties, or spatial relationships.
</p>

<p align="justify">
In contrast, U-Net architecture represents an advanced convolutional neural network specifically optimized for image segmentation tasks. This deep learning approach facilitates comprehensive feature learning, encompassing color distributions, textural patterns, morphological characteristics, and spatial correlations, through supervised training with annotated image-mask pairs. The architecture's learning capability enables it to identify complex segmentation patterns based on pixel-wise feature analysis without requiring explicit threshold specifications. U-Net demonstrates superior adaptability to varying image conditions and environmental factors, such as illumination variations and object appearance modifications, enhancing its applicability across diverse real-world scenarios.
</p>

<p align="justify">
While HSV thresholding offers simplicity and effectiveness for color-based segmentation in controlled environments, U-Net's sophisticated feature learning capabilities and robust performance across variable conditions establish it as a more comprehensive solution for complex segmentation challenges. This enhanced adaptability and feature recognition capability position U-Net as the preferred methodology for addressing advanced segmentation requirements in dynamic and challenging imaging scenarios.
</p>

### Data Preprocessing
<p align="justify">
The data loading process implements a streamlined approach utilizing two comprehensive global lists for storing images and their corresponding segmentation masks. This unified storage methodology optimizes dataset preparation by eliminating the complexity of managing class-specific directories or individual class datasets. Given that the primary objective focuses on area segmentation rather than classification (which is addressed separately by the classification model), the masks are processed as binary representations. This binary segmentation approach enables focused identification of regions of interest, independent of class-specific characteristics.
</p>

<p align="justify">
The preprocessing pipeline incorporates image standardization through resizing operations and normalization procedures, while masks undergo binary conversion to ensure dataset uniformity. Following the preprocessing phase, the dataset undergoes a global shuffling procedure before being partitioned into training (70%), validation (15%), and test (15%) sets. This systematic approach ensures random distribution while maintaining efficient data management protocols. Figure below shows a random selected images from the original image of Acute Lymphoblastic Leukemia and its masking.
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/b5efd2ba-1fc7-4c9d-b9b5-a01543721125" />
</p>

### Results and Analysis
<p align="justify"> All these tables shows 36 experiments to optimize U-Net architecture for segmentation. The experiments were performed using a Kaggle notebook with P100 GPU, maintaining consistent parameters across all trials: batch size of 32, 60 epochs, and Adam optimizer. The primary variables investigated were learning rate, dropout rate, and data loading configurations.
</p>

<div align="center">
  <h4>Experimental Results of the Base U-Net Model</h4>
</div>

<div align="center">
<table>
 <thead>
   <tr style="border-top: 2px solid black; border-bottom: 1px solid black;">
     <th colspan="2" style="text-align: center; padding: 8px;">Parameters</th>
     <th colspan="6" style="text-align: center; padding: 8px;">Segmentation Report on Test Score (%)</th>
   </tr>
   <tr style="border-bottom: 1px solid black;">
     <th style="text-align: center; padding: 8px;">Learning Rate</th>
     <th style="text-align: center; padding: 8px;">Dropout</th>
     <th style="text-align: center; padding: 8px;">Accuracy</th>
     <th style="text-align: center; padding: 8px;">Loss</th>
     <th style="text-align: center; padding: 8px;">Precision</th>
     <th style="text-align: center; padding: 8px;">Recall</th>
     <th style="text-align: center; padding: 8px;">IoU Score</th>
     <th style="text-align: center; padding: 8px;">Dice Coefficient</th>
   </tr>
 </thead>
 <tbody>
   <tr>
     <td rowspan="3" style="text-align: center; padding: 8px;">0.1</td>
     <td style="text-align: center; padding: 8px;">0.4</td>
     <td style="text-align: center; padding: 8px;">92.49</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
   </tr>
   <tr>
     <td style="text-align: center; padding: 8px;">0.2</td>
     <td style="text-align: center; padding: 8px;">92.49</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
   </tr>
   <tr style="border-bottom: 1px solid black;">
     <td style="text-align: center; padding: 8px;">0</td>
     <td style="text-align: center; padding: 8px;">92.49</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
   </tr>
   <tr>
     <td rowspan="3" style="text-align: center; padding: 8px;">0.01</td>
     <td style="text-align: center; padding: 8px;">0.4</td>
     <td style="text-align: center; padding: 8px;">93.73</td>
     <td style="text-align: center; padding: 8px;">15.16</td>
     <td style="text-align: center; padding: 8px;">64.55</td>
     <td style="text-align: center; padding: 8px;">36.51</td>
     <td style="text-align: center; padding: 8px;">18.29</td>
     <td style="text-align: center; padding: 8px;">37.62</td>
   </tr>
   <tr>
     <td style="text-align: center; padding: 8px;">0.2</td>
     <td style="text-align: center; padding: 8px;">97.25</td>
     <td style="text-align: center; padding: 8px;">6.44</td>
     <td style="text-align: center; padding: 8px;">77.05</td>
     <td style="text-align: center; padding: 8px;">90.23</td>
     <td style="text-align: center; padding: 8px;">48.37</td>
     <td style="text-align: center; padding: 8px;">72.55</td>
   </tr>
   <tr style="border-bottom: 1px solid black;">
     <td style="text-align: center; padding: 8px;">0</td>
     <td style="text-align: center; padding: 8px;">98.70</td>
     <td style="text-align: center; padding: 8px;">3.29</td>
     <td style="text-align: center; padding: 8px;">91.26</td>
     <td style="text-align: center; padding: 8px;">91.46</td>
     <td style="text-align: center; padding: 8px;">63.80</td>
     <td style="text-align: center; padding: 8px;">87.55</td>
   </tr>
   <tr>
     <td rowspan="3" style="text-align: center; padding: 8px; color: green;"><b>0.001</b></td>
     <td style="text-align: center; padding: 8px;">0.4</td>
     <td style="text-align: center; padding: 8px;">99.13</td>
     <td style="text-align: center; padding: 8px;">2.09</td>
     <td style="text-align: center; padding: 8px;">94.06</td>
     <td style="text-align: center; padding: 8px;">94.32</td>
     <td style="text-align: center; padding: 8px;">71.37</td>
     <td style="text-align: center; padding: 8px;">91.55</td>
   </tr>
   <tr>
     <td style="text-align: center; padding: 8px;">0.2</td>
     <td style="text-align: center; padding: 8px;">99.14</td>
     <td style="text-align: center; padding: 8px;">2.04</td>
     <td style="text-align: center; padding: 8px;">94.07</td>
     <td style="text-align: center; padding: 8px;">94.56</td>
     <td style="text-align: center; padding: 8px;">71.32</td>
     <td style="text-align: center; padding: 8px;">91.59</td>
   </tr>
   <tr style="border-bottom: 1px solid black;">
     <td style="text-align: center; padding: 8px;"><b>0</b></td>
     <td style="text-align: center; padding: 8px; color: green;"><b>99.15</b></td>
     <td style="text-align: center; padding: 8px; color: green;"><b>2.04</b></td>
     <td style="text-align: center; padding: 8px; color: green;"><b>94.22</b></td>
     <td style="text-align: center; padding: 8px; color: green;"><b>94.42</b></td>
     <td style="text-align: center; padding: 8px; color: green;"><b>71.72</b></td>
     <td style="text-align: center; padding: 8px; color: green;"><b>91.74</b></td>
   </tr>
   <tr>
     <td rowspan="3" style="text-align: center; padding: 8px;">0.0001</td>
     <td style="text-align: center; padding: 8px;">0.4</td>
     <td style="text-align: center; padding: 8px;">98.97</td>
     <td style="text-align: center; padding: 8px;">2.56</td>
     <td style="text-align: center; padding: 8px;">92.88</td>
     <td style="text-align: center; padding: 8px;">93.41</td>
     <td style="text-align: center; padding: 8px;">67.58</td>
     <td style="text-align: center; padding: 8px;">90.03</td>
   </tr>
   <tr>
     <td style="text-align: center; padding: 8px;">0.2</td>
     <td style="text-align: center; padding: 8px;">98.98</td>
     <td style="text-align: center; padding: 8px;">2.54</td>
     <td style="text-align: center; padding: 8px;">92.76</td>
     <td style="text-align: center; padding: 8px;">93.68</td>
     <td style="text-align: center; padding: 8px;">67.69</td>
     <td style="text-align: center; padding: 8px;">90.15</td>
   </tr>
   <tr style="border-bottom: 2px solid black;">
     <td style="text-align: center; padding: 8px;">0</td>
     <td style="text-align: center; padding: 8px;">98.95</td>
     <td style="text-align: center; padding: 8px;">2.61</td>
     <td style="text-align: center; padding: 8px;">92.85</td>
     <td style="text-align: center; padding: 8px;">93.24</td>
     <td style="text-align: center; padding: 8px;">67.58</td>
     <td style="text-align: center; padding: 8px;">90.03</td>
   </tr>
 </tbody>
</table>
</div>

<h4 align="center">U-Net Accuracy vs Loss Graph, Dice Coefficient, and IoU Graph With 0.001 Learning Rate and 0 Dropout</h4>
<p align="center">
  <img src="https://github.com/user-attachments/assets/a86fa9c8-6048-45ac-9e29-16f0b5687442" alt="Image 1" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/eb80fd22-17bc-4aee-9604-d0cd97a37e75" alt="Image 2" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/17a48b95-8c25-4e6c-83a2-0f495eb4a673" alt="Image 3" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/64121e50-176e-469f-9149-60f103afb7b8" alt="Image 4" width="200" style="margin: 0 10px;" />
</p>

<h4 align="center">U-Net Model Result on Random Selected Image: Ground Truth vs Predicted Mask</h4>
<p align="center">
  <img src="https://github.com/user-attachments/assets/0ef8a6f4-a52e-486f-ae02-03dc0e53e346" alt="Image 3" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/28750e53-1bfc-434b-9feb-a56b55da01d7" alt="Image 4" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/520716d3-42b9-4611-8b0f-9330bfdaf949" alt="Image 3" width="200" style="margin: 0 10px;" />
</p>

<p align="justify">
The baseline U-Net model achieved promising initial results with high accuracy and dice coefficient scores, demonstrating good convergence characteristics in the Accuracy vs Loss graph. However, significant performance degradation was observed in specific scenarios, as evidenced by the substantial drop in dice score shown in the third image. This degradation primarily occurred in cases involving overlapping cells, where the model struggled to accurately segment lymphoblasts in the presence of erythrocytes.
</p>

<p align="justify">
The performance limitations of the standard U-Net architecture stemmed from its restricted capacity for multi-scale context representation. As the encoder down samples, global contextual features can be lost due to the focus on increasingly localized patterns, particularly affecting the segmentation of objects with varying sizes or those spread across the image. To address these limitations, the study implemented a Feature Pyramid Network (FPN) within the encoder segment of the U-Net architecture. The FPN integration was motivated to enhance the model's multi-scale feature representation capabilities. 
</p>

<p align="justify">
As described by Lin et al. (2017) in their seminal work on Feature Pyramid Networks for Object Detection, the FPN constructs a semantically rich pyramid of feature maps across multiple resolutions. This architecture utilizes lateral connections to merge high-level semantic features with low-level spatial details, while its top-down pathway progressively refines and integrates up sampled feature maps.
</p>

<div align="center">
  <h4>Experimental Results of the U-Net + FPN Model in the Encoder</h4>
</div>

<div align="center">
<table>
 <thead>
   <tr style="border-top: 2px solid black; border-bottom: 1px solid black;">
     <th colspan="2" style="text-align: center; padding: 8px;">Parameters</th>
     <th colspan="6" style="text-align: center; padding: 8px;">Segmentation Report on Test Score (%)</th>
   </tr>
   <tr style="border-bottom: 1px solid black;">
     <th style="text-align: center; padding: 8px;">Learning Rate</th>
     <th style="text-align: center; padding: 8px;">Dropout</th>
     <th style="text-align: center; padding: 8px;">Accuracy</th>
     <th style="text-align: center; padding: 8px;">Loss</th>
     <th style="text-align: center; padding: 8px;">Precision</th>
     <th style="text-align: center; padding: 8px;">Recall</th>
     <th style="text-align: center; padding: 8px;">IoU Score</th>
     <th style="text-align: center; padding: 8px;">Dice Coefficient</th>
   </tr>
 </thead>
 <tbody>
   <tr>
     <td rowspan="3" style="text-align: center; padding: 8px;">0.1</td>
     <td style="text-align: center; padding: 8px;">0.4</td>
     <td style="text-align: center; padding: 8px;">92.49</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
   </tr>
   <tr>
     <td style="text-align: center; padding: 8px;">0.2</td>
     <td style="text-align: center; padding: 8px;">92.49</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
   </tr>
   <tr style="border-bottom: 1px solid black;">
     <td style="text-align: center; padding: 8px;">0</td>
     <td style="text-align: center; padding: 8px;">92.49</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
   </tr>
   <tr>
     <td rowspan="3" style="text-align: center; padding: 8px;">0.01</td>
     <td style="text-align: center; padding: 8px;">0.4</td>
     <td style="text-align: center; padding: 8px;">98.72	</td>
     <td style="text-align: center; padding: 8px;">3.24	</td>
     <td style="text-align: center; padding: 8px;">91.47	</td>
     <td style="text-align: center; padding: 8px;">91.44	</td>
     <td style="text-align: center; padding: 8px;">63.86	</td>
     <td style="text-align: center; padding: 8px;">87.35</td>
   </tr>
   <tr>
     <td style="text-align: center; padding: 8px;">0.2</td>
     <td style="text-align: center; padding: 8px;">97.26	</td>
     <td style="text-align: center; padding: 8px;">6.79	</td>
     <td style="text-align: center; padding: 8px;">81.08	</td>
     <td style="text-align: center; padding: 8px;">82.82	</td>
     <td style="text-align: center; padding: 8px;">48.78	</td>
     <td style="text-align: center; padding: 8px;">73.17</td>
   </tr>
   <tr style="border-bottom: 1px solid black;">
     <td style="text-align: center; padding: 8px;">0</td>
     <td style="text-align: center; padding: 8px;">97.17	</td>
     <td style="text-align: center; padding: 8px;">7.19	</td>
     <td style="text-align: center; padding: 8px;">81.84	</td>
     <td style="text-align: center; padding: 8px;">80.16	</td>
     <td style="text-align: center; padding: 8px;">44.31	</td>
     <td style="text-align: center; padding: 8px;">70.12</td>
   </tr>
   <tr>
     <td rowspan="3" style="text-align: center; padding: 8px; color: green;"><b>0.001</b></td>
     <td style="text-align: center; padding: 8px;"><b>0.4</b></td>
     <td style="text-align: center; padding: 8px;"><b>99.16</b></td>
     <td style="text-align: center; padding: 8px;"><b>1.99</b></td>
     <td style="text-align: center; padding: 8px;"><b>94.82</b></td>
     <td style="text-align: center; padding: 8px;"><b>93.95</b></td>
     <td style="text-align: center; padding: 8px;"><b>71.96</b></td>
     <td style="text-align: center; padding: 8px;"><b>91.81</b></td>
   </tr>
   <tr>
     <td style="text-align: center; padding: 8px;">0.2</td>
     <td style="text-align: center; padding: 8px;">99.13	</td>
     <td style="text-align: center; padding: 8px;">2.08	</td>
     <td style="text-align: center; padding: 8px;">93.84	</td>
     <td style="text-align: center; padding: 8px;">94.67	</td>
     <td style="text-align: center; padding: 8px;">71.45	</td>
     <td style="text-align: center; padding: 8px;">91.69</td>
   </tr>
   <tr style="border-bottom: 1px solid black;">
     <td style="text-align: center; padding: 8px;">0</td>
     <td style="text-align: center; padding: 8px; color: green;">99.11	</td>
     <td style="text-align: center; padding: 8px; color: green;">2.14	</td>
     <td style="text-align: center; padding: 8px; color: green;">94.10	</td>
     <td style="text-align: center; padding: 8px; color: green;">94.09	</td>
     <td style="text-align: center; padding: 8px; color: green;">70.40	</td>
     <td style="text-align: center; padding: 8px; color: green;">91.23</td>
   </tr>
   <tr>
     <td rowspan="3" style="text-align: center; padding: 8px;">0.0001</td>
     <td style="text-align: center; padding: 8px;">0.4</td>
     <td style="text-align: center; padding: 8px;">98.98	</td>
     <td style="text-align: center; padding: 8px;">2.50	</td>
     <td style="text-align: center; padding: 8px;">93.08	</td>
     <td style="text-align: center; padding: 8px;">93.39	</td>
     <td style="text-align: center; padding: 8px;">67.95	</td>
     <td style="text-align: center; padding: 8px;">90.06</td>
   </tr>
   <tr>
     <td style="text-align: center; padding: 8px;">0.2</td>
     <td style="text-align: center; padding: 8px;">98.97	</td>
     <td style="text-align: center; padding: 8px;">2.56	</td>
     <td style="text-align: center; padding: 8px;">93.39	</td>
     <td style="text-align: center; padding: 8px;">92.79	</td>
     <td style="text-align: center; padding: 8px;">66.78	</td>
     <td style="text-align: center; padding: 8px;">89.74</td>
   </tr>
   <tr style="border-bottom: 2px solid black;">
     <td style="text-align: center; padding: 8px;">0</td>
     <td style="text-align: center; padding: 8px;">99.01	</td>
     <td style="text-align: center; padding: 8px;">2.44	</td>
     <td style="text-align: center; padding: 8px;">92.93	</td>
     <td style="text-align: center; padding: 8px;">93.94	</td>
     <td style="text-align: center; padding: 8px;">67.95	</td>
     <td style="text-align: center; padding: 8px;">90.05</td>
   </tr>
 </tbody>
</table>
</div>

<h4 align="center">U-Net + FPN Accuracy vs Loss Graph, Dice Coefficient, and IoU Graph With 0.001 Learning Rate and 0.4 Dropout</h4>
<p align="center">
  <img src="https://github.com/user-attachments/assets/af73a089-dc55-4f89-8340-cf26d14578d0" alt="Image 1" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/42a31f71-deb1-4624-afd5-99c66f7171b1" alt="Image 2" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/f6705878-295d-4b03-81a9-eb97e09fc848" alt="Image 3" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/bb5f4e3c-8e91-4d6d-921a-e13a3426a16a" alt="Image 4" width="200" style="margin: 0 10px;" />
</p>

<h4 align="center">U-Net + FPN Model Result on Random Selected Image: Ground Truth vs Predicted Mask</h4>
<p align="center">
  <img src="https://github.com/user-attachments/assets/7d7cc313-8a9d-4c27-880e-ccb2b2823021" alt="Image 1" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/06e82708-024c-49d8-b055-e3d9fbd014d2" alt="Image 2" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/962e075a-4143-4af9-9e8f-2a7c2dffcb38" alt="Image 3" width="200" style="margin: 0 10px;" />
</p>

<p align="justify">
While the FPN integration yielded a modest improvement in dice coefficient and accuracy, further enhancement was achieved through the implementation of GELU activation functions within the FPN blocks. The transition from ReLU to GELU activation was motivated by several theoretical advantages in gradient handling and feature integration. GELU provides probabilistic smooth transitions based on input magnitude, preserving nuanced gradients crucial for multi-scale feature map generation, and enabling more refined learning of feature dependencies.
</p>

<p align="justify">
The feature integration benefits of GELU activation manifested in balanced feature processing across resolution scales, improved fusion of semantic and spatial features, and reduced potential feature redundancy through smoother activation patterns. The scale-specific optimization strategy retained ReLU in basic U-Net blocks for efficient down/up sampling while implementing GELU in FPN blocks for enhanced multi-scale feature handling. This hybrid approach resulted in improved handling of varying feature magnitudes across scales, contributing to the overall enhancement of the segmentation performance.
</p>

<p align="justify">
The combination of FPN architecture and GELU activation resulted in enhanced segmentation performance, particularly in challenging cases involving overlapping cells. This improvement can be attributed to the architecture's improved capability to maintain both semantic and spatial information across multiple scales while ensuring smooth feature integration through appropriate activation functions.
</p>

<div align="center">
  <h4>Experimental Results of the U-Net + FPN model with GELU Activation in the Encoder</h4>
</div>

<div align="center">
<table>
 <thead>
   <tr style="border-top: 2px solid black; border-bottom: 1px solid black;">
     <th colspan="2" style="text-align: center; padding: 8px;">Parameters</th>
     <th colspan="6" style="text-align: center; padding: 8px;">Segmentation Report on Test Score</th>
   </tr>
   <tr style="border-bottom: 2px solid black;">
     <th style="text-align: center; padding: 8px;">Learning Rate</th>
     <th style="text-align: center; padding: 8px;">Dropout</th>
     <th style="text-align: center; padding: 8px;">Accuracy</th>
     <th style="text-align: center; padding: 8px;">Loss</th>
     <th style="text-align: center; padding: 8px;">Precision</th>
     <th style="text-align: center; padding: 8px;">Recall</th>
     <th style="text-align: center; padding: 8px;">IoU Score</th>
     <th style="text-align: center; padding: 8px;">Dice Coefficient</th>
   </tr>
 </thead>
 <tbody>
   <tr>
     <td rowspan="3" style="text-align: center; padding: 8px;">0.1</td>
     <td style="text-align: center; padding: 8px;">0.4</td>
     <td style="text-align: center; padding: 8px;">92.07</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
   </tr>
   <tr>
     <td style="text-align: center; padding: 8px;">0.2</td>
     <td style="text-align: center; padding: 8px;">92.07</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
   </tr>
   <tr style="border-bottom: 1px solid black;">
     <td style="text-align: center; padding: 8px;">0</td>
     <td style="text-align: center; padding: 8px;">92.07</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;">0.00</td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
     <td style="text-align: center; padding: 8px;"><span style="color: red;">NaN</span></td>
   </tr>
   <tr>
     <td rowspan="3" style="text-align: center; padding: 8px;">0.01</td>
     <td style="text-align: center; padding: 8px;">0.4</td>
     <td style="text-align: center; padding: 8px;">98.72</td>
     <td style="text-align: center; padding: 8px;">3.13</td>
     <td style="text-align: center; padding: 8px;">91.68</td>
     <td style="text-align: center; padding: 8px;">91.23</td>
     <td style="text-align: center; padding: 8px;">63.85</td>
     <td style="text-align: center; padding: 8px;">87.56</td>
   </tr>
   <tr>
     <td style="text-align: center; padding: 8px;">0.2</td>
     <td style="text-align: center; padding: 8px;">98.78</td>
     <td style="text-align: center; padding: 8px;">3.02</td>
     <td style="text-align: center; padding: 8px;">91.74</td>
     <td style="text-align: center; padding: 8px;">92.03</td>
     <td style="text-align: center; padding: 8px;">65.01</td>
     <td style="text-align: center; padding: 8px;">88.04</td>
   </tr>
   <tr style="border-bottom: 1px solid black;">
     <td style="text-align: center; padding: 8px;">0</td>
     <td style="text-align: center; padding: 8px;">97.37</td>
     <td style="text-align: center; padding: 8px;">6.11</td>
     <td style="text-align: center; padding: 8px;">79.42</td>
     <td style="text-align: center; padding: 8px;">87.73</td>
     <td style="text-align: center; padding: 8px;">51.18</td>
     <td style="text-align: center; padding: 8px;">75.29</td>
   </tr>
   <tr>
     <td rowspan="3" style="text-align: center; padding: 8px; color: green;"><b>0.001</b></td>
     <td style="text-align: center; padding: 8px;">0.4</td>
     <td style="text-align: center; padding: 8px;">99.10</td>
     <td style="text-align: center; padding: 8px;">2.18</td>
     <td style="text-align: center; padding: 8px;">93.54</td>
     <td style="text-align: center; padding: 8px;">94.52</td>
     <td style="text-align: center; padding: 8px;">70.78</td>
     <td style="text-align: center; padding: 8px;">91.37</td>
   </tr>
   <tr>
     <td style="text-align: center; padding: 8px;">0.2</td>
     <td style="text-align: center; padding: 8px;">99.13</td>
     <td style="text-align: center; padding: 8px;">2.08</td>
     <td style="text-align: center; padding: 8px;">94.15</td>
     <td style="text-align: center; padding: 8px;">94.32</td>
     <td style="text-align: center; padding: 8px;">71.09</td>
     <td style="text-align: center; padding: 8px;">91.55</td>
   </tr>
   <tr style="border-bottom: 1px solid black;">
     <td style="text-align: center; padding: 8px;"><b>0</b></td>
     <td style="text-align: center; padding: 8px; color: green;"><b>99.17</b></td>
     <td style="text-align: center; padding: 8px; color: green;"><b>1.99</b></td>
     <td style="text-align: center; padding: 8px; color: green;"><b>94.46</b></td>
     <td style="text-align: center; padding: 8px; color: green;"><b>94.43</b></td>
     <td style="text-align: center; padding: 8px; color: green;"><b>72.31</b></td>
     <td style="text-align: center; padding: 8px; color: green;"><b>92.05</b></td>
   </tr>
   <tr>
     <td rowspan="3" style="text-align: center; padding: 8px;">0.0001</td>
     <td style="text-align: center; padding: 8px;">0.4</td>
     <td style="text-align: center; padding: 8px;">98.95</td>
     <td style="text-align: center; padding: 8px;">2.61</td>
     <td style="text-align: center; padding: 8px;">92.71</td>
     <td style="text-align: center; padding: 8px;">93.35</td>
     <td style="text-align: center; padding: 8px;">67.43</td>
     <td style="text-align: center; padding: 8px;">89.80</td>
   </tr>
   <tr>
     <td style="text-align: center; padding: 8px;">0.2</td>
     <td style="text-align: center; padding: 8px;">98.96</td>
     <td style="text-align: center; padding: 8px;">2.57</td>
     <td style="text-align: center; padding: 8px;">92.71</td>
     <td style="text-align: center; padding: 8px;">93.54</td>
     <td style="text-align: center; padding: 8px;">67.90</td>
     <td style="text-align: center; padding: 8px;">89.98</td>
   </tr>
   <tr style="border-bottom: 2px solid black;">
     <td style="text-align: center; padding: 8px;">0</td>
     <td style="text-align: center; padding: 8px;">98.95</td>
     <td style="text-align: center; padding: 8px;">2.61</td>
     <td style="text-align: center; padding: 8px;">92.92</td>
     <td style="text-align: center; padding: 8px;">93.16</td>
     <td style="text-align: center; padding: 8px;">67.88</td>
     <td style="text-align: center; padding: 8px;">89.95</td>
   </tr>
 </tbody>
</table>
</div>

<h4 align="center">U-Net + FPN with GELU Activation Accuracy vs Loss Graph, Dice Coefficient, and IoU Graph With 0.001 Learning Rate and 0 Dropout</h4>
<p align="center">
  <img src="https://github.com/user-attachments/assets/2af287fb-93f1-48e5-9c38-ce1909fc1968" alt="Image 1" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/0c377f0e-3983-4d4e-b9d7-544753411fb2" alt="Image 2" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/c37bd904-e927-49fe-bde3-34c12f069581" alt="Image 3" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/6713a0fd-d09a-4310-ac92-da4f0f4405b5" alt="Image 4" width="200" style="margin: 0 10px;" />
</p>

<h4 align="center">U-Net + FPN with GELU Activation Model Result on Random Selected Image: Ground Truth vs Predicted Mask</h4>
<p align="center">
  <img src="https://github.com/user-attachments/assets/4d204e89-a514-4175-8c90-6020b11546b8" alt="Image 1" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/f90bc7c1-f508-492a-bcd0-e59ab8dbe142" alt="Image 2" width="200" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/7d79ee35-8b27-4a0e-96e6-387ecb77e81d" alt="Image 3" width="200" style="margin: 0 10px;" />
</p>
