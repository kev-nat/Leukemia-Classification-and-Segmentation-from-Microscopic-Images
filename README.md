# Leukemia Classification and Segmentation from Microscopic Images

## Situation
<p align="justify"> 
Childhood cancer represents a significant global public health challenge, with increasing incidence rates annually [1]. Among pediatric malignancies, Acute Lymphoblastic Leukemia (ALL) emerges as the predominant form, accounting for 30% of all childhood cancers in the United States, with an annual incidence of 3.5 per 100,000 children [2]. In Indonesia, ALL maintains similar prevalence patterns, with an incidence rate of 2.5-4.0 per 100,000 children, translating to approximately 2,000-3,200 new cases annually [3].</p>

<p align="justify">
The disparity in survival outcomes between high-income countries (HICs) and low- and middle-income countries (LMICs) is particularly striking. While HICs like the United States and European nations report 5-year survival rates approaching 90% [4], Southeast Asian countries demonstrate significantly lower rates, with Malaysia at 69.4% and Thailand at 55.1% [4]. Indonesia faces particularly challenging outcomes, with reported 5-year survival rates of 28.9% and 31.8% at major medical centers [5,6]. These disparities are attributed to multiple factors, including high relapse rates, treatment abandonment, delayed diagnosis, and limited healthcare accessibility [7].
</p>

<p align="justify">
Current diagnostic approaches in Indonesia primarily rely on clinical and hematological parameters, including age, leukocyte count, and conventional morphological examination of bone marrow. While advanced molecular techniques like PCR-based detection of BCR-ABL1 fusion genes have enhanced diagnostic capabilities [8], the implementation of comprehensive genetic testing remains limited due to resource constraints. Although high-resolution genomic technologies have revolutionized ALL diagnosis in HICs, their high cost and infrastructure requirements make them impractical for routine use in LMICs [9].
</p>

<p align="justify">  
The integration of deep learning techniques with Peripheral Blood Smear (PBS) image analysis presents a promising alternative diagnostic approach. This methodology offers several advantages: it is non-invasive, cost-effective, and capable of detecting subtle morphological abnormalities indicative of leukemia. The application of Convolutional Neural Networks (CNNs) to PBS image analysis provides consistent, reproducible results while potentially reducing diagnostic errors and improving early detection capabilities. This approach is particularly relevant for resource-limited settings, where it could enhance diagnostic accuracy and facilitate timely intervention, ultimately contributing to improved survival outcomes for children with ALL.
</p>

## Dataset
<p align="justify"> 

The dataset used in this study was obtained from the [Kaggle repository](https://www.kaggle.com/datasets/mehradaria/leukemia), specifically focusing on the classification and segmentation of Acute Lymphoblastic Leukemia (ALL) and its subtypes. The dataset comprises 3,256 microscopic images derived from 89 patients under investigation for ALL at the bone marrow laboratory of Taleqani Hospital in Tehran, Iran. Among these patients, 25 were diagnosed as healthy with benign conditions (hematogones), while 64 were confirmed to have various ALL subtypes.  
</p>

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
From all these tables it shows 48 experiments comparing InceptionResNetV2 and NASNetLarge architectures for classification. The experiments were performed using a Kaggle notebook with P100 GPU, maintaining consistent parameters across all trials: batch size of 32, 60 epochs, and Adam optimizer. The primary variables investigated were learning rate, dropout rate, and data loading configurations.
</p>

<h4 style="text-align: center"> Experimental Results of the InceptionResNetV2 Model using Data Loading Option 1</h4>
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

<h4 style="text-align: center">Experimental Results of the InceptionResNetV2 Model using Data Loading Option 1</h4>
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

<h4 style="text-align: center"> Experimental Results of the InceptionResNetV2 Model using Data Loading Option 2</h4>
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

<h4 style="text-align: center"> Experimental Results of the NASNETLarge Model using Data Loading Option 2</h4>
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
      <td rowspan="3" style="text-align: center">0.001</td>
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
      <td style="text-align: center">0.2</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">0.97</td>
      <td style="text-align: center">99.75	</td>
      <td style="text-align: center">1.67	</td>
      <td style="text-align: center">96.83	</td>
      <td style="text-align: center">11.92	</td>
      <td style="text-align: center">96.67	</td>
      <td style="text-align: center">11.66</td>
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
      <td rowspan="3" style="text-align: center"><b>0.0001</b></td>
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
      <td style="text-align: center"><b>0</b></td>
      <td style="text-align: center"><b>0.97	</b></td>
      <td style="text-align: center"><b>0.96</b></td>
      <td style="text-align: center"><b>0.96</b></td>
      <td style="text-align: center"><b>99.50	</b></td>
      <td style="text-align: center"><b>2.67	</b></td>
      <td style="text-align: center"><b>96.50	</b></td>
      <td style="text-align: center"><b>11.60	</b></td>
      <td style="text-align: center"><b>95.00	</b></td>
      <td style="text-align: center"><b>11.58</b></td>
    </tr>
  </tbody>
</table>

<p align="justify">
All these table shows that the experimental results demonstrate that InceptionResNetV2 consistently outperformed NASNetLarge in classification accuracy. Three critical parameters emerged as significant determinants of model performance: dropout rate, learning rate and data loading configurations. While both architectures achieved high accuracy, the traditional InceptionResNetV2 architecture exhibited superior performance compared to the modern NASNetLarge architecture on the given dataset.
</p>

<h4 style="text-align: center"> Summary of Model Performance for InceptionResNetV2 and NASNetLarge Across Two Data Loading Options: Row 1 Represents InceptionResNetV2 with Data Loading Option 2, While Row 2 Represents NASNetLarge with Data Loading Option 1.</h4>
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

<h4 align="center">InceptionResNetV2 Accuracy vs Loss graph with 0.0001 Learning Rate and 0 Dropout.</h4>
<p align="center">
  <img src="https://github.com/user-attachments/assets/f1396a18-d8c7-4f0a-a254-d69b02704b63" width="250" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/ec3cafe7-dd1a-4afb-b6f1-eb040b9348dd" width="250" style="margin: 0 10px;" />
</p>

<p align="justify">
Initial experiments with InceptionResNetV2 as it shown in a figure above promising accuracy but exhibited overfitting tendencies, particularly when implemented without dropout (dropout rate = 0). To address this, the model was fine-tuned by adjusting the dropout rate to 0.1 while maintaining the original learning rate. The optimization process yielded significant improvements in model stability, culminating in an impressive accuracy of 98.16%. This enhanced performance is comprehensively documented through multiple evaluation metrics, as illustrated in a figure below, which displays the relationship between accuracy and loss across training epochs. 
</p>

<h4 align="center">Final Optimized InceptionResNetV2 Model Accuracy vs Loss graph with a Learning Rate of 0.0001 and a Dropout Rate of 0.1.</h4>
<p align="center">
  <img src="https://github.com/user-attachments/assets/1e071f66-ec69-4fd4-af2a-c823e7e046d6" width="250" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/83da6fc8-052c-43f0-8cba-b060c210c73b" width="250" style="margin: 0 10px;" />
</p>

<h4 style="text-align: center"> Final Optimized InceptionResNetV2 Classification Report</h4>

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
</div>

<h4 align="center">Final Optimized InceptionResNetV2 Confusion Matrix & ROC Curves</h4>
<p align="center">
  <img src="https://github.com/user-attachments/assets/ce872d0e-0180-4830-951f-cabba7a00831" width="400" style="margin: 0 10px;" />
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
  <img src="https://github.com/user-attachments/assets/3207775d-fb87-4a4a-a65a-3f4fae625f61" alt="Image 1" width="225" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/385a3015-1f57-46c0-b319-31725782b877" alt="Image 2" width="225" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/4bc904dc-9aec-4129-8f49-6ff054fc4d34" alt="Image 3" width="225" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/258476cc-7511-45d0-b044-5b715597780a" alt="Image 4" width="225" style="margin: 0 10px;" />
</p>

#### Evolution of CNN Architectures: Traditional vs. Modern Approaches

<p align="justify">
The experimental results observed in this study reflect broader trends in the evolution of Convolutional Neural Networks (CNNs). Since AlexNet's breakthrough performance in the 2012 ImageNet competition, CNN architectures have undergone significant advancement, demonstrating capabilities that frequently surpass human performance in image processing tasks.
</p>

<p align="justify">
The fundamental concept of CNNs, though implemented over three decades ago, gained prominence primarily due to two factors: the democratization of computational resources and the availability of large-scale datasets. This evolution has particularly addressed the inherent challenges of image processing, where traditional neural networks faced significant limitations due to high-dimensional input data. For instance, a 1000 × 1000-pixel image presents one million features, requiring approximately 10¹² parameters in a single-layer feed-forward neural network, a scale that poses substantial challenges in terms of both computational resources and potential overfitting.

<p align="justify">
CNNs effectively address these challenges by leveraging two key characteristics of image data. The first is feature localization, which describes the correlation between adjacent pixels in representing semantic features. The second is feature independence of location, which refers to the invariance of feature significance regardless of spatial position.
</p>

<h4 align="center">CNN Architecture Evolution </h4>
<p align="center">
  <img src="https://github.com/user-attachments/assets/4130cd06-716e-46a3-9f6d-6e0d40e84ff1" />
</p>

<p align="justify">
These properties are implemented through shared parameters and locally connected networks as it shown above which dramatically reducing the parameter space from 10¹² to 10³ in typical applications. The CNN architecture achieves this efficiency through three primary components: convolution layers, pooling layers, and fully connected layers. The convolution layers implement filters with defined stride patterns for feature extraction, while pooling layers provide feature resilience and parameter reduction through max or average operations. Finally, fully connected layers connect flattened convolutional features to output classifications, completing the network's processing pipeline.
</p>

<p align="justify">
This progression contextualizes our experimental findings, where InceptionResNetV2, representing the architecture engineering phase, demonstrated superior performance compared to NASNetLarge, an AutoML-based approach. While NASNetLarge exemplifies the potential of reinforcement learning in architecture design, achieving state-of-the-art results on ImageNet, its implementation requires substantial computational resources that limit its accessibility. 
</p>

<p align="justify">
The superior performance of InceptionResNetV2 in this study of leukemia classification task, as detailed in a figure below, suggests that architecture engineering approaches remain highly relevant for specific applications, particularly with moderate-sized datasets. However, the future trajectory of CNN development appears oriented toward AutoML methodologies, despite their current limitations in computational requirements.
</p>

<h4 align="center">CNN Architecture Evolution </h4>
<p align="center">
  <img src="https://github.com/user-attachments/assets/7f5a976e-9158-422c-9d98-56fcdcfeb020" />
</p>

<p align="justify">
This evolution reflects the ongoing challenge in deep learning: balancing model complexity, computational efficiency, and generalization capability. While AutoML approaches like NASNetLarge represent the cutting edge of architectural innovation, our results demonstrate that well-engineered traditional architectures can still provide optimal solutions for specific medical imaging applications. This observation is particularly relevant for healthcare implementations where computational resources may be constrained, and dataset sizes are moderate.
</p>

## Segmentation Task
### Data Preprocessing
### Results and Analysis

## Result
