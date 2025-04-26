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

<h4 style="text-align: center"> Experimental Results of the NASNETLarge Model using Data Loading Option 1</h4>
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
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">0.96</td>
      <td style="text-align: center">97.57</td>
      <td style="text-align: center">6.59</td>
      <td style="text-align: center">95.67</td>
      <td style="text-align: center">11.25</td>
      <td style="text-align: center">95.83</td>
      <td style="text-align: center">95.83</td>
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
      <td style="text-align: center">99.82</td>
      <td style="text-align: center">1.14</td>
      <td style="text-align: center">97.17</td>
      <td style="text-align: center">6.86</td>
      <td style="text-align: center">96.67</td>
      <td style="text-align: center">7.36</td>
    </tr>
    <tr>
      <td style="text-align: center">0</td>
      <td style="text-align: center">0.98	</td>
      <td style="text-align: center">0.99</td>
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

## Segmentation Task
### Data Preprocessing
### Results and Analysis

## Result
