#set document(
  title: "CYBERML Project Report",
  author: "Group X",
)

#set page(
  paper: "a4",
  margin: auto,
)

#set text(
  font: "Times New Roman",
  size: 13pt,
  lang: "en"
)

// --- Title Page ---
#align(center + horizon)[
  #text(size: 24pt, weight: "bold")[CYBERML Project Report]
  
  #v(2em)
  
  #text(size: 16pt)[Design, Deployment, and Evaluation of a Cybersecurity Data Analysis Chain]
  #v(4em)
  #text(size: 14pt)[Quentin Prunet, Oscar Le Dauphin, Hugo Schreiber, Aleksei Kotliarov] \
  #v(1em)
  #text(size: 14pt)[Group 14]
]

#pagebreak()

// --- Configuration for Content Pages ---
// Header setup (skipped on title page)
#set page(
  header: align(right)[
    #text(size: 9pt, fill: luma(100))[Quentin Prunet, Oscar Le Dauphin, Hugo Schreiber, Aleksei Kotliarov -- Group 14]
    #v(-0.5em)
    #line(length: 100%, stroke: 0.5pt + luma(150))
  ]
)

// Spacing configuration
#show heading: set block(above: 2.5em, below: 1.5em, sticky: true)
#set heading(numbering: "1.1")

// Link styling
#show link: set text(fill: blue)
#show link: underline

// Table global styling
#show table: set align(center)
#show table: set block(breakable: false)

// --- Table of Contents ---
#outline(
  title: "Table of Contents",
  indent: auto,
)

#pagebreak()

// --- Content ---

= Introduction

The rapid proliferation of Internet of Things (IoT) devices has exponentially increased the attack surface for cyber threats. Securing these networks requires robust, automated systems capable of detecting malicious activity in real-time. This project focuses on the design, deployment, and evaluation of a data handling chain dedicated to the analysis of cybersecurity data using batch processing. The complete implementation and experimental results are available in the accompanying Google Colab notebook: #link("https://colab.research.google.com/drive/18zDQF8IGrm2KfeUh-yLokUWrIQYx5TAf?usp=sharing")[Analysis Notebook].

The primary scope of this work involves the analysis of the *CIC IoT-DIAD 2024* dataset, a modern collection of network traffic data representing various IoT attack scenarios. By leveraging this dataset, we aim to build a pipeline that can not only classify specific types of attacks but also detect anomalies without prior knowledge of attack signatures.

The core objectives of this project are focused on several critical milestones. We first aim to design and implement a complete data handling chain, including ingestion, preprocessing, and feature engineering, capable of processing data in batches. Following this deployment, we perform a detailed characterization of the dataset under study, analyzing feature distributions and class balances to understand the underlying data structure. We then evaluate and compare the performance of three unsupervised algorithms for anomaly detection and three supervised algorithms for classification using standard metrics such as Precision, Recall, and MCC. Finally, as a bonus objective, we evaluate the robustness of the trained models against adversarial evasion attacks, specifically using the Fast Gradient Sign Method (FGSM).

= Fundamentals of Anomaly Detection

Anomaly detection refers to the identification of patterns in data that do not conform to expected behavior. In the context of cybersecurity, these "outliers" often correspond to malicious activities such as intrusions, fraud, or system faults. However, effective anomaly detection faces several significant challenges in network environments. The high dimensionality of network traffic data, which often consists of dozens of features, makes distance-based calculations computationally expensive and less effective, a phenomenon known as the "curse of dimensionality." Furthermore, class imbalance is a common hurdle; while benign traffic usually vastly outnumbers attack traffic in real-world scenarios, specific datasets like CIC IoT-DIAD 2024 present the inverse challenge where attacks dominate the volume. Additionally, the definition of "normal" behavior changes over time due to concept drift, necessitating models that can adapt to evolving network conditions.

We employ two distinct approaches in this study to address these challenges. Unsupervised learning methods are utilized under the assumption that anomalies are rare and statistically different from the majority. These models operate without labeled training data and are essential for detecting zero-day attacks. Complementing this, supervised learning methods are employed to learn specific signatures from labeled "attack" and "benign" examples, typically offering higher accuracy for known threats but requiring high-quality labeled datasets.

= Dataset Characterization & Exploration

== Dataset Description
For this project, we selected the *CIC IoT-DIAD 2024* dataset. This dataset is designed to provide a comprehensive view of modern IoT network traffic, including both benign activities and various cyberattacks. The data is structured as network flows derived from PCAP files, with each entry containing features such as Flow Duration, Total Fwd Packets, and Protocol, alongside a specific label indicating the type of traffic.

== Exploratory Data Analysis (EDA)

=== Class Distribution
A critical finding during our exploratory analysis was the extreme class imbalance present in the dataset. Contrary to typical anomaly detection scenarios where attacks are rare events (often < 1%), the CIC IoT-DIAD 2024 dataset is heavily dominated by attack traffic.

In our sampled subset, the distribution was approximately:
- *Attack:* ~98%
- *Benign:* ~2%

#figure(
  image("extracted_images/image_12_0.png", width: 80%),
  caption: [Distribution of Classes (0 = Benign, 1 = Attack)]
)

This inversion significantly impacts the performance of unsupervised algorithms, which typically rely on the assumption that the "normal" class is the majority. We will discuss the implications of this in the Benchmarking section.

=== Feature Correlation
To understand the redundancy and relationship between features, we computed a correlation matrix for the numeric attributes.

#figure(
  image("extracted_images/image_13_1.png", width: 80%),
  caption: [Feature Correlation Matrix]
)

The analysis revealed high collinearity among several flow-based statistics (e.g., packet counts and byte counts), suggesting that dimensionality reduction techniques like PCA could be effective for feature extraction, although potentially limited by the nonlinear nature of the attack signatures.

== Preprocessing Strategy
Given the massive size of the dataset, which reaches 1.89 GB in its compressed form and expands significantly upon extraction, we adopted a resource-efficient preprocessing strategy. Due to significant hardware and time constraints, we chose to load only a random 1% subset of each CSV file. This decision was largely driven by the limitations of the Google Colab environment, where loading and processing the full multi-gigabyte dataset would be near impossible or, at the very least, could not be completed in a timely manner within the session limits. By sampling 1% of each file, we maintained a manageable memory footprint that stayed within the available RAM while still preserving the overall statistical properties and class distributions of the data. During the cleaning phase, we handled infinite values and NaNs by replacing them with zeros and removed duplicate entries to ensure data quality. Categorical features, such as IP addresses and Protocols, were transformed using Label Encoding to make them suitable for mathematical modeling. Finally, all features were standardized using \`StandardScaler\` to ensure zero mean and unit variance, a crucial step for distance-based algorithms like K-Means and Isolation Forest.

= Data Handling Chain Deployment

== Pipeline Architecture
The data handling chain was designed to operate in *batch mode*, processing static files rather than a live stream. This architecture ensures reproducibility and allows for complex global analysis. The pipeline follows a sequence of ingestion, iterative loading, and transformation. The system first uses the \`glob\` library to recursively locate all CSV files within the extracted dataset directory. To prevent memory overflows, the pipeline iterates through the file list, reading and cleaning each file individually before appending it to a master DataFrame, while explicitly calling garbage collection to manage RAM usage. Finally, the aggregated data undergoes feature selection and standardization as described in the preprocessing strategy.

== Train/Test Split
To evaluate our models rigorously, we employed a stratified train/test split strategy:
- *Ratio:* 70% Training, 30% Testing.
- *Stratification:* The split was stratified based on the target label ($y$).

Stratification was essential due to the extreme class imbalance. It ensures that the minority class (Benign) is represented proportionally in both the training and testing sets, preventing a scenario where the test set might contain zero benign instances.

= Algorithm Benchmarking: Unsupervised Learning

Unsupervised learning algorithms are critical in cybersecurity for detecting "zero-day" attacks—threats that have not been seen before and thus have no corresponding label in the training data. These models operate on the assumption that anomalies are rare and statistically distinct from normal traffic.

However, as noted in our dataset characterization, the CIC IoT-DIAD 2024 dataset presents a unique challenge: the "anomalous" traffic (Attacks) constitutes ~98% of the data. This inversion of the typical normal/anomaly ratio heavily impacts the performance of these algorithms.

== Isolation Forest

=== Theoretical Description
Isolation Forest is an ensemble-based anomaly detection method that differs from distance-based techniques by relying on the principle of isolation. The algorithm builds an ensemble of random trees for a given dataset, operating on the premise that anomalies are "few and different." Random partitioning of the feature space is more likely to isolate anomalies in fewer steps, resulting in shorter path lengths compared to normal instances, which are clustered and require more cuts to isolate. An anomaly score is then calculated based on the average path length required to isolate a point, where shorter paths imply a higher likelihood of being an anomaly.

=== Results & Analysis
*Configuration:* Contamination = 0.1

#table(
  columns: 2,
  table.header([Metric], [Value]),
  [Precision (Weighted)], [0.9480],
  [Recall (Weighted)], [0.1095],
  [Balanced Accuracy], [0.4183],
  [MCC], [-0.0650]
)

#align(center)[
  *Confusion Matrix:*
  #table(
    columns: 3,
    [], [Pred: Benign], [Pred: Attack],
    [Actual: Benign], [880], [315],
    [Actual: Attack], [71733], [7980]
  )
]

*Analysis:*
The model performed poorly, with a very low Recall (0.1095) and a negative MCC (-0.0650). This failure is directly attributable to the class imbalance. Isolation Forest assumes anomalies are the minority. Here, attacks are the majority. The algorithm likely isolated the *actual* minority (Benign traffic) as anomalies, or simply failed to distinguish the massive volume of attack traffic from "normal" behavior because the attack traffic *is* the statistical norm in this dataset.

== K-Means Clustering

=== Theoretical Description
K-Means is a centroid-based clustering algorithm that partitions data into $k$ distinct clusters. It iteratively assigns data points to the nearest cluster centroid and updates the centroids to minimize the within-cluster variance using Euclidean distance. In an unsupervised anomaly detection setting, we assume that one cluster represents normal traffic while the other represents attack traffic, or that points significantly distant from their centroids are anomalies.

=== Results & Analysis
*Configuration:* $k=2$

#table(
  columns: 2,
  table.header([Metric], [Value]),
  [Precision (Weighted)], [0.9701],
  [Recall (Weighted)], [0.9375],
  [Balanced Accuracy], [0.4799],
  [MCC], [-0.0227]
)

#align(center)[
  *Confusion Matrix:*
  #table(
    columns: 3,
    [], [Pred: Benign], [Pred: Attack],
    [Actual: Benign], [10], [1185],
    [Actual: Attack], [3869], [75844]
  )
]

*Analysis:*
While the Recall appears high (0.9375), the Balanced Accuracy (0.4799) reveals the truth: the model is essentially guessing the majority class. It correctly identified most attacks simply because most data *are* attacks, but it failed to cluster the benign traffic separately (only 10 benign instances correctly classified). The negative MCC confirms that the correlation between predictions and ground truth is worse than random guessing.

== PCA Reconstruction

=== Theoretical Description
Principal Component Analysis (PCA) is a dimensionality reduction technique that identifies the principal components, or directions of maximum variance, in the data. An anomaly detection model based on PCA projects data into a lower-dimensional subspace and then reconstructs it. The detection logic is based on the expectation that normal data, having been used to learn the correlation structure, will have a low reconstruction error, while anomalies that do not conform to this structure will exhibit a high reconstruction error.

=== Results & Analysis
*Configuration:* 95% Variance Retention

#table(
  columns: 2,
  table.header([Metric], [Value]),
  [Precision (Weighted)], [0.8968],
  [Recall (Weighted)], [0.0968],
  [Balanced Accuracy], [0.2420],
  [MCC], [-0.2075]
)

#align(center)[
  *Confusion Matrix:*
  #table(
    columns: 3,
    [], [Pred: Benign], [Pred: Attack],
    [Actual: Benign], [468], [727],
    [Actual: Attack], [72349], [7364]
  )
]

*Analysis:*
PCA yielded the worst performance among the three unsupervised methods. The high number of False Negatives (72,349 attacks classified as benign) suggests that the attack traffic shares significant structural variance with the benign traffic—or, more likely, because the model was trained on the full dataset (dominated by attacks), it learned the *attack* structure as the "normal" baseline, thus failing to flag attacks as anomalies.

= Algorithm Benchmarking: Supervised Learning

In contrast to unsupervised methods, supervised learning algorithms leverage labeled data to learn the specific characteristics of "Benign" and "Attack" traffic. Given the distinct signatures of attacks in the CIC IoT-DIAD 2024 dataset, we hypothesized that supervised models would significantly outperform their unsupervised counterparts.

== Decision Tree

=== Theoretical Description
A Decision Tree is a non-parametric supervised learning method that predicts the value of a target variable by learning simple decision rules inferred from the data features. It structures these rules as a tree where each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label. The tree splits data to maximize Information Gain or minimize Gini Impurity, creating homogeneous subsets of data at each node.

=== Results & Analysis

#table(
  columns: 2,
  table.header([Metric], [Value]),
  [Precision (Weighted)], [1.0000],
  [Recall (Weighted)], [1.0000],
  [Balanced Accuracy], [1.0000],
  [MCC], [1.0000]
)

#align(center)[
  *Confusion Matrix:*
  #table(
    columns: 3,
    [], [Pred: Benign], [Pred: Attack],
    [Actual: Benign], [1195], [0],
    [Actual: Attack], [0], [79713]
  )
]

*Analysis:*
The Decision Tree achieved perfect classification (1.0 across all metrics). It correctly identified all 1,195 benign instances and all 79,713 attack instances with zero False Positives or False Negatives. This indicates that the features in the dataset are linearly separable or follow a very distinct logic that a simple rule-based model can fully capture.

== Random Forest

=== Theoretical Description
Random Forest is an ensemble learning method that builds a multitude of decision trees at training time using a technique called Bagging. Each tree is trained on a random subset of the data with replacement, and at each split, only a random subset of features is considered. The final output is determined by the majority vote of the individual trees, an approach that effectively corrects for the tendency of individual decision trees to overfit to their training sets.

=== Results & Analysis
*Configuration:* n_estimators=100

#table(
  columns: 2,
  table.header([Metric], [Value]),
  [Precision (Weighted)], [0.9996],
  [Recall (Weighted)], [0.9996],
  [Balanced Accuracy], [0.9870],
  [MCC], [0.9855]
)

#align(center)[
  *Confusion Matrix:*
  #table(
    columns: 3,
    [], [Pred: Benign], [Pred: Attack],
    [Actual: Benign], [1164], [31],
    [Actual: Attack], [3], [79710]
  )
]

*Analysis:*
Random Forest performed exceptionally well, though slightly "worse" than the single Decision Tree (missing 3 attacks and misclassifying 31 benign flows). In most real-world scenarios, this small error margin is preferred over a perfect score, as it suggests the model is generalizing rather than overfitting. However, given the Decision Tree's perfect score on the test set, the "errors" here might simply be due to the random feature subspace sampling excluding the critical feature for those specific instances.

== XGBoost

=== Theoretical Description
XGBoost, or Extreme Gradient Boosting, is an optimized distributed gradient boosting library. Unlike Random Forest, which builds trees in parallel, XGBoost builds trees sequentially where each new tree is trained to correct the errors of the previous ones. The algorithm uses a gradient descent approach to minimize the loss function and includes regularization terms to control overfitting, making it exceptionally powerful for structured data analysis.

=== Results & Analysis

#table(
  columns: 2,
  table.header([Metric], [Value]),
  [Precision (Weighted)], [1.0000],
  [Recall (Weighted)], [1.0000],
  [Balanced Accuracy], [1.0000],
  [MCC], [1.0000]
)

#align(center)[
  *Confusion Matrix:*
  #table(
    columns: 3,
    [], [Pred: Benign], [Pred: Attack],
    [Actual: Benign], [1195], [0],
    [Actual: Attack], [0], [79713]
  )
]

*Analysis:*
Like the Decision Tree, XGBoost achieved perfect performance. This reinforces the conclusion that the dataset contains highly distinct feature patterns that separate benign traffic from attacks. The sequential error-correction of XGBoost, combined with the clear signal in the data, resulted in a flawless classifier for this test set.

= Adversarial Attacks

While our supervised models achieved near-perfect accuracy on standard test data, a critical question remains: Are these models robust against active evasion attempts? Adversarial attacks involve introducing small, carefully calculated perturbations to input data, designed to deceive a machine learning model into making an incorrect classification.

== Methodology
To evaluate this robustness, we utilized the Adversarial Robustness Toolbox (ART). We first trained a simple Neural Network using Keras to act as our surrogate target model, consisting of dense layers with ReLU activation. For the attack itself, we employed the Fast Gradient Sign Method (FGSM). This algorithm computes the gradient of the loss function with respect to the input data and then creates a new feature vector by adding a small amount of noise in the direction of the gradient that maximizes the loss. We set the perturbation magnitude $epsilon$ to 0.2 to test the model's sensitivity to even moderate changes.

== Results
We generated adversarial examples for a subset of the test data and evaluated the model's performance on both clean and adversarial inputs.

#table(
  columns: 2,
  table.header([Data Type], [Model Accuracy]),
  [Clean Data], [100.00%],
  [Adversarial Data (FGSM)], [74.00%]
)

The attack successfully degraded the model's performance by 26%. For instance, we observed specific feature perturbations where an original value of -1.3493 was shifted to -1.5493 to induce a classification error.

== Analysis
The experiment demonstrates a critical vulnerability. Despite the model having 100% accuracy on standard traffic, a relatively simple gradient-based attack was able to force misclassification in more than a quarter of the cases. This proves that high accuracy does not equate to security. In a real-world scenario, an attacker aware of the defense system could slightly modify packet timings or sizes to bypass detection entirely.

= Cybersecurity Conclusions

This project successfully deployed a batch processing data chain to analyze the CIC IoT-DIAD 2024 dataset, yielding significant insights into both the data and the efficacy of various machine learning approaches.

== Dataset & Methodology
Our analysis revealed that the dataset is overwhelmingly dominated by attack traffic, which constitutes approximately 98% of the samples. This inverted imbalance rendered standard unsupervised anomaly detection methods, such as Isolation Forest and PCA, largely ineffective, as these algorithms generally operate under the assumption that anomalies are rare occurrences. Conversely, supervised models like Decision Trees and XGBoost achieved near-perfect results, suggesting that the attack signatures in this dataset are highly distinct and linearly separable from benign traffic.

== Security Implications
The high performance of supervised metrics can create an "illusion of safety." As demonstrated in our adversarial experiments, even "perfect" models can be brittle, with a 26% drop in accuracy observed from a simple FGSM attack. To build a truly robust defense-in-depth strategy, we recommend a multi-faceted approach. First, supervised models like XGBoost should be deployed for their high precision against known threats. Second, these models should be hardened through adversarial training to increase their resilience against evasion techniques. Finally, unsupervised models should be re-calibrated—for instance, by training exclusively on benign traffic—to serve as a secondary safety net for detecting zero-day anomalies that may not follow established attack patterns.

= References
+ CIC IoT-DIAD 2024 Dataset, University of New Brunswick.
+ Scikit-learn: Machine Learning in Python.
+ Adversarial Robustness Toolbox (ART) Documentation.
