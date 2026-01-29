#set page(paper: "presentation-16-9", margin: 2cm)
#set text(font: "DejaVu Sans", size: 20pt)

// Template for slides
#let slide(title: none, body) = {
  if title != none {
    align(center, text(24pt, weight: "bold", title))
    v(1em)
  }
  align(horizon, body)
  pagebreak()
}

// Title Slide
#slide(title: "CYBERML Project Report")[
  #text(size: 18pt, weight: "bold")[Design, Deployment, and Evaluation of a Cybersecurity Data Analysis Chain]
  
  #v(2em) 
  
  *Group 14* 
  Quentin Prunet, Oscar Le Dauphin, Hugo Schreiber, Aleksei Kotliarov
  
  #v(2em)
  2025-2026
]

// Introduction & Context
#slide(title: "Introduction & Context")[
  - *Context:* The proliferation of IoT devices has exponentially increased the cyber attack surface.
  - *Goal:* Design a robust, automated system for detecting malicious activity.
  - *Approach:* Batch processing data chain for cybersecurity analysis.
  - *Dataset:* CIC IoT-DIAD 2024.
]

// Project Objectives
#slide(title: "Project Objectives")[
  1. *Pipeline Deployment:* Implement a complete data handling chain (Ingestion $->$ Preprocessing $->$ Analysis).
  2. *Dataset Characterization:* Analyze distributions and underlying structure.
  3. *Benchmarking:*
     - *Unsupervised:* Anomaly Detection (3 algorithms).
     - *Supervised:* Classification (3 algorithms).
  4. *Bonus:* Evaluate robustness against *Adversarial Attacks*.
]

// Dataset Characterization
#slide(title: "Dataset Characterization")[
  *CIC IoT-DIAD 2024*

  #grid(
    columns: (2fr, 1fr),
    gutter: 1em,
    [
      - *Critical Finding:* Extreme Class Imbalance.
        - *Attack:* ~98% (Dominant)
        - *Benign:* ~2% (Rare)
      - *Implication:* Standard anomaly detection assumptions (anomalies are rare) are inverted.
    ],
    [
      #image("extracted_images/image_12_0.png", width: 100%)
    ]
  )
]

// Feature Correlation
#slide(title: "Feature Correlation")[
  *Correlation Matrix*
  
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      - High collinearity among flow-based statistics.
      - *Insight:* Suggests redundancy.
      - *Action:* Dimensionality reduction (PCA) could be effective, though non-linearity limits it.
    ],
    [
      #image("extracted_images/image_13_1.png", width: 100%)
    ]
  )
]

// Data Handling & Preprocessing
#slide(title: "Data Handling & Preprocessing")[
  - *Resource Strategy:*
    - Sampled *1%* of each file due to massive dataset size (GBs) and hardware constraints.
    - Preserved statistical properties.
  - *Preprocessing Steps:*
    1. *Cleaning:* Handle NaNs/Infinites, remove duplicates.
    2. *Encoding:* Label Encoding for Categoricals (IPs, Protocols).
    3. *Scaling:* `StandardScaler` (Zero mean, Unit variance).
  - *Split:* 70% Training / 30% Testing (Stratified by label).
]

// Benchmark: Unsupervised Learning
#slide(title: "Benchmark: Unsupervised Learning")[
  _Hypothesis: Anomalies are rare and statistically distinct._

  - *Algorithms:* Isolation Forest, K-Means, PCA Reconstruction.
  - *Results:*
    - *Isolation Forest:* Failed (Recall ~11%). Confused majority attack traffic for normal.
    - *K-Means:* Balanced Accuracy ~48%. Essentially guessed the majority class.
    - *PCA:* Worst performance.
  - *Conclusion:* Failed due to inverted class imbalance. The "anomaly" (attack) was the statistical norm.
]

// Benchmark: Supervised Learning
#slide(title: "Benchmark: Supervised Learning")[
  _Hypothesis: Labeled training can identify distinct attack signatures._

  - *Algorithms:* Decision Tree, Random Forest, XGBoost.
  - *Results:*
    - *Decision Tree:* *100%* across all metrics.
    - *XGBoost:* *100%* across all metrics.
    - *Random Forest:* *~99.96%* Precision/Recall.
  - *Conclusion:* Attack signatures are highly distinct and linearly separable from benign traffic.
]

// Bonus: Adversarial Attacks
#slide(title: "Bonus: Adversarial Attacks")[
  *Robustness Check*
  - *Method:* Fast Gradient Sign Method (FGSM) using Adversarial Robustness Toolbox (ART).
  - *Target:* Simple Neural Network (Surrogate).
  - *Perturbation:* $epsilon = 0.2$

  *Results*
  - *Clean Data Accuracy:* 100%
  - *Adversarial Data Accuracy:* *74%*
  - *Impact:* 26% drop in performance from minor noise injection.
]

// Conclusions
#slide(title: "Conclusions")[
  1. *Dataset Insight:* The 98% attack volume makes standard anomaly detection ineffective without specific calibration (e.g., training only on benign).
  2. *Model Performance:* Supervised models (XGBoost/Decision Tree) are extremely effective for this specific dataset.
  3. *Security Warning:* High accuracy $!=$ Security.
     - Models are brittle to adversarial examples.
]

// Recommendations
#slide(title: "Recommendations")[
  To build a true Defense-in-Depth strategy:

  1. *Deploy Supervised Models:* Use XGBoost for high-precision detection.
  2. *Adversarial Training:* Harden models against evasion attacks (FGSM).
  3. *Refine Unsupervised:* Train anomaly detectors *exclusively* on benign traffic to identify zero-day threats.
]
