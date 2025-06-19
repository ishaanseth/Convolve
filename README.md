# Convolve - Pan IIT AI ML Hackathon Submissions
This repository contains the submissions of **Team Backprop Buddies** for Convolve, the Pan IIT AI ML Hackathon.
With these submissions, our team was placed in the **Top 52 out of 1500+ registered teams**.

The repository includes solutions and documentation for two distinct rounds:
1.  A risk prediction model (Round 1).
2.  An optimal email engagement time slot prediction model (Round 2).

Each round's details, including the problem statement, solution approach, technologies used, and key insights, are documented in their respective sections below.

---

## Teammates
  * Ishaan Seth (EE23B110)
  * Aditya Sawant (CS23B003)
  * Ritwik Mishra (ME23B237)

---

## Round 1: Risk Prediction

### Problem Statement
The primary goal of this round was to predict a "bad_flag" indicating the risk of customer default based on various customer attributes. These attributes included 'onus_attribute' (related to credit and account specifics, with 'onus_attribute_1' identified as credit limit), 'transaction_attribute', 'bureau_enquiry_attribute', and 'bureau_attribute' (credit bureau data).

### Solution Approach
The solution involved several key stages:
1.  **Feature Engineering**:
    *   **Imputation**: Missing values were handled by imputing transaction attributes with 0, onus and bureau_enquiry attributes with their respective medians, and bureau attributes with their means.
    *   **Feature Selection**: Low-variance features were removed, followed by correlation-based pruning (threshold of 0.90) to reduce multicollinearity. `RandomForest` feature importance was also checked.
2.  **Modeling**:
    *   A **Feed Forward Neural Network** was the primary model.
    *   **Advanced Techniques**:
        *   **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to address class imbalance in the training data (target minority class set to 5% of the majority class).
        *   **Class weights** were computed and used during model training to give more importance to the minority class.
        *   The neural network architecture included **`Dense` layers** (32 units -> 16 units -> 1 unit sigmoid output), **`Dropout` layers** (0.4 after each dense layer) for regularization, and **`L2` kernel regularization** (1e-4) on the `Dense` layers.
        *   The **`Adam` optimizer** (learning rate 0.0005) was used with **`binary_crossentropy`** loss.
        *   **`EarlyStopping`** (monitoring validation AUC with patience 5, restoring best weights) was employed to prevent overfitting.
3.  **Data Exploration (Bad Flag Subset)**:
    *   `KMeans` clustering (k=3) was performed on the subset of data where `bad_flag == 1`.
    *   `PCA` was used to visualize these clusters in 2D.

### Frameworks/Libraries Used
-   **Data Manipulation**: `numpy`, `pandas`
-   **Machine Learning & Neural Networks**: `tensorflow`, `keras` (for `Sequential` models, `Dense`, `Dropout`, `Adam`, `l2`, `EarlyStopping`)
-   **Preprocessing & Metrics**: `scikit-learn` (for `RandomForestClassifier`, `VarianceThreshold`, `StandardScaler`, `class_weight`, `train_test_split`, `metrics`, `KMeans`, `PCA`)
-   **Imbalanced Data Handling**: `imblearn` (for `SMOTE`)
-   **Plotting**: `seaborn`, `matplotlib`
-   **Environment**: The project was developed in a `google.colab` environment.

### Results & Insights
-   The Neural Network model demonstrated a significant performance improvement of approximately 20% in accuracy over classical ML models like `LogisticRegression` and `DecisionTree`s.
-   A validation accuracy of around 85% was achieved by the Neural Network.
-   Key influential features identified included `onus_attribute_1` (credit limit) and `transaction_attribute` (strongly associated with payments to merchants, signaling potential financial strain).
-   Data analysis revealed that approximately 15% of users were classified as high-risk.
-   Model performance was highlighted by Precision-Recall curves and AUC scores, with the Neural Network achieving an AUC of approximately 0.85 compared to ~0.65 for classical models.

## Round 2: Email Engagement Time Slot Prediction

### Problem Statement
The objective of this round was to optimize bank marketing campaigns by predicting and ranking 28 weekly time slots (4 daily slots × 7 days, from 9 AM to 9 PM in 3-hour blocks) for each customer. The goal was to maximize email open rates while considering regulatory communication limits and cost constraints, requiring personalized timing predictions.

### Data Sources & Preprocessing
The solution utilized two main data sources:
1.  **Communication History Data (`train_action_history.csv`, `test_action_history.csv`):** Contained ~6 months of communication records for approximately 200,000 customers, including send/open timestamps and offer details.
    *   **Cleaning:** Duplicates based on `customer_code`, `batch_id`, `Offer_subid` were handled. Timestamps were validated to ensure `send_timestamp <= open_timestamp`.
2.  **Customer CDNA Data (`train_cdna_data.csv`, `test_cdna_data.csv`):** Provided weekly snapshots of customer demographic and behavioral features (303 columns).
    *   **Preprocessing (CustomImputer Strategy):**
        *   Focused on columns with ≤25% null values initially.
        *   Age (`v80`) was imputed using the median based on age group (`v2`).
        *   City tier (`v101`) was imputed using the mode based on city (`v6`).
        *   Gender (`v54`) and Country (`v9`) fields were normalized.
        *   Other missing values were imputed based on data type and cardinality ('MISSING' for high-cardinality objects, mode for low-cardinality objects/booleans, median for high-nunique numerics, mode for low-nunique numerics).
        *   Certain redundant or processed columns (`v2`, `v27`, `v29`, `v31`) were dropped.

### Feature Engineering
Several types of features were engineered:
1.  **Basic Engagement Metrics:**
    *   `is_open`: Binary flag derived from `open_timestamp`.
    *   `day_of_week`, `hour`: Extracted from `send_timestamp`.
    *   `send_slot`, `open_slot`: Categorical features mapping timestamps to the 28 weekly slots (e.g., `slot_1`).
    *   `send_slot_ML`, `open_slot_ML`: Numerical representation of these slots (0-27 or 1-28).
    *   `open_delay`: Time difference in seconds between email send and open.
2.  **Customer-Level Aggregations:**
    *   Metrics like `open_rate`, `total_opens`, `total_emails`, `unique_slots` used, `most_common_send_slot`, `most_common_open_slot`, `avg_response_time`, and `median_response_time` were calculated per customer.
3.  **Time-Based Aggregations:**
    *   Hourly, daily, and slot-wise statistics (counts, means, sums of `is_open`) were generated to capture temporal patterns.
4.  **Data Merging:**
    *   `train_action` was merged with the imputed `train_cdna` data using `pd.merge_asof`. This time-sensitive merge joined action data with the latest available CDNA data up to the `send_date`, grouped by `customer_code`.
    *   `test_action` was merged with imputed `test_cdna` data using a standard left merge on `customer_code`.
5.  **Memory Optimization:**
    *   DataFrame dtypes were optimized (e.g., `float64` to `float32`, `int64` to `int32`).
    *   Large merged DataFrames were saved and processed in chunks.

### Modeling Approach
A Feed Forward Neural Network was employed for predicting the optimal time slot.
1.  **Feature Preparation for Model:**
    *   A curated list of 13 `CATEGORICAL_COLS` (e.g., `product_category`, `send_slot`, `v6`, `v54`, `v101`) and 21 `NUMERICAL_COLS` (e.g., `is_open`, `hour`, `v80`, `v287-v302` series) were selected.
    *   **Categorical Features:** Underwent `LabelEncoding`. 'MISSING' values and unseen categories in the test set were handled.
    *   **Numerical Features:** Standardized using `StandardScaler` and missing values filled with 0.
2.  **Neural Network Architecture:**
    *   **Input Layer:** `Dense` layer with 256 units, `ReLU` activation, and `L2` kernel regularization (0.01).
    *   **Hidden Layers:**
        *   `Dense` (128 units, `ReLU`) -> `BatchNormalization` -> `Dropout` (0.5)
        *   `Dense` (128 units, `ReLU`) -> `BatchNormalization` -> `Dropout` (0.4) (Note: Documentation mentions 256->128->64, but code might differ slightly or this is a summary)
        *   `Dense` (64 units, `ReLU`) -> `BatchNormalization` -> `Dropout` (0.3)
    *   **Output Layer:** `Dense` layer with 28 units (one for each slot) and `softmax` activation.
3.  **Training:**
    *   **Optimizer:** `Adam` with a learning rate of 0.0003.
    *   **Loss Function:** `CategoricalCrossentropy`.
    *   **Target Variable:** The `send_slot_ML` was one-hot encoded for training.
    *   **Callbacks:** `EarlyStopping` was used, monitoring `val_loss` with a patience of 3 epochs and restoring the best weights.
    *   **Batch Size:** 1024.

### Frameworks/Libraries Used
-   **Data Handling**: `pandas`, `numpy`
-   **Neural Networks**: `tensorflow.keras` (`Sequential`, `Dense`, `Dropout`, `BatchNormalization`, `Adam`, `EarlyStopping`, `l2`)
-   **Preprocessing**: `scikit-learn` (`BaseEstimator`, `TransformerMixin`, `StandardScaler`, `LabelEncoder`, `train_test_split`)
-   **Logging**: `logging`

### Prediction & Submission
-   For test data, predictions from the model (probabilities for each of the 28 slots) were generated.
-   If a customer had multiple actions in the test set, their predicted probabilities were averaged.
-   Slots were then ranked based on these (averaged) probabilities for each customer.
-   A **default slot ranking** (initially sequential, then updated based on overall prediction frequencies from processed customers) was used for customers in the submission list but not present in the `test_action` data or those who ended up with no features after preprocessing.
-   The final submission included `customer_code` and `predicted_slots_order`.
