# Optimal Email Time Slot Prediction
## Bank Marketing Campaign Optimization
### Technical Documentation

## 1. Problem Statement

The bank faces a critical optimization challenge in their marketing communications:

1. **Regulatory Constraints**: Limited number of communications allowed per timeframe
2. **Cost Constraints**: Need to maximize ROI on each communication
3. **Engagement Challenge**: Must predict optimal timing for maximum open rates
4. **Personalization Requirement**: Different customers have different optimal engagement times

The core task is to predict and rank 28 weekly time slots (4 daily slots × 7 days) for each customer, optimizing for email open rates. The prediction must account for historical engagement patterns, customer characteristics, and temporal behavior patterns.

## 2. Data Analysis & Preprocessing

### 2.1 Data Sources

#### Communication History Data
- **Timespan**: 6 months
- **Scale**: ~200,000 customers
- **Key Fields**:
  ```
  - Customer_Code: Unique identifier
  - Offer_id/subid: Offer identifiers
  - Batch_id: Communication tracking
  - Send_timestamp: Email send time
  - Open_timestamp: Email open time
  ```

#### Customer CDNA Data
- **Nature**: Demographic and behavioral features
- **Update Frequency**: Weekly
- **Key Characteristics**:
  - 303 total columns
  - Mix of categorical and numerical features
  - Varying levels of missing data
  - Multiple data types (object, int64, float64, bool)

### 2.2 Data Quality Analysis

#### Missing Data Analysis
```python
# Distribution of null ratios
(-0.002, 0.1]     62 columns
(0.1, 0.2]         5 columns
(0.2, 0.3]         7 columns
(0.3, 0.4]        32 columns
(0.4, 0.5]         3 columns
(0.5, 0.6]        37 columns
(0.6, 0.7]        14 columns
(0.7, 0.8]        15 columns
(0.8, 0.9]        16 columns
(0.9, 1.0]       112 columns
```

Based on this analysis, we implemented a strategic feature selection:
- Primary focus on columns with ≤25% nulls (68 columns)
- Secondary consideration for columns with ≤50% nulls (109 columns)
- Exclusion of high-null columns to maintain data integrity

### 2.3 Data Preprocessing Pipeline

#### 2.3.1 Time Slot Definition
```python
def get_slot_number_for_send(timestamp):
    """
    Maps timestamps to 28 weekly slots
    Returns slot_1 to slot_28 based on:
      - day_of_week: 0..6 (Mon..Sun)
      - hour: from 9..21 in 3-hour increments
    """
    day = timestamp.weekday()
    hour = timestamp.hour
    if hour < 9 or hour >= 21:
        return None
    block = (hour - 9) // 3
    slot_index = day*4 + block + 1
    return f"slot_{slot_index}"
```

#### 2.3.2 Custom Imputation Strategy
```python
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.strategies = {}
        self.fill_values = {}
        self.age_medians = None
        self.city_tiers = None

    def fit(self, X, y=None):
        # Age imputation using age group correlations
        mask_train = X['v2'].notna() & X['v80'].notna()
        df_train_valid = X.loc[mask_train, ['v2', 'v80']]
        self.age_medians = df_train_valid.groupby('v2')['v80'].median()
        
        # City-tier mapping for location-based features
        X['v6'] = X['v6'].str.upper()
        mask = X['v6'].notna() & X['v101'].notna()
        df_valid = X.loc[mask, ['v6', 'v101']]
        mode_df = df_valid.groupby('v6')['v101'].agg(
            lambda x: x.value_counts().index[0]
        )
        self.city_tiers = dict(zip(mode_df.index, mode_df.values))
```

Imputation strategies were chosen based on:
1. Data type
2. Cardinality
3. Business logic
4. Statistical distribution

## 3. Feature Engineering

### 3.1 Selected Features

#### Categorical Features (13 columns)
```python
CATEGORICAL_COLS = [
    'product_category',      # Product type
    'product_sub_category',  # Product subtype
    'send_slot',            # Time slot
    'v6',                   # Location
    'v9',                   # Country
    'v10',                  # Region
    'v34', 'v35', 'v36',   # Geographic
    'v37',                  # Location type
    'v54',                  # Gender
    'v101',                # City tier
    'v102'                 # Employment
]
```

#### Numerical Features (21 columns)
```python
NUMERICAL_COLS = [
    # Engagement metrics
    'is_open', 
    'day_of_week', 
    'hour',
    'send_slot_ML',
    
    # Customer metrics
    'v80',                  # Age
    'v287', 'v288', 'v289', # Financial indicators
    'v290', 'v291', 'v292', # Activity metrics
    'v293', 'v294', 'v295', # Response patterns
    'v296', 'v297', 'v298', # Behavioral scores
    'v299', 'v300', 'v301', # Usage metrics
    'v302'                  # Engagement score
]
```

### 3.2 Feature Transformation

#### Encoding Strategy
1. **Categorical Variables**:
   - Label encoding for low-cardinality features
   - Special handling for high-cardinality features
   ```python
   if X_filtered[column].nunique() > 100:
       self.strategies[column] = 'constant'
       self.fill_values[column] = 'MISSING'
   else:
       self.strategies[column] = 'mode'
       self.fill_values[column] = X_filtered[column].mode()[0]
   ```

2. **Numerical Variables**:
   - Standardization for continuous features
   - Custom binning for specific metrics
   ```python
   if X_filtered[column].nunique() > 20:
       self.strategies[column] = 'median'
       self.fill_values[column] = X_filtered[column].median()
   else:
       self.strategies[column] = 'mode'
       self.fill_values[column] = X_filtered[column].mode()[0]
   ```

## 4. Neural Network Architecture

### 4.1 Model Design
```python
def build_model(input_dim, num_slots=28):
    model = Sequential([
        # Input Layer
        Dense(256, activation='relu', 
              kernel_regularizer=l2(0.01), 
              input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.5),
        
        # Hidden Layer 1
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        # Hidden Layer 2
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output Layer
        Dense(num_slots, activation='softmax')
    ])
```

### 4.2 Architecture Decisions

#### Layer Design Rationale
1. **Input Layer (256 units)**
   - Large initial layer to capture complex feature interactions
   - L2 regularization to prevent overfitting
   - High dropout (0.5) for robust feature learning

2. **Hidden Layer 1 (128 units)**
   - Reduced dimensionality for feature abstraction
   - Batch normalization for training stability
   - Moderate dropout (0.4) for regularization

3. **Hidden Layer 2 (64 units)**
   - Further dimensionality reduction
   - Final feature refinement
   - Lower dropout (0.3) to preserve learned patterns

4. **Output Layer (28 units)**
   - One unit per time slot
   - Softmax activation for probability distribution
   - No dropout to maintain prediction stability

#### Optimization Configuration
```python
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

- **Learning Rate**: 0.0003
  - Conservative rate to ensure stable convergence
  - Prevents overshooting optimal weights
  - Balanced between speed and accuracy

- **Loss Function**: Categorical Crossentropy
  - Appropriate for multi-class classification
  - Handles probability distributions well
  - Provides smooth gradients

### 4.3 Training Process

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=1024,
    callbacks=[early_stopping],
    verbose=1
)
```

#### Training Parameters
- **Batch Size**: 1024
  - Optimized for memory usage
  - Provides stable gradient updates
  - Good balance of speed and convergence

- **Early Stopping**
  - Monitors validation loss
  - Patience of 3 epochs
  - Restores best weights automatically

## 5. Prediction Pipeline

### 5.1 Prediction Process
```python
def train_and_predict(train_data, test_data, customer_codes):
    # Feature preparation
    X_train, encoders, scaler = prepare_features(
        train_data, is_training=True
    )
    
    # Target encoding
    y_slots = train_data['send_slot_ML'].apply(
        lambda x: x - 1 if x > 0 else 27
    )
    y_train = pd.get_dummies(y_slots, columns=range(28))
    
    # Model training
    model = build_model(X_train.shape[1])
    
    # Prediction and ranking
    pred_probs = model.predict(X_test)
    
    # Generate slot rankings
    slot_rankings = []
    for customer_code in customer_codes:
        if customer_code in customer_preds:
            probs = customer_preds[customer_code]
            ranked_slots = np.argsort(probs)[::-1]
            slot_names = [f"slot_{i+1}" for i in ranked_slots]
            slot_rankings.append(slot_names)
```

### 5.2 Validation Framework

#### Data Validation
```python
def validate_predictions(predictions, slot_rankings):
    """Comprehensive validation suite"""
    # Format validation
    for customer_predictions in slot_rankings:
        if len(customer_predictions) != 28:
            raise ValueError(
                f"Each prediction must have 28 slots, "
                f"got {len(customer_predictions)}"
            )
        
        # Duplicate check
        if len(set(customer_predictions)) != 28:
            raise ValueError("Predictions contain duplicate slots")
        
        # Slot format validation
        for slot in customer_predictions:
            if not slot.startswith("slot_"):
                raise ValueError(f"Invalid slot format: {slot}")
```

#### Performance Metrics
```python
def evaluate_model(model, X_val, y_val, history):
    # Calculate metrics
    predictions = model.predict(X_val)
    y_pred = predictions.argmax(axis=1)
    y_true = y_val.argmax(axis=1)
    
    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Training analysis
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    loss_diff = final_val_loss - final_train_loss
```

## 6. Memory Optimization

### 6.1 Data Type Optimization
```python
def optimize_dataframe(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
        elif df[col].dtype == 'object':
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
```

### 6.2 Chunked Processing
```python
def save_in_chunks(df, filename, chunk_size=100000):
    # First chunk with headers
    df.iloc[:chunk_size].to_csv(filename, index=False)
    
    # Append remaining chunks
    for i in range(chunk_size, len(df), chunk_size):
        df.iloc[i:i+chunk_size].to_csv(
            filename, 
            mode='a', 
            header=False, 
            index=False
        )
```

## 7. Implementation Notes

### 7.1 Key Dependencies
```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
```

### 7.2 Environment Requirements
- Python 3.7+
- TensorFlow 2.x
- 16GB+ RAM
- CUDA-capable GPU (recommended)

This technical documentation provides a comprehensive overview of our implemented solution, focusing on the actual components and decision-making process that went into building this prediction system.
