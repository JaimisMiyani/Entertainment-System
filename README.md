## Data Preprocessing and Exploration

This section covers the preprocessing and exploration steps performed on the dataset to prepare it for modeling.

### 1. Importing Required Libraries

We begin by importing essential libraries for data manipulation, visualization, and preprocessing:

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **matplotlib & seaborn**: For data visualization.
- **scikit-learn**: For machine learning tasks, including data splitting, scaling, and encoding.
- **scipy**: For statistical operations, including outlier detection.

### 2. Loading the Dataset

The dataset is loaded into a pandas DataFrame using the `read_csv`.

## Data Visualization

This section presents various visualizations to explore the distribution and relationships of key features in the dataset.

### 1. Distribution of Numerical Features

The distribution of important numerical features—**popularity**, **vote_average**, and **vote_count**—was visualized using histograms. These visualizations provide insights into the spread and skewness of the data:

- **Popularity**: Shows how popular the movies are based on the dataset.
- **Vote Average**: Represents the average rating of movies.
- **Vote Count**: Displays the number of votes each movie received.

### 2. Correlation Matrix

A correlation matrix was plotted to analyze the relationships between numerical features in the dataset. This helps in identifying multicollinearity, where features may be highly correlated with each other:

- The heatmap shows correlations ranging from -1 (perfect negative correlation) to +1 (perfect positive correlation).
- Features with high correlation can be considered for dimensionality reduction techniques if needed.

### 3. Categorical Feature Analysis: Language Distribution

A count plot was created to visualize the distribution of movies by their original language. This helps in understanding the linguistic diversity in the dataset:

- The count plot shows the number of movies available in each language, which may influence model predictions if the language is an important feature.

### 4. Yearly Analysis: Movies by Release Year

A line plot was generated to observe the distribution of movies released each year. This visualization helps to:

- Identify trends in the number of movies released over time.
- Understand the impact of historical events on the film industry.

### 5. Scatter Plots for Numerical Feature Relationships

Scatter plots were used to explore the relationships between pairs of numerical features:

- **Popularity vs. Vote Average**: Analyzes how movie popularity correlates with average votes.
- **Vote Count vs. Vote Average**: Examines whether movies with more votes have higher average ratings.

These plots are color-coded by the original language, providing additional insights into how language may influence these relationships.

### 6. Popularity Over Time

Finally, a line plot was created to analyze the trend of average movie popularity over time:

- This plot shows how the popularity of movies has evolved across different release years.
- It can reveal whether newer or older movies tend to be more popular.

These visualizations provide a comprehensive understanding of the dataset and help in feature selection and model-building processes.

![download](https://github.com/user-attachments/assets/5ef14115-db7e-4a1d-9718-d71ee173b282)
![download (1)](https://github.com/user-attachments/assets/bad843e5-6f41-48dc-8e94-1271f8addd05)
![download (2)](https://github.com/user-attachments/assets/ec70ad47-6a14-48b0-9a5a-f500322896b5)
![download (3)](https://github.com/user-attachments/assets/22e09c99-73ff-466b-908c-6d38a00b0927)


## Neural Collaborative Filtering (NCF) Model

This section describes the implementation of a Neural Collaborative Filtering (NCF) model, a popular approach used in recommender systems.

### 1. Model Architecture

The NCF model architecture involves:

- **User and Item Inputs**: Separate input layers for users and items.
- **Embedding Layers**: Each user and item is represented as an embedding vector. The embedding dimension is set to 50.
- **Flattening**: The embedding vectors are flattened to create user and item vectors.
- **Concatenation**: The user and item vectors are concatenated to form a combined vector.
- **Dense Layers**: The combined vector is passed through a fully connected (dense) layer with 128 units and ReLU activation.
- **Output Layer**: The final output is a single unit with a sigmoid activation function, representing the probability of a user interacting with an item.

### 2. Compilation

The model is compiled using:

- **Optimizer**: Adam optimizer is used for its efficiency in handling sparse data and adaptability.
- **Loss Function**: Binary crossentropy is used as the loss function since the model predicts interaction (binary classification).
- **Metrics**: Accuracy is tracked as the primary metric for evaluation.

### 3. Training the NCF Model

For demonstration purposes, simulated training data is generated:

- **User IDs**: Randomly generated integers representing user IDs.
- **Item IDs**: Randomly generated integers representing item IDs.
- **Labels**: Random binary labels indicating interaction (1) or no interaction (0).

The model is trained using this simulated data over 10 epochs with an 80-20 split for training and validation.

### 4. Practical Application

In a real-world scenario, the model would be trained on actual user-item interaction data, such as user ratings or purchase history. The embeddings learned by the model can be used to predict user preferences and recommend items accordingly.

### 5. Future Enhancements

To improve the model further, consider:

- **Adding More Layers**: Increase the complexity of the model by adding more dense layers or increasing the embedding dimensions.
- **Regularization**: Implement techniques like dropout or L2 regularization to prevent overfitting.
- **Hyperparameter Tuning**: Experiment with different values for the embedding dimensions, dense layer units, and activation functions to optimize performance.


## Convolutional Neural Network (CNN) Model

This section outlines the implementation of a Convolutional Neural Network (CNN) for a binary classification task using simulated data.

### 1. Model Architecture

The CNN model consists of the following layers:

- **Convolutional Layer**: 
  - Filters: 32 
  - Kernel Size: 3x3 
  - Activation: ReLU 
  - Input Shape: 64x64x3 (for RGB images)

- **Max Pooling Layer**:
  - Pool Size: 2x2 

- **Flatten Layer**: 
  - Converts the 2D matrix to a 1D vector.

- **Fully Connected (Dense) Layer**:
  - Units: 100 
  - Activation: ReLU

- **Dropout Layer**:
  - Rate: 30% 
  - To prevent overfitting.

- **Output Layer**:
  - Units: 1 
  - Activation: Sigmoid 
  - For binary classification.

### 2. Model Compilation

The model is compiled with:

- **Optimizer**: Adam 
- **Loss Function**: Binary Crossentropy 
- **Metrics**: Accuracy

### 3. Data Preparation

- **Simulated Data**:
  - Training set: 8361 samples, 64x64 RGB images.
  - Testing set: 2000 samples, 64x64 RGB images.
  - Binary labels (0 or 1).

- **Normalization**:
  - Pixel values are normalized by dividing by 255.0.

### 4. Training the CNN Model

The model is trained with the following hyperparameters:

- **Epochs**: 10
- **Batch Size**: 32
- **Validation Split**: 20%

During training, the loss values for both training and validation sets are tracked to monitor the model's performance.

### 5. Model Evaluation

After training, the model is evaluated using the test set, and the following metrics are calculated:

- **Root Mean Squared Error (RMSE)**: Measures the model's prediction error.
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors.
- **R² Score**: Indicates the proportion of variance captured by the model.

### 6. Training and Validation Loss Visualization

The training process is visualized by plotting the training and validation loss over epochs to help identify any potential overfitting.

### 7. Future Work

To enhance the model, consider experimenting with different:

- Filter sizes and counts
- Pooling strategies
- Dropout rates
- Optimizers and learning rates

![download (4)](https://github.com/user-attachments/assets/25e94f57-ebbd-41c5-b172-5692d5018350)

## Recurrent Neural Network (RNN) Model

This section details the implementation of a Recurrent Neural Network (RNN) for a binary classification task using simulated sequential data.

### 1. Data Preparation

- **Sequential Data Simulation**:
  - The model is trained on a dataset with 8361 samples for training and 2000 samples for testing.
  - Each sample is a sequence of 10 time steps, with 1 feature at each time step.

- **Labels**:
  - Binary labels (0 or 1) are used to classify the sequences.

### 2. Model Architecture

The RNN model is built using the following layers:

- **SimpleRNN Layer**:
  - Units: 128 
  - Activation: ReLU 
  - Input Shape: (10, 1) (for sequences with 10 time steps and 1 feature per step)

- **Dense Output Layer**:
  - Units: 1 
  - Activation: Sigmoid 
  - For binary classification.

### 3. Model Compilation

The model is compiled with:

- **Optimizer**: Adam 
- **Loss Function**: Binary Crossentropy 
- **Metrics**: Accuracy

### 4. Training the RNN Model

The model is trained using the following hyperparameters:

- **Epochs**: 10
- **Batch Size**: 32
- **Validation Split**: 20%

The training process is monitored by tracking the loss values for both training and validation sets.

### 5. Model Evaluation

After training, the model is evaluated on the test set using the following metrics:

- **Root Mean Squared Error (RMSE)**: Measures the model's prediction error.
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors.
- **R² Score**: Indicates the proportion of variance captured by the model.

### 6. Training and Validation Loss Visualization

The training process is visualized by plotting the training and validation loss over epochs to help identify any potential overfitting.

### 7. Future Work

To improve the RNN model's performance, consider experimenting with:

- Different RNN architectures (e.g., LSTM, GRU)
- Varying the number of time steps and features
- Tuning hyperparameters such as the number of units, learning rate, and dropout rate.

![download (5)](https://github.com/user-attachments/assets/8c7879a6-b284-4e26-bf1c-78fc1f8d5a54)

## Project Summary

This project integrates predictions from three deep learning models—Neural Collaborative Filtering (NCF), Convolutional Neural Networks (CNN), and Recurrent Neural Networks (RNN)—to recommend movies to users. The NCF model predicts user preferences based on historical interactions, the CNN model evaluates visual content, and the RNN model analyzes sequential watch history. The final movie recommendations are generated by combining the outputs of these models, weighted according to their predictive strengths. The result is a list of movie titles tailored to a user's tastes, offering a robust and personalized recommendation system.


