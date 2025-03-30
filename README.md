# Sentiment Analysis using BERT and Variants on Women's E-Commerce Clothing Reviews

## Overview
This project aims to perform sentiment analysis on women's clothing e-commerce reviews using various BERT-based models. The goal is to classify the sentiment of reviews into different categories using Natural Language Processing (NLP) techniques. 

## Dataset
The dataset used in this project contains user-generated reviews on women's clothing items, including features such as:
- **Clothing ID**
- **Age**
- **Title**
- **Review Text**
- **Rating** (1-5)
- **Recommended IND** (Binary: 0 or 1)
- **Division, Department, and Class Name**

## Models Used
We implemented and evaluated the following transformer-based models:
- **BERT Base**
- **BERT Dense**
- **DistilBERT**
- **ALBERT**
- **RoBERTa**

Additionally, an **ensemble model** was created by aggregating predictions from all models to improve performance.

## Methodology
1. **Data Preprocessing**:
   - Text cleaning and tokenization
   - Combination of textual and numerical features into structured text inputs
   - Encoding labels for classification tasks
2. **Feature Engineering**:
   - Creation of a "Combined Text" feature incorporating numerical and categorical data into text format
3. **Model Training**:
   - Fine-tuning BERT-based models
   - Using different architectures to compare performance
4. **Evaluation Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
5. **Ensemble Model**:
   - Majority voting approach for final prediction

## Results
Performance evaluation of the models based on Accuracy, Precision, Recall, and F1 Score showed that:
- **ALBERT** performed best for the **Rating** classification task.
- **RoBERTa** performed best for the **Recommended IND** classification task.
- The **Ensemble Model** improved overall accuracy and F1 score.

## Installation & Requirements
To run the project, install the required dependencies:

```bash
pip install transformers torch scikit-learn pandas numpy matplotlib seaborn
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   ```
2. Navigate to the project folder:
   ```bash
   cd your-repo
   ```
3. Run the training script:
   ```bash
   python train.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Future Improvements
- Further fine-tuning of models with domain-specific data.
- Integration of additional linguistic features.
- Optimizing computational efficiency for real-time sentiment analysis.

## Contributors
- **Sulaiman Muhammad**
- **Sai Shreesh Josyula**
- **Anusha Sekar**
- **Aryan Sharma**

## References
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [ALBERT: A Lite BERT for Self-Supervised Learning](https://arxiv.org/abs/1910.01108)
- [DistilBERT: A Distilled Version of BERT](https://arxiv.org/abs/1910.01108)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

## License
This project is licensed under the MIT License.
