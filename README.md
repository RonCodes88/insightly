# insightly
authors: Ronald Li, Adarsh Mallya
## description of question and research topic:

Today, with the rise of e-commerce, there is a vast number of products available. Customers often struggle to quickly understand the true quality of a product because numeric ratings can be inconsistent and reviews are unstructured. This makes it difficult to identify products that genuinely meet their needs. Machine learning can help by automatically analyzing review text to predict sentiment and assign meaningful ratings. Such insights allow shoppers to make informed decisions faster and provide sellers with actionable feedback to improve their products.

## project outline / plan

**insightly** is an interactive ML-powered web app that analyzes e-commerce product reviews. Users can type or paste a product name or review and instantly see the **predicted sentiment** and a **rating assigned based on it**. The system uses a **PyTorch-based sentiment analysis model** to convert textual feedback into actionable insights for both consumers and sellers.

---

### main features

- **Real-Time Sentiment Prediction:** Analyze product reviews and get instant sentiment classification (**Positive / Negative / Neutral**).  
- **Rating Assignment:** Automatically assigns a rating based on the predicted sentiment.  
- **Optional Feature Insights:** Highlight sentiment for specific aspects of a product (e.g., battery, shipping, quality).  
- **Visual Feedback:** Color-coded sentiment, confidence bars, or charts for better user understanding.

---

### frontend (next.js)

- Input box for **review text** or **product name**.  
- “Analyze” button triggers an **API call** to the backend.  
- Displays **predicted sentiment**, assigned **rating**, and optionally **confidence score**.  
- Built with **Tailwind CSS** for a clean, modern UI.

---

### backend (FastAPI + PyTorch)

- Receives text input from the frontend.  
- Preprocesses the input (tokenization, normalization).  
- Runs the **sentiment analysis model** to classify sentiment.  
- Returns **predicted sentiment**, **confidence score**, and **assigned rating** to the frontend.

---

### visualizations
- **Color-Coded Sentiment:** Green = Positive, Red = Negative, Gray = Neutral.  
- **Confidence Bars:** Optional display of model confidence per prediction.  
- **Feature-Level Insights:** Optionally highlight sentiment for key product features.
  
## data collection plan
https://www.kaggle.com/code/aaroha33/e-commerce-sentiment-analysis/notebook

The first dataset is a publicly available E-commerce - Sentiment Analysis dataset from Kaggle. The dataset contains over 34,000 reviews for Amazon products and has 8 columns including product name, brand, categories, review text, and sentiment labels. For this project, we focus on the reviews.text, reviews.title, and sentiment columns to train our ML model. Both training and test splits are provided, and we clean and preprocess the data by removing duplicates, handling missing values, and normalizing the text. 

## model plans

The second model we will use is a standard Recurrent Neural Network (RNN) to analyze e-commerce product reviews for sentiment classification. RNNs are designed to handle sequential data, making them suitable for processing text where word order matters. Each review will be tokenized, converted into sequences, and padded to a fixed length before feeding into the RNN. The model will output a prediction for each review as Positive or Negative, and optionally assign a numeric rating based on confidence scores. We will evaluate the model using accuracy, precision, recall, and F1-score to ensure reliable predictions on unseen reviews.

