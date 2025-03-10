# IMDb Review Sentiment Analysis 🎬📊  

![Python](https://img.shields.io/badge/python-3.9%2B-blue)  
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange)  
![Natural Language Processing](https://img.shields.io/badge/NLP-✔️-green)  
![License](https://img.shields.io/github/license/n7kadda/imdb_review)  

A deep learning-based sentiment analysis model to classify IMDb movie reviews as **positive** or **negative** using Natural Language Processing (NLP).  

## 🚀 About the Project  

This project leverages **deep learning** techniques using **LSTM (Long Short-Term Memory) networks** to classify IMDb movie reviews into positive or negative sentiments. The model is trained on textual data and applies word embeddings for efficient representation.  

## 📂 Dataset  

The dataset used in this project is the **IMDb Large Movie Review Dataset**, containing 50,000 reviews split evenly between positive and negative sentiments. It is loaded using **TensorFlow's `tf.keras.datasets` module**.  

## 🛠 Getting Started  

Follow these steps to set up the project on your local machine.  

### ✅ Prerequisites  

Ensure you have Python 3.9+ installed. You'll also need the following dependencies:  

- TensorFlow 2.x  
- Keras  
- NumPy  
- Pandas  
- Matplotlib  

### 💾 Installation  

1. **Clone the repository**  
   ```sh
   git clone https://github.com/n7kadda/imdb_review.git
   cd imdb_review
   ```  

2. **Install the required dependencies**  
   ```sh
   pip install -r requirements.txt
   ```  


3. **Explore the Jupyter Notebook**  
   You can also check out `bidirectionalrnn.ipynb` for a detailed walkthrough of the training and evaluation process.  


## 🧠 Explanation of the Bidirectional RNN Used in IMDb Review Sentiment Analysis
In this project, a Bidirectional Recurrent Neural Network (BiRNN) with Long Short-Term Memory (LSTM) units is used to improve the performance of sentiment classification.

**🔹 What is a Bidirectional RNN?**
A Bidirectional RNN (BiRNN) is an extension of a standard RNN that processes input sequences in both forward and backward directions. This helps capture contextual dependencies from both past and future words in a sentence, making it particularly useful for Natural Language Processing (NLP) tasks like sentiment analysis.

**🔹 Why Use a BiRNN for Sentiment Analysis?**
Standard RNNs process text sequentially (left to right), but in many NLP tasks, the meaning of a word depends on both its previous and next words.
BiRNNs capture both past and future dependencies, making them more effective for understanding the full context of a sentence.
In sentiment analysis, a single word like "not" can completely change the meaning of a review (e.g., "The movie was not good" → Negative). A BiLSTM ensures the model learns both forward and backward relationships.
**🔹 BiLSTM Architecture Used in the Model**
The core model consists of:

Embedding Layer – Converts words into dense vector representations.
Bidirectional LSTM Layer – Processes the text in both directions using two LSTM units (one for forward, one for backward).
Dropout Layer – Reduces overfitting by randomly deactivating neurons.
Dense Layer – Fully connected layer for sentiment classification.
Sigmoid Activation – Outputs probabilities for binary classification (positive/negative sentiment).

## 📊 Results  

- **Accuracy**: Achieved ~88% accuracy on test data  
- **Loss Function**: Binary Cross-Entropy  
- **Optimizer**: Adam  

## 🤝 Contributing  

Contributions are always welcome!  

1. Fork the repository  
2. Create a new branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m "Add some feature"`)  
4. Push to the branch (`git push origin feature/YourFeature`)  
5. Open a Pull Request  

## 📜 License  

Distributed under the **MIT License**. See `LICENSE` for more information.  

Project Link: [https://github.com/n7kadda/imdb_review](https://github.com/n7kadda/imdb_review)  

---  

This README provides an engaging overview of the project, including instructions on setup, usage, and contributions. Let me know if you'd like any modifications! 🚀
