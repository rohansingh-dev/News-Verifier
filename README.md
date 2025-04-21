# News Verifier

**News Verifier** is a machine learning-powered application designed to assess the authenticity of news articles. By analyzing textual content, it determines the likelihood of a news piece being genuine or fake, aiding users in discerning credible information.

## 🧠 Features

- **Fake News Detection**:Utilizes trained machine learning models to classify news articles as real or fake
- **User-Friendly Interface**:Provides a web-based platform for users to input news content and receive instant verification results
- **Model Training**:Includes scripts to train and save models for future use

## 🛠️ Tech Stack

- **Backend** Python, Flak
- **Frontend** HTML, CSS, JavaScrit
- **Machine Learning** Scikit-learn, TfidfVectorizr

## 📁 Project Structur



```plaintext
News-Verifier/
├── app.py                     # Main application script
├── model.py                   # Script to train and save the ML model
├── utils.py                   # Utility functions
├── config.py                  # Configuration settings
├── templates/                 # HTML templates
├── static/                    # Static files (CSS, JS, images)
├── requirements.txt           # Python dependencies
├── TfidfVectorizer.sav        # Saved TfidfVectorizer model
├── TfidfVectorizer-ChronNet.sav # Alternative saved model
├── Fake.zip                   # Dataset containing fake news articles
├── True.zip                   # Dataset containing true news articles
├── scraped.zip                # Additional scraped data
└── README.md                  # Project documentation
```



## 🚀 Getting Started

### Prerequisites
- Python3.x
- pip (Python package instaler)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/rohansingh-dev/News-Verifier.git
   ``



2. **Navigate to the project directory**:

   ```bash
   cd News-Verifier
   ``



3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ``



4. **Train and save the model**:

   ```bash
   python model.py
   ``



5. **Run the application**:

   ```bash
   python app.py
   ``



   Access the application by navigating to `http://localhost:5000` in your web browser.

## 🤝 Contributng

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fxes.

## 📄 Licnse

This project is licensed under the [MIT License](LICNSE).

