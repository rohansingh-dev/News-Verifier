import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Define ChronNet
class ChronNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ChronNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return self.softmax(out)

# Custom Dataset
class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx].toarray(), dtype=torch.float32), torch.tensor(int(self.y[idx]), dtype=torch.long)

def train_model():
    """Train the ChronNet model."""
    news = pd.read_csv('scraped.csv')
    news['label'] = news['label'].fillna(-1)
    valid_labels = [0, 1]
    news = news[news['label'].isin(valid_labels)]

    if news.empty:
        raise ValueError("The dataset is empty after filtering for valid labels.")

    text = news['text'].astype('U').values
    label = news['label'].astype(int).values

    text_train, text_test, label_train, label_test = train_test_split(text, label, test_size=0.25, random_state=5)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, lowercase=True)
    X_train = vectorizer.fit_transform(text_train)
    X_test = vectorizer.transform(text_test)
    pickle.dump(vectorizer, open('TfidfVectorizer-ChronNet.sav', 'wb'))

    train_dataset = NewsDataset(X_train, label_train)
    test_dataset = NewsDataset(X_test, label_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = 2
    model = ChronNet(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 2
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss:.4f}")

    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(y_batch.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    torch.save(model.state_dict(), 'ChronNetModel.pth')
    print("ChronNet model saved successfully.")

if __name__ == "__main__":
    print("Training the ChronNet model...")
    train_model()
    print("Model training completed.")