import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import shap

from google.colab import files, drive
import io

# بارگذاری فایل
uploaded = files.upload()

# چاپ نام فایل‌های بارگذاری شده
print(uploaded.keys())

# خواندن داده‌ها از فایل CSV
data = pd.read_csv(io.BytesIO(uploaded['bnpl_credit_data_final.csv']))



# Preprocess the data
data.fillna(0, inplace=True)
data['Age Condition'] = np.where(data['Age'] < 18, 0, 1)
data['Credit_Condition'] = np.where(data['Credit Score'] > 519, 1, 0)

# Define purchase columns
purchase_freq_cols = ['Monthly Purchase Frequency 1', 'Monthly Purchase Frequency 2',
                       'Monthly Purchase Frequency 3', 'Monthly Purchase Frequency 4',
                       'Monthly Purchase Frequency 5', 'Monthly Purchase Frequency 6']
purchase_amount_cols = ['Monthly Purchase Amount 1', 'Monthly Purchase Amount 2',
                         'Monthly Purchase Amount 3', 'Monthly Purchase Amount 4',
                         'Monthly Purchase Amount 5', 'Monthly Purchase Amount 6']

data['Total_Purchase_Frequency'] = data[purchase_freq_cols].sum(axis=1)
data['Total_Purchase_Amount'] = data[purchase_amount_cols].sum(axis=1)
data['Repeat Usage'] = data['Repeat Usage'].map({'Yes': 1, 'No': 0})

# Create a new column for credit amount and repayment period based on conditions
def determine_credit(row):
    if row['Credit_Condition'] == 0:
        return 0, 0  # No credit
    if row['Payment Status'] == 'No':
        if row['Total_Purchase_Amount'] > 310000001:
            return 10000000, 1  # 10M for 1 month
        elif row['Total_Purchase_Amount'] > 150000001:
            return 5000000, 1  # 5M for 1 month
    else:
        if row['Total_Purchase_Frequency'] > 79 and row['Total_Purchase_Amount'] > 220000000:
            return 10000000, 3  # 10M for 3 months
        elif row['Total_Purchase_Frequency'] > 79 and row['Total_Purchase_Amount'] < 220000001:
            return 10000000, 1  # 10M for 1 month
        elif row['Total_Purchase_Frequency'] < 80 and row['Total_Purchase_Amount'] > 110000000:
            return 5000000, 3  # 5M for 3 months
        elif row['Total_Purchase_Frequency'] < 80 and row['Total_Purchase_Amount'] < 1100000001:
            return 5000000, 1  # 5M for 1 month
        elif row['Total_Purchase_Frequency'] < 41 and row['Total_Purchase_Amount'] < 80000001:
            return 2000000, 1  # 2M for 1 month
    return 0, 0  # Default no credit

data[['Credit Amount', 'Repayment Period']] = data.apply(determine_credit, axis=1, result_type='expand')

# Define target variable
data['Target'] = np.where(data['Credit_Condition'] & (data['Total_Purchase_Amount'] > 10), 1, 0)

# Prepare features and target
features = data[['Age', 'Credit Score', 'Total_Purchase_Frequency', 'Total_Purchase_Amount', 'Age Condition', 'Rating', 'Repeat Usage']]
target = data['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=64)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define RNN (Recurrent Neural Network)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, nonlinearity='tanh')

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device) # Hidden state

        # Forward propagate RNN
        out, _ = self.rnn(x.unsqueeze(1), h0)  # x.unsqueeze(1) to add sequence dimension

        # Get the last time step output
        out = out[:, -1, :]  # Only take the last output of the sequence

        # Pass through the fully connected layer
        out = self.fc(out)
        return out

# Model parameters
input_size = 7  # Number of features
hidden_size = 128  # Hidden state size
num_layers = 4  # Number of layers
output_size = 1  # Number of classes (for binary classification)
dropout = 0.2  # Dropout value

# Create the model
model = RNNModel(input_size, hidden_size, num_layers, output_size, dropout)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.009)
criterion = nn.BCEWithLogitsLoss()  # For binary classification

train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        predictions = torch.sigmoid(outputs).round()
        total_correct += (predictions == y_batch).sum().item()
        total_samples += y_batch.size(0)

    train_loss = total_loss / total_samples
    train_accuracy = total_correct / total_samples

    # Initialize validation variables
    val_total_loss = 0
    val_total_correct = 0
    val_total_samples = 0
    all_predictions = []
    all_y_test = []

    # Validation
    model.eval()
    with torch.no_grad():
        for X_val_batch, y_val_batch in test_loader:
            val_outputs = model(X_val_batch)
            val_loss = criterion(val_outputs, y_val_batch)
            val_total_loss += val_loss.item() * X_val_batch.size(0)

            val_predictions = torch.sigmoid(val_outputs).round()
            val_total_correct += (val_predictions == y_val_batch).sum().item()
            val_total_samples += y_val_batch.size(0)

            # Store predictions and true labels
            all_predictions.extend(val_predictions.numpy())
            all_y_test.extend(y_val_batch.numpy())

    val_loss = val_total_loss / val_total_samples
    val_accuracy = val_total_correct / val_total_samples

    all_y_test = np.array(all_y_test).flatten()
    all_predictions = np.array(all_predictions).flatten()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Save metrics
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Calculate metrics
accuracy = accuracy_score(all_y_test, all_predictions)
recall = recall_score(all_y_test, all_predictions)
precision = precision_score(all_y_test, all_predictions)
f1 = f1_score(all_y_test, all_predictions)
conf_matrix = confusion_matrix(all_y_test, all_predictions)
roc_auc = roc_auc_score(all_y_test, all_predictions)

# Print metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print('Confusion Matrix:\n', conf_matrix)

# Save the model
torch.save(model.state_dict(), '/content/drive/My Drive/Data/improved_transformer_model.pth')
# ایجاد DataFrame برای پیش‌بینی‌ها و واقعی‌ها
results_df = pd.DataFrame({'Actual': all_y_test, 'Predicted': all_predictions.flatten()})

# محاسبه امتیازهای مثبت و منفی
results_df['Positive'] = np.where(results_df['Actual'] == 1, results_df['Predicted'], 0)
results_df['Negative'] = np.where(results_df['Actual'] == 0, results_df['Predicted'], 0)

# محاسبه توزیع تجمعی
cum_pos = results_df['Positive'].cumsum() / results_df['Positive'].sum()
cum_neg = results_df['Negative'].cumsum() / results_df['Negative'].sum()

# محاسبه KS
ks_statistic = np.max(np.abs(cum_pos - cum_neg))
ks_threshold = np.argmax(np.abs(cum_pos - cum_neg))

print(f'KS Statistic: {ks_statistic:.4f} at threshold: {ks_threshold}')

# محاسبه KS Statistic
# محاسبه توزیع CDF برای مثبت‌ها و منفی‌ها
positive_scores = all_predictions[all_y_test == 1]
negative_scores = all_predictions[all_y_test == 0]

# ایجاد هیستوگرام برای محاسبه توزیع تجمعی
hist_pos, bin_edges_pos = np.histogram(positive_scores, bins=10, density=True)
hist_neg, bin_edges_neg = np.histogram(negative_scores, bins=10, density=True)

# محاسبه توزیع تجمعی
cdf_pos = np.cumsum(hist_pos) / np.sum(hist_pos)
cdf_neg = np.cumsum(hist_neg) / np.sum(hist_neg)

ks_statisticc = np.max(np.abs(cdf_pos - cdf_neg))
print(f'KS Statisticcc: {ks_statisticc:.4f}')



# رسم منحنی KS
plt.figure(figsize=(10, 6))
plt.plot(cum_pos, label='Cumulative Positive', color='blue')
plt.plot(cum_neg, label='Cumulative Negative', color='orange')
plt.axhline(y=ks_statistic, color='red', linestyle='--', label='KS Statistic')
plt.title('KS Statistic Curve_RNN', fontsize=16)
plt.xlabel('Threshold', fontsize=14)
plt.ylabel('Cumulative Distribution', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()



# Compute and plot confusion matrix
conf_matrix = confusion_matrix(all_y_test, all_predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix_RNN')
plt.show()

# Compute Precision-Recall curve
precision_vals, recall_vals, thresholds_pr = precision_recall_curve(all_y_test, all_predictions, pos_label=1)

# Define threshold range for interpolation
threshold_range = np.linspace(0, 1, 100)

# Interpolate precision and recall values
precision_interp = np.interp(threshold_range, np.append(thresholds_pr, 1), precision_vals, left=1, right=0)
recall_interp = np.interp(threshold_range, np.append(thresholds_pr, 1), recall_vals, left=0, right=0)

# Plot Precision-Recall Curve
plt.figure(figsize=(10, 6))
plt.plot(recall_interp, precision_interp, marker='.', label='Precision-Recall Curve', color='blue', lw=2)
plt.fill_between(recall_interp, precision_interp, alpha=0.1, color='blue')
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve (Detailed)_RNN', fontsize=16)
plt.xlim([-0.02, 1.015])
plt.ylim([-0.02, 1.015])
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(fontsize=12)
plt.show()

# Compute ROC curve
fpr, tpr, thresholds_roc = roc_curve(all_y_test, all_predictions)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC Curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)  # Diagonal line
plt.xlim([-0.02, 1.00])
plt.ylim([-0.02, 1.07])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve (Detailed)_RNN', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(fontsize=12)
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training & Validation Accuracy_RNN', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training & Validation Loss_RNN', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()


# Create a DataFrame for metrics
metrics_df = pd.DataFrame({
    'Accuracy': train_accuracies,
    'Val_Accuracy': val_accuracies
})

# Scatter Plot of Training and Validation Accuracy
plt.figure(figsize=(10, 8))
sns.scatterplot(data=metrics_df, x='Accuracy', y='Val_Accuracy')
plt.title('Scatter Plot of Training and Validation Accuracy_RNN')
plt.xlabel('Training Accuracy')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.show()



# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Payment Status', y='Total_Purchase_Amount', data=data)
plt.xlabel('Payment Status')
plt.ylabel('Total Purchase Amount')
plt.title('Box Plot of Total Purchase Amount by Payment Status_RNN')
plt.show()







# انتخاب ویژگی‌ها
features = data[['Age', 'Credit Score', 'Total_Purchase_Frequency',
                 'Total_Purchase_Amount', 'Age Condition', 'Rating', 'Repeat Usage']]

# نام ستون هدف
target_column_name = 'Target'  # نام ستون هدف را به درستی تنظیم کنید

target = data[target_column_name]  # ستون هدف

# نرمال‌سازی داده‌ها
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# تقسیم داده‌ها به مجموعه‌های آموزش و آزمون
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)

# تبدیل داده‌ها به تنسور
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# تعریف مدل
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(X_train_tensor.shape[1], 1)  # تعداد ویژگی‌ها

    def forward(self, x):
        return self.layer(x)

# ایجاد مدل
model = MyModel()

# تعریف تابع پیش‌بینی
def model_predict(x):
    x_tensor = torch.tensor(x, dtype=torch.float32)  # تبدیل به Tensor
    if x_tensor.ndim == 1:
        x_tensor = x_tensor.unsqueeze(0)  # اضافه کردن بعد دسته اگر ورودی یک بعدی باشد
    with torch.no_grad():
        return model(x_tensor).detach().numpy()

# SHAP Explainer
explainer_shap = shap.Explainer(model_predict, X_train_tensor.numpy())

# محاسبه مقادیر SHAP
shap_values = explainer_shap(X_test_tensor.numpy())

# نام ویژگی‌ها
feature_names = features.columns.tolist()

# رسم خلاصه SHAP با نام‌های ویژگی‌ها
shap.summary_plot(shap_values, X_test_tensor.numpy(), feature_names=feature_names)





# Line chart
plt.figure(figsize=(10, 6))
plt.plot(data['Customer ID'], data['Total_Purchase_Amount'], label='Total Purchase Amount')
plt.xlabel('Customer ID')
plt.ylabel('Total Purchase Amount')
plt.title('Line Chart of Total Purchase Amount_RNN')
plt.legend()
plt.show()

# Scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(data['Total_Purchase_Frequency'], data['Total_Purchase_Amount'], c=data['Target'], cmap='Set1_r', alpha=0.7)
plt.xlabel('Total Purchase Frequency')
plt.ylabel('Total Purchase Amount')
plt.title('Scatter Plot of Purchase Frequency vs. Amount_RNN')
plt.colorbar(label='Credit Granted')
plt.show()

# Histogram
plt.figure(figsize=(10, 8))
plt.hist(data['Total_Purchase_Amount'], bins=30, edgecolor='k')
plt.xlabel('Total Purchase Amount')
plt.ylabel('Frequency')
plt.title('Histogram of Total Purchase Amount_RNN')
plt.show()



# Plotting the scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(data=metrics_df, x='Accuracy', y='Val_Accuracy')
plt.title('Scatter Plot of Training and Validation Accuracy_RNN')
plt.xlabel('Training Accuracy')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.show()


# Age distribution chart
plt.figure(figsize=(10, 8))
sns.histplot(data['Age'], bins=30, kde=True)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution Chart_RNN')
plt.show()

# Credit score distribution chart
plt.figure(figsize=(10, 8))
sns.histplot(data['Credit Score'], bins=30, kde=True)
plt.xlabel('Credit Score')
plt.ylabel('Frequency')
plt.title('Credit Score Distribution Chart_RNN')
plt.show()



# Pie chart of credit prediction status
credit_status_counts = data['Target'].value_counts()
sizes = np.array([10, 8])  # Your sizes data here
labels = ['Credit Granted', 'Credit Not Granted']  # Your labels
colors = ['#66b3ff', '#ff9999']  # Example colors
explode = (0.1, 0)  # only "explode" the 1st slice
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', explode=explode, startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Credit Prediction Status_RNN')
plt.show()

# Save the results to a new CSV file
data.to_csv('customer_credit_offers_RNN.csv', index=False)
files.download('customer_credit_offers_RNN.csv')
