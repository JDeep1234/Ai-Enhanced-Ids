import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset (replace with your dataset path)
data_columns = ["Fwd Seg Size Min", "Init Bwd Win Byts", "Init Fwd Win Byts", "Fwd Seg Size Min", "Fwd Seg Size Avg", "Label", "Timestamp"]
data_dtypes = {"Fwd Pkt Len Mean": float, "Fwd Seg Size Avg": float, "Init Fwd Win Byts": int, "Init Bwd Win Byts": int, "Fwd Seg Size Min": int, "Label": str}
date_col = ["Timestamp"]
raw_data = pd.read_csv("path", usecols=data_columns, dtype=data_dtypes, parse_dates=date_col, index_col=None)
sorted_data = raw_data.sort_values("Timestamp")
processed_data = sorted_data.drop(columns=["Timestamp"])

X = processed_data.drop(columns=["Label"])
y = processed_data["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier with class_weight
clf_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test).astype(str)

# Train the Logistic Regression model
clf_lr = LogisticRegression(max_iter=1000, class_weight='balanced')
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test).astype(str)

# Train the K-Nearest Neighbors model
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test).astype(str)

y_pred_rf_bin = (y_pred_rf == 'threat').astype(int)
y_pred_lr_bin = (y_pred_lr == 'threat').astype(int)
y_pred_knn_bin = (y_pred_knn == 'threat').astype(int)
combined_predictions = (y_pred_rf_bin + y_pred_lr_bin + y_pred_knn_bin) >= 2

# Evaluate the RandomForestClassifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
confusion_rf = confusion_matrix(y_test, y_pred_rf)

# Evaluate the Logistic Regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr, average='weighted')
recall_lr = recall_score(y_test, y_pred_lr, average='weighted')
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
confusion_lr = confusion_matrix(y_test, y_pred_lr)

# Evaluate the K-Nearest Neighbors model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average='weighted')
recall_knn = recall_score(y_test, y_pred_knn, average='weighted')
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
confusion_knn = confusion_matrix(y_test, y_pred_knn)

# Function to display statements
def display_statements(statements):
    for statement in statements:
        text_box.insert(tk.END, statement + '\n')

# Function to display confusion matrix plot
def display_confusion_matrix(confusion_matrix, class_labels, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    canvas = FigureCanvasTkAgg(plt.gcf(), master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Function to display user feedback
def display_user_feedback(feedback):
    messagebox.showinfo("User Feedback", feedback)

# GUI setup
window = tk.Tk()
window.title("INTRUSION DETECTION SYSTEM ")

# Text box to display statements
text_box = scrolledtext.ScrolledText(window, width=80, height=20)
text_box.pack(padx=10, pady=10)

# Button to trigger the display of statements and graphs
display_button = tk.Button(window, text="Please Click To Check For Threats", command=lambda: display_data())
display_button.pack(pady=10)

# Function to display ML results
# Function to display ML results
def display_data():
    # Display statements
    display_statements(["Random Forest Classifier:",
                        f"Accuracy: {accuracy_rf}",
                        f"Precision: {precision_rf}",
                        f"Recall: {recall_rf}",
                        f"F1 Score: {f1_rf}",
                        "\nLogistic Regression:",
                        f"Accuracy: {accuracy_lr}",
                        f"Precision: {precision_lr}",
                        f"Recall: {recall_lr}",
                        f"F1 Score: {f1_lr}",
                        "\nK-Nearest Neighbors:",
                        f"Accuracy: {accuracy_knn}",
                        f"Precision: {precision_knn}",
                        f"Recall: {recall_knn}",
                        f"F1 Score: {f1_knn}"])

    # Display confusion matrix plots
    display_confusion_matrix(confusion_rf, clf_rf.classes_, "Random Forest Classifier - Confusion Matrix")

    # Create separate FigureCanvasTkAgg instances for Logistic Regression and K-Nearest Neighbors
    canvas_lr = FigureCanvasTkAgg(plt.figure(figsize=(8, 6)), master=window)
    canvas_knn = FigureCanvasTkAgg(plt.figure(figsize=(8, 6)), master=window)

    # Display confusion matrix plot for Logistic Regression
    plt.figure(plt.gcf().number)  # Set the current figure to the one created for Logistic Regression
    sns.heatmap(confusion_lr, annot=True, fmt="d", cmap="Blues", xticklabels=clf_lr.classes_, yticklabels=clf_lr.classes_)
    plt.title("Logistic Regression - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    canvas_lr.draw()
    canvas_lr.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Display confusion matrix plot for K-Nearest Neighbors
    plt.figure(plt.gcf().number)  # Set the current figure to the one created for K-Nearest Neighbors
    sns.heatmap(confusion_knn, annot=True, fmt="d", cmap="Blues", xticklabels=clf_knn.classes_, yticklabels=clf_knn.classes_)
    plt.title("K-Nearest Neighbors - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    canvas_knn.draw()
    canvas_knn.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Display user feedback
    display_user_feedback("Threat detected!!!! Danger Please check the logs " if any(combined_predictions) else "The system is safe. No threats and intrusion detected. :)")

# Run the GUI
window.mainloop()
