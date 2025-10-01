📊 Sales Prediction using Linear Regression

This project predicts product sales based on advertising spend on TV, Radio, and Newspaper using a Linear Regression model.
It demonstrates how businesses can leverage data to make smarter marketing budget allocations and improve ROI.

🚀 Features

📂 Loads and preprocesses Advertising dataset

🔎 Exploratory Data Analysis (EDA) with correlation insights

🤖 Trains Linear Regression model with scikit-learn

📈 Evaluates performance using MSE and R² Score

🔮 Custom function for predicting sales based on user input

📂 Project Structure
├── data/                # Dataset (Advertising.csv)
├── notebooks/           # Jupyter notebook with analysis & model training
├── src/                 # Python scripts (model training, prediction)
├── models/              # Saved trained model
├── requirements.txt     # Dependencies
└── README.md            # Documentation

⚙️ Installation

Clone the repo and install dependencies:

git clone https://github.com/your-username/sales-prediction-linear-regression.git
cd sales-prediction-linear-regression
pip install -r requirements.txt

📖 Usage
1. Train & Evaluate Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


✅ Results:

Mean Squared Error (MSE): 3.17

R² Score: 0.89

2. Predict Sales for Custom Inputs
def predict_sales(tv, radio, newspaper):
    input_data = pd.DataFrame({'TV':[tv], 'radio':[radio], 'newspaper':[newspaper]})
    predicted_sales = model.predict(input_data)
    return predicted_sales

# Example
print(predict_sales(450, 22, 100))  # Output → ~27.54

📊 Visualization (Example)

You can plot Actual vs Predicted Sales for better insights:

plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

📌 Future Enhancements

🚀 Deploy model using Streamlit/Flask for web-based predictions

🔄 Add more regression models for comparison (Ridge, Lasso, Random Forest)

📊 Build interactive dashboard for visualization

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

📜 License

This project is licensed under the MIT License.
