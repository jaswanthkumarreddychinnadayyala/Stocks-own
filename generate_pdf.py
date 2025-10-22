from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Stock Price Prediction Project Summary', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 11)
        self.cell(0, 10, title, 0, 1)
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 10, body)
        self.ln()

pdf = PDF()
pdf.add_page()

# Problem Statement
pdf.chapter_title('Problem Statement')
pdf.chapter_body(
    'The stock market is highly volatile, making it difficult for investors to make consistent and profitable decisions. Fluctuations in stock prices occur due to numerous factors, including economic conditions, market sentiment, and geopolitical events. Predicting future stock prices can help investors reduce risks and optimize investment strategies.'
)

# Project Objectives
pdf.chapter_title('Project Objectives')
pdf.chapter_body(
    '- Collect and preprocess historical stock price data for Tata Steel from reliable sources.\n'
    '- Engineer technical features such as moving averages, relative strength index (RSI), moving average convergence divergence (MACD), Bollinger Bands, and others to improve predictive accuracy.\n'
    '- Train a machine learning regression model (XGBoost) using time-series cross-validation and hyperparameter tuning.\n'
    '- Develop a user-friendly interactive web app using Streamlit that allows dynamic input of stock indicators and provides real-time price predictions.\n'
    '- Visualize historical stock prices using interactive candlestick charts and display prediction trends with dynamic charts.\n'
    '- Enable users to save and review prediction history during the app session.'
)

pdf.output('Stock_Price_Prediction_Project_Summary.pdf')
print("PDF generated successfully.")
