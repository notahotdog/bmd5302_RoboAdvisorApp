# Financial Robot Advisor

## Description
Financial Robot Advisor is an interactive web application developed for the BMD5302 Financial Modeling course. This application allows users to determine their investor profile, calculate their risk aversion level, and receive portfolio recommendations optimized based on their profile.

## Key Features
- **Risk Assessment Questionnaire**: Determines the investor profile and risk aversion level
- **Optimized Portfolio**: Recommends optimal asset allocation based on risk profile
- **Efficient Frontier Analysis**: Visualizes the efficient frontier and allows comparison of different optimal portfolios
- **Future Projections**: Simulation of future performance based on Monte Carlo methods
- **Interactive Interface**: User-friendly interface developed with Streamlit

## Data Structure
The application uses historical NAV data for 15 investment funds from various asset classes and regions:

### Cash/Money Market
- Fullerton SGD Cash Fund
- Fidelity US Dollar Cash A

### Fixed Income
- GS FUNDS III - US DOLLAR CREDIT P CAP USD
- JPMORGAN FUNDS - US AGGREGATE BOND A (ACC) SGD-H
- FIDELITY EUROPEAN HIGH YIELD A-MDIST-SGD
- FIDELITY US HIGH YIELD A-MDIST-SGD

### Mixed/Balanced
- FTIF - FRANKLIN INCOME A MDIS SGD-H1
- ALLIANZ INCOME AND GROWTH CL AM DIS H2-SGD

### Equity
- ALLIANZ ORIENTAL INCOME ET ACC SGD
- NIKKO AM SINGAPORE DIVIDEND EQUITY ACC USD
- CT UK EQUITY INCOME CLASS 1 ACC GBP
- EASTSPRING INVESTMENTS UNIT TRUSTS - DRAGON PEACOCK A SGD
- JPMORGAN FUNDS - US SELECT EQUITY PLUS A (ACC) SGD
- JPMORGAN FUNDS - US TECHNOLOGY A (ACC) SGD

### Other
- East Spring Investment Unit Trust

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Installing Dependencies
```bash
pip install streamlit pandas numpy scipy plotly matplotlib
```

### File Structure
The project expects to find Excel/CSV files in the `funds_ref` folder. Each file should contain historical NAV data for a specific fund.

### Data File Format
Each data file should contain at minimum:
- NAV Date column: Dates of the NAV values
- NAV Price column: Historical NAV values in the fund's currency

The application can detect these columns automatically based on common naming patterns.

## Usage

### Running the Application
```bash
streamlit run app.py
```

### Application Workflow
1. **Questionnaire**: Answer questions about your investment goals, risk tolerance, and time horizon
2. **Risk Aversion Score**: Receive a personalized risk aversion score and investor profile
3. **Portfolio Optimization**: View recommended fund allocation based on your profile
4. **Portfolio Analysis**: Explore historical performance, efficient frontier, and future projections

## Methodology

### Modern Portfolio Theory
The application implements Markowitz's Modern Portfolio Theory to construct the efficient frontier, representing portfolios that offer the best return for a given level of risk.

### Risk Aversion Model
The investor's utility function is modeled as:
```
U = r - σ² × (A/2)
```
Where:
- r is the expected return
- σ is the standard deviation (risk)
- A is the risk aversion coefficient

### Data Processing Features
- Support for multiple file formats (Excel, CSV)
- Automatic detection of NAV date and price columns
- Handling of duplicate dates
- Conversion of string values to numeric format
- Robust error handling with fallback mechanisms

### Optimization Approach
The optimal portfolio is determined by maximizing the investor's utility function, taking into account their risk aversion level calculated from questionnaire responses.

## Key Improvements from Previous Version
- **Expanded Investment Universe**: Increased from 10 stocks to 15 diverse funds
- **Enhanced Data Handling**: Added support for Excel files and improved detection of data columns
- **Improved Robustness**: Better handling of various edge cases in data processing
- **International Diversification**: Now includes funds covering multiple regions and currencies
- **Asset Class Diversification**: Added cash, bond, and mixed-asset funds for more complete portfolios

## Limitations
- Past performance does not guarantee future results
- The model assumes normal distribution of returns
- Transaction costs and tax implications are not considered
- Currency risk may not be fully captured in the model

## Technologies Used
- Python
- Streamlit for web interface
- Pandas for data manipulation
- NumPy and SciPy for optimization calculations
- Plotly for interactive visualizations

## Author
This application was developed as part of the BMD5302 Financial Modeling course, AY 2024/25 Semester 2.

## License
This project is for educational purposes only. All rights reserved.