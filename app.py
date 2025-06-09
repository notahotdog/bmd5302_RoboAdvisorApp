import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import random
import os
import glob

debug_mode = False

# Streamlit page configuration
st.set_page_config(
    page_title="Financial Robot Advisor - Fund Selection",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title('Financial Robot Advisor - BMD5302')
st.markdown('---')

# Selection of funds based on the new list
funds = {
    'Fullerton_sgd_cash_fund': 'Fullerton SGD Cash Fund',
    'East_Spring_Investment_unit_trust': 'East Spring Investment Unit Trust',
    'Fidelity_US_Dollar_Cash_A': 'Fidelity US Dollar Cash A',
    'GS_FUNDS_III_US_DOLLAR_CREDIT_P_CAP_USD': 'GS FUNDS III - US DOLLAR CREDIT P CAP USD',
    'JPMORGAN_FUNDS_US_AGGREGATE_BOND_A_ACC_SGD_H': 'JPMORGAN FUNDS - US AGGREGATE BOND A (ACC) SGD-H',
    'FIDELITY_EUROPEAN_HIGH_YIELD_A_MDIST_SGD': 'FIDELITY EUROPEAN HIGH YIELD A-MDIST-SGD',
    'FIDELITY_US_HIGH_YIELD_A_MDIST_SGD': 'FIDELITY US HIGH YIELD A-MDIST-SGD',
    'FTIF_FRANKLIN_INCOME_A_MDIS_SGD_H1': 'FTIF - FRANKLIN INCOME A MDIS SGD-H1',
    'ALLIANZ_INCOME_AND_GROWTH_CL_AM_DIS_H2_SGD': 'ALLIANZ INCOME AND GROWTH CL AM DIS H2-SGD',
    'ALLIANZ_ORIENTAL_INCOME_ET_ACC_SGD': 'ALLIANZ ORIENTAL INCOME ET ACC SGD',
    'NIKKO_AM_SINGAPORE_DIVIDEND_EQUITY_ACC_USD': 'NIKKO AM SINGAPORE DIVIDEND EQUITY ACC USD',
    'CT_UK_EQUITY_INCOME_CLASS_1_ACC_GBP': 'CT UK EQUITY INCOME CLASS 1 ACC GBP',
    'EASTSPRING_INVESTMENTS_UNIT_TRUSTS_DRAGON_PEACOCK_A_SGD': 'EASTSPRING INVESTMENTS UNIT TRUSTS - DRAGON PEACOCK A SGD',
    'JPMORGAN_FUNDS_US_SELECT_EQUITY_PLUS_A_ACC_SGD': 'JPMORGAN FUNDS - US SELECT EQUITY PLUS A (ACC) SGD',
    'JPMORGAN_FUNDS_US_TECHNOLOGY_A_ACC_SGD': 'JPMORGAN FUNDS - US TECHNOLOGY A (ACC) SGD'
}

# Function to read fund data from local Excel files
@st.cache_data(ttl=3600)
def get_fund_data(fund_list):
    """Get data from local Excel files for funds"""
    # Specify the path to the fund data directory
    fund_data_dir = 'funds_ref'
    
    # Dictionary to store individual fund data frames
    fund_data_frames = {}
    
    # Track successful reads
    success_count = 0
    
    # Check if the directory exists
    if not os.path.exists(fund_data_dir):
        st.error(f"Directory not found: {fund_data_dir}")
        return create_backup_data(fund_list)
    
    for fund in fund_list:
        # Construct file path for this fund - try different possible extensions
        possible_extensions = ['.xlsx', '.xls', '.csv']
        file_path = None
        
        for ext in possible_extensions:
            temp_path = os.path.join(fund_data_dir, f"{fund}{ext}")
            if os.path.exists(temp_path):
                file_path = temp_path
                break
                
        # If no file found with any extension, try glob pattern
        if file_path is None:
            pattern = os.path.join(fund_data_dir, f"*{fund}*")
            matching_files = glob.glob(pattern)
            if matching_files:
                file_path = matching_files[0]
        
        try:
            # Check if file exists
            if file_path is None or not os.path.exists(file_path):
                st.warning(f"File not found for fund: {fund}")
                continue
                
            # Read Excel file - different approach than CSV
            if file_path.endswith('.csv'):
                fund_data = pd.read_csv(file_path)
            else:
                fund_data = pd.read_excel(file_path)
            
            # Check if data is valid
            if fund_data.empty:
                st.warning(f"No valid data in file for {fund}")
                continue
            
            # Try to identify column names based on common patterns
            date_col = None
            price_col = None
            
            for col in fund_data.columns:
                col_lower = str(col).lower()
                if 'date' in col_lower or 'nav date' in col_lower:
                    date_col = col
                elif 'price' in col_lower or 'nav price' in col_lower or 'nav' in col_lower:
                    price_col = col
            
            # If we couldn't identify columns, use positional assumptions based on example
            if date_col is None or price_col is None:
                if len(fund_data.columns) >= 4:  # Based on example format
                    date_col = fund_data.columns[-1]  # Last column is date
                    price_col = fund_data.columns[0]  # First column is price
            
            # Check if we found our columns
            if date_col is None or price_col is None:
                st.warning(f"Couldn't identify date and price columns for {fund}")
                continue
            
            # Ensure the price column is numeric
            try:
                # Convert price column to numeric, coercing errors to NaN
                fund_data[price_col] = pd.to_numeric(fund_data[price_col], errors='coerce')
                
                # Drop rows with NaN prices
                fund_data = fund_data.dropna(subset=[price_col])
                
                if fund_data.empty:
                    st.warning(f"No valid numeric data in price column for {fund}")
                    continue
            except Exception as e:
                st.warning(f"Error converting price to numeric for {fund}: {str(e)}")
                continue
                
            # Convert date column to datetime
            try:
                fund_data[date_col] = pd.to_datetime(fund_data[date_col])
            except:
                st.warning(f"Couldn't convert dates for {fund}")
                continue
            
            # Check for and handle duplicate dates
            if fund_data[date_col].duplicated().any():
                if debug_mode:
                    st.warning(f"Duplicate dates found in {fund}. Using most recent value.")
                # Sort by date (ascending) and keep the last occurrence of each date
                fund_data = fund_data.sort_values(date_col)
                fund_data = fund_data.drop_duplicates(subset=[date_col], keep='last')
                
            # Create a simple series with dates as index and price as values
            price_series = pd.Series(fund_data[price_col].values, index=fund_data[date_col])
            
            # Store in our dictionary
            fund_data_frames[fund] = price_series
            
            success_count += 1
            
        except Exception as e:
            st.warning(f"Error reading data for {fund}: {str(e)}")
    
    # Check if we have any data
    if len(fund_data_frames) == 0 or success_count == 0:
        st.warning("Failed to read any fund data from Excel files. Using backup data.")
        return create_backup_data(fund_list)
    
    try:
        # Create a DataFrame from the dictionary - this aligns on the dates automatically
        all_data = pd.DataFrame(fund_data_frames)
        
        # Sort by date
        all_data = all_data.sort_index()
        
        # Fill any missing values
        all_data = all_data.fillna(method='ffill').fillna(method='bfill')
        
        # Do a final check to ensure all data is numeric
        for col in all_data.columns:
            all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
        
        # Drop any rows that still have NaN after conversion
        all_data = all_data.dropna()
        
        if all_data.empty:
            st.warning("After data cleaning, no valid numeric data remains. Using backup data.")
            return create_backup_data(fund_list)
            
    except Exception as e:
        st.error(f"Error creating combined DataFrame: {str(e)}")
        # Fallback to generated data
        return create_backup_data(fund_list)
    
    return all_data

# Function to create backup data if Excel reading fails
def create_backup_data(fund_list):
    """Create synthetic fund data as fallback"""
    st.warning("Using generated data for demonstration purposes.")
    
    # Create backup data as fallback (random but realistic fund price movements)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)  # 3 years
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    backup_data = pd.DataFrame(index=date_range)
    
    # Generate semi-realistic fund data for each fund
    for fund in fund_list:
        # Different starting prices based on fund type
        if 'cash' in fund.lower() or 'bond' in fund.lower():
            # Cash and bond funds have lower volatility and price
            start_price = random.uniform(1.0, 5.0)
            annual_drift = random.uniform(0.01, 0.05)  # 1-5% annual return
            annual_volatility = random.uniform(0.01, 0.1)  # 1-10% volatility
        elif 'equity' in fund.lower() or 'stock' in fund.lower() or 'growth' in fund.lower():
            # Equity funds have higher volatility and potentially higher returns
            start_price = random.uniform(5.0, 20.0)
            annual_drift = random.uniform(0.04, 0.15)  # 4-15% annual return
            annual_volatility = random.uniform(0.1, 0.25)  # 10-25% volatility
        else:
            # Mixed or other funds
            start_price = random.uniform(1.0, 10.0)
            annual_drift = random.uniform(0.02, 0.1)  # 2-10% annual return
            annual_volatility = random.uniform(0.05, 0.2)  # 5-20% volatility
            
        daily_drift = annual_drift / 252
        daily_volatility = annual_volatility / np.sqrt(252)
        
        # Generate price series
        prices = [start_price]
        for i in range(1, len(date_range)):
            previous_price = prices[-1]
            change = np.random.normal(daily_drift, daily_volatility)
            new_price = previous_price * (1 + change)
            prices.append(max(0.5, new_price))  # Ensure price doesn't go too low
            
        backup_data[fund] = prices
        
    return backup_data

# Function to calculate returns and covariance matrix with error handling
def calculate_portfolio_stats(fund_data):
    try:
        # Calculate daily returns with error checking
        returns = fund_data.pct_change().dropna()
        
        # Handle empty returns dataframe
        if returns.empty:
            raise ValueError("Empty returns dataframe")
        
        # Calculate annual average returns (252 trading days per year)
        mean_returns = returns.mean() * 252
        
        # Calculate annualized covariance matrix
        cov_matrix = returns.cov() * 252
        
        return returns, mean_returns, cov_matrix
    except Exception as e:
        st.error(f"Error in portfolio statistics calculation: {e}")
        
        # Create dummy data for demonstration
        num_assets = len(fund_data.columns)
        dummy_returns = pd.DataFrame(np.random.normal(0.0005, 0.01, size=(100, num_assets)),
                                     columns=fund_data.columns)
        dummy_mean_returns = pd.Series([0.08, 0.10, 0.12, 0.07, 0.09, 0.11, 0.08, 0.10, 0.07, 0.09, 0.08, 0.10, 0.09, 0.11, 0.13][:num_assets],
                                       index=fund_data.columns)
        dummy_cov = pd.DataFrame(np.random.uniform(0.01, 0.3, size=(num_assets, num_assets)),
                                index=fund_data.columns, columns=fund_data.columns)
        
        # Make sure covariance matrix is symmetric and positive definite
        dummy_cov = (dummy_cov + dummy_cov.T) / 2
        for i in range(num_assets):
            dummy_cov.iloc[i, i] = random.uniform(0.04, 0.25)
            
        return dummy_returns, dummy_mean_returns, dummy_cov

# Portfolio optimization functions
def portfolio_performance(weights, mean_returns, cov_matrix):
    try:
        returns = np.sum(mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return std, returns
    except Exception as e:
        st.error(f"Error in portfolio performance calculation: {e}")
        return 0.15, 0.08  # Default values

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    try:
        p_std, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
        if p_std == 0:
            return 0  # Avoid division by zero
        return -(p_ret - risk_free_rate) / p_std
    except Exception as e:
        return 0  # Default value

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0.02, constraint_set=(0, 1)):
    try:
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix, risk_free_rate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = constraint_set
        bounds = tuple(bound for asset in range(num_assets))
        
        # Use a more robust solver method
        result = sco.minimize(neg_sharpe_ratio, num_assets * [1./num_assets], args=args,
                             bounds=bounds, constraints=constraints, method='SLSQP')
        
        if not result['success']:
            st.warning("Optimization for maximum Sharpe ratio failed to converge.")
            # Fallback to equal weights
            return {'x': num_assets * [1./num_assets], 'fun': 0}
            
        return result
    except Exception as e:
        st.error(f"Error in max Sharpe ratio calculation: {e}")
        num_assets = len(mean_returns)
        return {'x': num_assets * [1./num_assets], 'fun': 0}  # Equal weights as fallback

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix, constraint_set=(0, 1)):
    try:
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = constraint_set
        bounds = tuple(bound for asset in range(num_assets))
        
        result = sco.minimize(portfolio_volatility, num_assets * [1./num_assets], args=args,
                             bounds=bounds, constraints=constraints, method='SLSQP')
        
        if not result['success']:
            st.warning("Optimization for minimum variance failed to converge.")
            # Fallback to equal weights
            return {'x': num_assets * [1./num_assets], 'fun': 0}
            
        return result
    except Exception as e:
        st.error(f"Error in min variance calculation: {e}")
        num_assets = len(mean_returns)
        return {'x': num_assets * [1./num_assets], 'fun': 0}  # Equal weights as fallback

def efficient_frontier(mean_returns, cov_matrix, returns_range, constraint_set=(0, 1)):
    efficient_portfolios = []
    for ret in returns_range:
        try:
            constraints = ({'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[1] - ret},
                          {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple(constraint_set for asset in range(len(mean_returns)))
            result = sco.minimize(portfolio_volatility, len(mean_returns) * [1. / len(mean_returns)],
                                args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds,
                                constraints=constraints)
            
            if result['success']:
                efficient_portfolios.append(result['x'])
            else:
                # In case of failure, use equal weights
                efficient_portfolios.append(len(mean_returns) * [1. / len(mean_returns)])
        except Exception:
            # In case of calculation error, use equal weights
            efficient_portfolios.append(len(mean_returns) * [1. / len(mean_returns)])
            
    return efficient_portfolios

def calculate_efficient_frontier(mean_returns, cov_matrix, allow_short=False):
    try:
        # Define returns range
        min_ret = min(mean_returns)
        max_ret = max(mean_returns)
        returns_range = np.linspace(min_ret, max_ret, 50)
        
        # Constraints based on whether short selling is allowed
        constraint_set = (-1, 1) if allow_short else (0, 1)
        
        # Calculate minimum variance portfolio (GMVP)
        min_vol_result = min_variance(mean_returns, cov_matrix, constraint_set)
        min_vol_weights = min_vol_result['x']
        min_vol_std, min_vol_ret = portfolio_performance(min_vol_weights, mean_returns, cov_matrix)
        
        # Calculate portfolio with maximum Sharpe ratio
        max_sharpe_result = max_sharpe_ratio(mean_returns, cov_matrix, constraint_set=constraint_set)
        max_sharpe_weights = max_sharpe_result['x']
        max_sharpe_std, max_sharpe_ret = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix)
        
        # Calculate efficient frontier
        efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, returns_range, constraint_set)
        
        # Extract returns and risks for the efficient frontier
        efficient_stds = [portfolio_performance(weights, mean_returns, cov_matrix)[0] for weights in efficient_portfolios]
        efficient_returns = [portfolio_performance(weights, mean_returns, cov_matrix)[1] for weights in efficient_portfolios]
        
        return {
            'efficient_stds': efficient_stds,
            'efficient_returns': efficient_returns,
            'min_vol': {'weights': min_vol_weights, 'std': min_vol_std, 'ret': min_vol_ret},
            'max_sharpe': {'weights': max_sharpe_weights, 'std': max_sharpe_std, 'ret': max_sharpe_ret}
        }
    except Exception as e:
        st.error(f"Error calculating efficient frontier: {e}")
        # Return default values
        num_assets = len(mean_returns)
        equal_weights = num_assets * [1./num_assets]
        equal_std, equal_ret = portfolio_performance(equal_weights, mean_returns, cov_matrix)
        
        return {
            'efficient_stds': [equal_std],
            'efficient_returns': [equal_ret],
            'min_vol': {'weights': equal_weights, 'std': equal_std, 'ret': equal_ret},
            'max_sharpe': {'weights': equal_weights, 'std': equal_std, 'ret': equal_ret}
        }

# Function to determine the optimal portfolio based on risk aversion
def optimal_portfolio(mean_returns, cov_matrix, risk_aversion, allow_short=False):
    try:
        constraint_set = (-1, 1) if allow_short else (0, 1)
        
        # Optimization based on utility function U = r - (sigma^2 * A/2)
        def objective(weights):
            returns = np.sum(mean_returns * weights)
            variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            utility = returns - (variance * risk_aversion / 2)
            return -utility  # Minimize the negative utility
        
        num_assets = len(mean_returns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple(constraint_set for asset in range(num_assets))
        
        result = sco.minimize(objective, num_assets * [1./num_assets],
                             bounds=bounds, constraints=constraints, method='SLSQP')
        
        if not result['success']:
            st.warning("Optimization for optimal portfolio failed to converge. Using equal weights.")
            weights = num_assets * [1./num_assets]
        else:
            weights = result['x']
            
        returns = np.sum(mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        return {'weights': weights, 'returns': returns, 'std': std, 'utility': -result.get('fun', 0)}
    except Exception as e:
        st.error(f"Error in optimal portfolio calculation: {e}")
        num_assets = len(mean_returns)
        weights = num_assets * [1./num_assets]  # Equal weights as fallback
        returns = np.sum(mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return {'weights': weights, 'returns': returns, 'std': std, 'utility': 0}

# Definition of questions and associated scores to calculate risk aversion
questions = [
    {
        "id": 1,
        "question": "Is this for yourself or for someone else?",
        "options": ["For myself", "For someone else"],
        "scores": [0, 0]  # No impact on score
    },
    {
        "id": 2,
        "question": "What is the person's age?",
        "type": "slider",
        "min": 18,
        "max": 100,
        "default": 35,
        "score_func": lambda age: (age - 18) / (100 - 18) * 3  # Higher age increases risk aversion (0-3)
    },
    {
        "id": 3,
        "question": "What is the gender?",
        "options": ["Male", "Female", "Other"],
        "scores": [0, 0.5, 0.25]  # Slight difference based on life expectancy statistics
    },
    {
        "id": 4,
        "question": "Where does the person live?",
        "options": ["North America", "Europe", "Asia", "Other"],
        "scores": [0, 0, 0, 0]  # No impact on score
    },
    {
        "id": 5,
        "question": "How much do you want to invest?",
        "type": "slider",
        "min": 1000,
        "max": 1000000,
        "default": 10000,
        "step": 1000,
        "format": "$%d",
        "score_func": lambda amount: max(0, 2 - (amount / 100000))  # Higher amount decreases risk aversion (max 2)
    },
    {
        "id": 6,
        "question": "How long do you want to invest for?",
        "type": "slider",
        "min": 1,
        "max": 30,
        "default": 10,
        "format": "%d years",
        "score_func": lambda years: max(0, 3 - (years / 10))  # Longer duration decreases risk aversion (max 3)
    },
    {
        "id": 7,
        "question": "What is your risk tolerance?",
        "options": ["Low", "Medium", "High"],
        "scores": [2, 1, 0]  # Direct impact on risk aversion score
    }
]

# User interface with Streamlit
def main():
    # Sidebar with tabs
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a section", ["Questionnaire", "Optimized Portfolio", "Efficient Frontier", "About"])
    
    # Add logo and title in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### BMD5302 Financial Modeling")
    st.sidebar.markdown("#### Project: Financial Robot Advisor")
    st.sidebar.markdown("AY 2024/25 Semester 2")
    st.sidebar.markdown("---")
    st.sidebar.info("This robot advisor helps you determine your investor profile and suggests a portfolio of funds tailored to your risk aversion level.")
    
    try:
        # Retrieve fund data from local Excel files if not already cached
        with st.spinner("Loading financial data from local Excel files... (This may take a moment)"):
            data = get_fund_data(list(funds.keys()))
        
        # Safety check
        if data is None or data.empty:
            st.error("Could not retrieve any financial data. Please check the Excel files.")
            return
        
        # Check if data has sufficient rows for analysis
        if len(data) < 30:  # Need at least a month of data
            st.warning("Limited historical data available. Results may not be reliable.")
            
        st.success(f"Successfully loaded data for {data.shape[1]} funds over {data.shape[0]} trading days.")
        
        if page == "Questionnaire":
            st.header("Risk Aversion Questionnaire")
            st.write("Answer the following questions to determine your investor profile and risk aversion level.")
            
            # Initialize risk aversion score
            risk_aversion_score = 0
            responses = {}
            
            # Display questions and collect responses
            for q in questions:
                st.subheader(f"Question {q['id']}: {q['question']}")
                
                if q.get("type") == "slider":
                    value = st.slider(
                        "Your answer",
                        min_value=q["min"],
                        max_value=q["max"],
                        value=q["default"],
                        step=q.get("step", 1),
                        format=q.get("format", "%d")
                    )
                    responses[q["id"]] = value
                    risk_aversion_score += q["score_func"](value)
                else:
                    option = st.radio("Your answer", q["options"])
                    option_index = q["options"].index(option)
                    responses[q["id"]] = option
                    risk_aversion_score += q["scores"][option_index]
            
            # Normalize risk aversion score on a scale of 1 to 10
            normalized_risk_aversion = max(1, min(10, risk_aversion_score))
            
            st.markdown("---")
            st.subheader("Your Investor Profile")
            
            # Display investor profile
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Risk Aversion Score", f"{normalized_risk_aversion:.2f}/10")
                
                # Determine investor profile
                if normalized_risk_aversion < 3:
                    investor_profile = "Aggressive"
                    description = "You are comfortable with high volatility and prioritize potentially high returns."
                elif normalized_risk_aversion < 7:
                    investor_profile = "Balanced"
                    description = "You seek a balance between growth and security in your investments."
                else:
                    investor_profile = "Conservative"
                    description = "You prioritize security and capital preservation over growth."
                    
                st.info(f"**Investor Profile:** {investor_profile}\n\n{description}")
                
            with col2:
                # Create a gauge chart to visualize risk aversion score
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = normalized_risk_aversion,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Aversion"},
                    gauge = {
                        'axis': {'range': [0, 10]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 3], 'color': "green"},
                            {'range': [3, 7], 'color': "yellow"},
                            {'range': [7, 10], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': normalized_risk_aversion
                        }
                    }
                ))
                st.plotly_chart(fig)
            
            # Save responses and score in the session
            st.session_state["risk_aversion_score"] = normalized_risk_aversion
            st.session_state["investor_profile"] = investor_profile
            st.session_state["responses"] = responses
            
            st.markdown("---")
            st.write("Go to the 'Optimized Portfolio' section to see your recommended fund allocation.")
            
        elif page == "Optimized Portfolio": #Page 2 
            st.header("Your Optimized Fund Portfolio")
            
            # Check if the user has already completed the questionnaire
            if "risk_aversion_score" not in st.session_state:
                st.warning("Please complete the questionnaire first to get your optimized portfolio.")
                if st.button("Go to Questionnaire"):
                    st.session_state["_pages"] = "Questionnaire"
                return
            
            # Retrieve risk aversion score
            risk_aversion_score = st.session_state["risk_aversion_score"]
            investor_profile = st.session_state["investor_profile"]
            
            st.subheader(f"Investor Profile: {investor_profile} (Risk Aversion Score: {risk_aversion_score:.2f}/10)")
            
            # Parameters for optimization
            allow_short = st.checkbox("Allow short selling", value=False)
            
            # Calculate portfolio statistics
            with st.spinner("Calculating portfolio statistics..."):
                returns, mean_returns, cov_matrix = calculate_portfolio_stats(data)
            
            # Find optimal portfolio based on risk aversion
            with st.spinner("Optimizing portfolio..."):
                optimal_portfolio_result = optimal_portfolio(mean_returns, cov_matrix, risk_aversion_score, allow_short)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Recommended Fund Allocation")
                
                # Create DataFrame for display
                weights_df = pd.DataFrame({
                    'Fund': [funds[fund] for fund in data.columns],
                    'Symbol': data.columns,
                    'Allocation (%)': optimal_portfolio_result['weights'] * 100
                })
                
                weights_df = weights_df.sort_values('Allocation (%)', ascending=False).reset_index(drop=True)
                
                # Display weights table
                st.dataframe(weights_df.round(2))
                
                # Portfolio metrics
                st.metric("Expected Annual Return", f"{optimal_portfolio_result['returns'] * 100:.2f}%")
                st.metric("Annual Volatility", f"{optimal_portfolio_result['std'] * 100:.2f}%")
                
                sharpe_ratio = (optimal_portfolio_result['returns'] - 0.02) / optimal_portfolio_result['std']
                st.metric("Sharpe Ratio (Rf=2%)", f"{sharpe_ratio:.2f}")
           
            ## New Version 2 
            with col2:
                st.subheader("Portfolio Distribution")

                # Prepare allocations
                pie_data = weights_df.copy()
                positive_allocations = pie_data[pie_data['Allocation (%)'] > 0].copy()
                negative_allocations = pie_data[pie_data['Allocation (%)'] < 0].copy()

                # Compute summary for long vs short
                total_abs = positive_allocations['Allocation (%)'].sum() + abs(negative_allocations['Allocation (%)']).sum()
                summary_df = pd.DataFrame({
                    'Position': ['Long', 'Short'],
                    'Allocation': [
                        positive_allocations['Allocation (%)'].sum(),
                        abs(negative_allocations['Allocation (%)'].sum())
                    ]
                })

                # Combine for bar chart
                combined_df = pd.concat([positive_allocations, negative_allocations])
                combined_df = combined_df.sort_values('Allocation (%)')

                # Chart selector (carousel-style)
                chart_option = st.radio(
                    "📊 Select Chart View",
                    ("Long Allocation Pie", "Short Allocation Pie", "Long vs Short Exposure", "Bar Chart: All Funds")
                )

                # Chart logic
                if chart_option == "Long Allocation Pie" and len(positive_allocations) > 0:
                    fig = px.pie(
                        positive_allocations,
                        values='Allocation (%)',
                        names='Symbol',
                        title="Distribution of Positive Allocations",
                        hover_data=['Fund'],
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    st.plotly_chart(fig)
                elif chart_option == "Short Allocation Pie":
                    if len(negative_allocations) > 0 and allow_short:
                            fig = px.pie(
                                negative_allocations,
                                values=negative_allocations['Allocation (%)'].abs(),
                                names='Symbol',
                                title="Distribution of Short Sales",
                                hover_data=['Fund'],
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            st.plotly_chart(fig)
                    elif not allow_short:
                        st.info("🔒 Short selling is disabled.\n\nEnable the allow short selling checkbox to view short allocation details.")
                    else:
                        st.warning("No short positions were recommended in this portfolio.")

                elif chart_option == "Long vs Short Exposure":
                    fig = px.pie(
                        summary_df,
                        names='Position',
                        values='Allocation',
                        title='Portfolio Exposure Breakdown (Long vs Short)',
                        color_discrete_map={'Long': 'green', 'Short': 'red'}
                    )
                    st.plotly_chart(fig)

                elif chart_option == "Bar Chart: All Funds":
                    fig = px.bar(
                        combined_df,
                        x='Allocation (%)',
                        y='Symbol',
                        orientation='h',
                        color=combined_df['Allocation (%)'] > 0,
                        color_discrete_map={True: 'green', False: 'red'},
                        title='All Fund Allocations (Long & Short)',
                        hover_data=['Fund']
                    )
                    st.plotly_chart(fig)

                # Net exposure summary
                net_exposure = positive_allocations['Allocation (%)'].sum() + negative_allocations['Allocation (%)'].sum()
                st.metric("Net Market Exposure", f"{net_exposure:.2f}%")


            ## New VERSION 2 

            # Add section for historical performance
            # modification1: use only historical data at each time step to calculate returns, covariance; not the full data.
            # modification2: to avoid that some funds are not available at the time of calculation, we need to check the availability of funds at each time step.
            # e.g. a fund only has data starting from 2024-09
            st.subheader("Simulated Historical Performance")

            # --- Get returns data BEFORE the loop ---
            # Calculate returns on the full, initially loaded data (before any date filtering)
            # We need the full returns series to check availability at each step
            try:
                full_returns = data.pct_change() # Keep NaNs for now
                # Drop the first row which is all NaN after pct_change
                if not full_returns.empty:
                    full_returns = full_returns.iloc[1:]
            except Exception as e:
                 st.error(f"Error calculating initial returns: {e}")
                 return # Cannot proceed without returns

            # Safety check for returns
            if full_returns is None or full_returns.empty:
                st.warning("Unable to calculate historical performance due to insufficient data.")
            else:
                # Simulate historical portfolio returns using an expanding window
                min_window_size = 252 # Minimum data points needed for first calculation in the window, use 1 year
                portfolio_daily_returns = []
                benchmark_daily_returns_list = [] # Initialize list for benchmark returns
                start_index = min_window_size # Start simulation after the minimum window

                # Check if data length is sufficient for the initial window
                if len(full_returns) > start_index: # Use start_index here, as loop starts from it
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_steps = len(full_returns) - start_index

                    with st.spinner(f"Running backtest (Min Window: {min_window_size} days)..."):
                        # Loop through the data, starting after the minimum window period
                        # The loop index 'i' represents the day whose return we are calculating,
                        # using data up to 'i-1'.
                        for i in range(start_index, len(full_returns)):
                
                            current_date = full_returns.index[i]              
                            progress = (i - start_index + 1) / total_steps
                            progress_bar.progress(progress)
                            status_text.text(f"Running backtest... {progress*100:.0f}% ({current_date.strftime('%Y-%m-%d')})")

                            window_returns_raw = full_returns.iloc[0 : i]       
                            # --- Identify valid funds for optimization within the window ---
                            # A fund is valid if it has enough non-NaN returns AND non-zero variance in the window
                            valid_funds_mask = (window_returns_raw.notna().sum() >= min_window_size // 2) & \
                                               (window_returns_raw.var() > 1e-10) # Check variance and sufficient data points
                            valid_funds = full_returns.columns[valid_funds_mask]

                            optimal_weights = pd.Series(0.0, index=full_returns.columns) # Initialize weights as zeros

                            if len(valid_funds) > 0:
                                # Filter window returns for valid funds and drop NaNs within this subset for calculation
                                window_returns_subset = window_returns_raw[valid_funds].dropna()

                                # Check if enough data remains after dropna for subset calculation
                                if len(window_returns_subset) >= min_window_size // 2 :
                                    try:
                                        # Calculate stats ONLY for the valid subset in the window
                                        subset_mean_returns = window_returns_subset.mean() * 252
                                        subset_cov_matrix = window_returns_subset.cov() * 252

                                        # Ensure subset cov matrix is valid (e.g., not NaN, positive definite)
                                        if subset_cov_matrix.isnull().values.any() or \
                                           (len(valid_funds) > 1 and np.linalg.det(subset_cov_matrix) == 0) or \
                                           (len(valid_funds) == 1 and subset_cov_matrix.iloc[0,0] <= 1e-10): # Handle single asset case
                                             raise ValueError("Subset Covariance matrix issue")

                                        # Find optimal portfolio for the subset
                                        subset_optimal_result = optimal_portfolio(subset_mean_returns, subset_cov_matrix, risk_aversion_score, allow_short)
                                        subset_weights = subset_optimal_result['weights']

                                        # Assign calculated weights to the valid funds
                                        optimal_weights.loc[valid_funds] = subset_weights

                                    except Exception as e:
                                        # Fallback for subset optimization failure: Equal weight among funds available *at day i*
                                        fallback_reason = f"Subset optimization failed: {e}"
                                        print(f"Fallback Triggered: {fallback_reason}")
                                        # Determine available funds at day i (non-NaN in full_returns.iloc[i])
                                        day_i_returns = full_returns.iloc[i]
                                        valid_day_i_funds_mask = day_i_returns.notna()
                                        num_valid_day_i = valid_day_i_funds_mask.sum()
                                        if num_valid_day_i > 0:
                                            optimal_weights.loc[valid_day_i_funds_mask] = 1.0 / num_valid_day_i
                                else:
                                     # Fallback if not enough data even for subset after dropna
                                     fallback_reason = "Not enough valid data points in window subset"
                                     print(f"Fallback Triggered: {fallback_reason}")
                                     day_i_returns = full_returns.iloc[i]
                                     valid_day_i_funds_mask = day_i_returns.notna()
                                     num_valid_day_i = valid_day_i_funds_mask.sum()
                                     if num_valid_day_i > 0:
                                         optimal_weights.loc[valid_day_i_funds_mask] = 1.0 / num_valid_day_i

                            else:
                                # Fallback if no funds are valid in the window: Equal weight among funds available *at day i*
                                fallback_reason = "No valid funds in window"
                                print(f"Fallback Triggered: {fallback_reason}")
                                day_i_returns = full_returns.iloc[i]
                                valid_day_i_funds_mask = day_i_returns.notna()
                                num_valid_day_i = valid_day_i_funds_mask.sum()
                                if num_valid_day_i > 0:
                                    optimal_weights.loc[valid_day_i_funds_mask] = 1.0 / num_valid_day_i

                            # --- Convert optimal_weights Series to numpy array for dot product ---
                            optimal_weights_array = optimal_weights.values

                            # --- Calculate Optimized Portfolio Return for day i ---
                            day_i_returns = full_returns.iloc[i]
                            # Use fillna(0) for dot product; weight for NaN return fund is 0 anyway
                            day_return = day_i_returns.fillna(0).dot(optimal_weights_array)
                            portfolio_daily_returns.append(day_return)

                            # --- Calculate Benchmark Return for day i ---
                            # Equal weight only among funds with non-NaN returns on day i
                            valid_day_i_returns = day_i_returns.dropna()
                            if not valid_day_i_returns.empty:
                                benchmark_day_return = valid_day_i_returns.mean()
                            else:
                                benchmark_day_return = 0.0 # Assign 0 if no fund has data on this day
                            benchmark_daily_returns_list.append(benchmark_day_return)

                    # --- After the loop ---
                    # Create Series for the calculated daily returns, aligning index
                    portfolio_returns_series = pd.Series(portfolio_daily_returns, index=full_returns.index[start_index:])
                    benchmark_returns_series = pd.Series(benchmark_daily_returns_list, index=full_returns.index[start_index:]) # <-- Create benchmark series

                    # Calculate cumulative values
                    cumulative_returns = (1 + portfolio_returns_series).cumprod()
                    benchmark_cumulative_returns = (1 + benchmark_returns_series).cumprod() #  Calculate benchmark cumulative
                    # --- MODIFICATION END ---

                # Safety check for cumulative returns
                if cumulative_returns.empty:
                    st.warning("No historical performance data available.")
                else:
                    # Create DataFrame for chart
                    performance_df = pd.DataFrame({
                        'Optimized Portfolio': cumulative_returns,
                        'Benchmark (equal weight)': benchmark_cumulative_returns
                    })
                    
                    # Initial investment amount
                    if 'responses' in st.session_state and 5 in st.session_state['responses']:
                        initial_investment = st.session_state['responses'][5]
                    else:
                        initial_investment = 10000  # Default value
                    
                    # Project portfolio value
                    st.write(f"Projection based on an initial investment of ${initial_investment:,.2f}")

                    # Display performance chart
                    fig = px.line(
                        performance_df,
                        title="Cumulative Performance (base 1)",
                        labels={"value": "Relative Value", "variable": ""}
                    )
                    st.plotly_chart(fig)
                    
                    # Create another chart with projected value in dollars
                    value_df = performance_df * initial_investment
                    fig_value = px.line(
                        value_df,
                        title=f"Projected Portfolio Value (Initial $: {initial_investment:,.2f})",
                        labels={"value": "Value ($)", "variable": ""}
                    )
                    st.plotly_chart(fig_value)
                    
                    # Performance statistics
                    annual_return = portfolio_returns_series.mean() * 252
                    annual_volatility = portfolio_returns_series.std() * np.sqrt(252)
                    
                    # Safely calculate Sharpe ratio
                    if annual_volatility > 0:
                        sharpe_ratio = annual_return / annual_volatility
                    else:
                        sharpe_ratio = 0
                    
                    # Safely calculate max drawdown
                    if len(cumulative_returns) > 0:
                        cummax = cumulative_returns.cummax()
                        drawdown = (cumulative_returns / cummax - 1)
                        max_drawdown = drawdown.min()
                    else:
                        max_drawdown = 0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Annualized Return", f"{annual_return * 100:.2f}%")
                    col2.metric("Annualized Volatility", f"{annual_volatility * 100:.2f}%")
                    col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    col4.metric("Maximum Drawdown", f"{max_drawdown * 100:.2f}%")
                    
                    # Future projections
                    st.subheader("Future Projections")
                    
                    if 'responses' in st.session_state and 6 in st.session_state['responses']:
                        investment_horizon = st.session_state['responses'][6]
                    else:
                        investment_horizon = 10  # Default value
                    
                    # Simulate future returns
                    n_simulations = 1000
                    n_years = investment_horizon
                    
                    # Safely get the last value of cumulative returns with error handling
                    if len(cumulative_returns) > 0:
                        last_value = cumulative_returns.iloc[-1]
                    else:
                        last_value = 1.0  # Default to 1.0 (no growth) if no data
                    
                    # Create a normal distribution for returns with safeguards
                    # Use absolute values to avoid negative volatility
                    safe_volatility = max(0.001, annual_volatility)  # Ensure positive minimum volatility
                    simulated_returns = np.random.normal(
                        annual_return / 252,  # Daily mean
                        safe_volatility / np.sqrt(252),  # Daily standard deviation
                        (n_simulations, 252 * n_years)  # 252 trading days per year
                    )
                    
                    # Simulate future paths
                    simulated_paths = np.zeros((n_simulations, 252 * n_years))
                    simulated_paths[:, 0] = last_value
                    for i in range(1, 252 * n_years):
                        simulated_paths[:, i] = simulated_paths[:, i-1] * (1 + simulated_returns[:, i])
                    
                    # Extract some paths for display
                    n_paths_display = 50
                    indices = np.random.choice(n_simulations, n_paths_display, replace=False)
                    
                    # Create a DataFrame for display
                    dates = pd.date_range(
                        start=data.index[-1] + pd.Timedelta(days=1) if hasattr(data.index, '__getitem__') else datetime.now(),
                        periods=252 * n_years,
                        freq='B'  # Business days
                    )
                    
                    future_df = pd.DataFrame(
                        simulated_paths[indices, :].T,
                        index=dates,
                        columns=[f'Path {i+1}' for i in range(n_paths_display)]
                    )
                    
                    # Add the last historical value for a smooth transition
                    hist_last = pd.DataFrame(
                        {f'Path {i+1}': [last_value] for i in range(n_paths_display)},
                        index=[dates[0] - pd.Timedelta(days=1)]
                    )
                    
                    future_df = pd.concat([hist_last, future_df])
                    
                    # Display projection
                    future_value_df = future_df * initial_investment
                    
                    # Create simulations chart
                    fig_sim = go.Figure()
                    
                    # Add simulated paths
                    for col in future_value_df.columns:
                        fig_sim.add_trace(go.Scatter(
                            x=future_value_df.index,
                            y=future_value_df[col],
                            mode='lines',
                            opacity=0.3,
                            line=dict(width=1),
                            showlegend=False
                        ))
                    
                    # Add percentiles
                    percentiles = [5, 50, 95]
                    percentile_values = np.percentile(simulated_paths[:, -1], percentiles)
                    
                    for i, p in enumerate(percentiles):
                        final_value = percentile_values[i] * initial_investment
                        fig_sim.add_trace(go.Scatter(
                            x=[future_value_df.index[-1]],
                            y=[final_value],
                            mode='markers',
                            marker=dict(size=10),
                            name=f'Percentile {p}% (${final_value:,.2f})'
                        ))
                    
                    # Formatting
                    fig_sim.update_layout(
                        title=f"Projection over {n_years} years",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        height=600
                    )
                    
                    st.plotly_chart(fig_sim)
                    
                    # Summary table of projection statistics
                    final_values = simulated_paths[:, -1] * initial_investment
                    projection_stats = {
                        'Statistic': ['Minimum', 'Percentile 5%', 'Median', 'Mean', 'Percentile 95%', 'Maximum'],
                        'Projected Value ($)': [
                            f"${np.min(final_values):,.2f}",
                            f"${np.percentile(final_values, 5):,.2f}",
                            f"${np.percentile(final_values, 50):,.2f}",
                            f"${np.mean(final_values):,.2f}",
                            f"${np.percentile(final_values, 95):,.2f}",
                            f"${np.max(final_values):,.2f}"
                        ]
                    }
                    
                    st.table(pd.DataFrame(projection_stats))
                    
        elif page == "Efficient Frontier":
            st.header("Efficient Frontier Analysis")
            
            # Parameters for optimization
            allow_short = st.checkbox("Allow short selling", value=False)
            
            # Calculate portfolio statistics
            with st.spinner("Calculating portfolio statistics..."):
                returns, mean_returns, cov_matrix = calculate_portfolio_stats(data)
            
            # Calculate efficient frontier
            with st.spinner("Calculating efficient frontier..."):
                ef_results = calculate_efficient_frontier(mean_returns, cov_matrix, allow_short)
            
            # Create efficient frontier chart
            fig = go.Figure()
            
            # Add efficient frontier with error checking
            if len(ef_results['efficient_stds']) > 1:
                fig.add_trace(go.Scatter(
                    x=ef_results['efficient_stds'],
                    y=ef_results['efficient_returns'],
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='blue', width=2)
                ))
            
            # Add minimum variance portfolio
            fig.add_trace(go.Scatter(
                x=[ef_results['min_vol']['std']],
                y=[ef_results['min_vol']['ret']],
                mode='markers',
                name='Minimum Variance Portfolio',
                marker=dict(color='green', size=15, symbol='diamond')
            ))
            
            # Add maximum Sharpe ratio portfolio
            fig.add_trace(go.Scatter(
                x=[ef_results['max_sharpe']['std']],
                y=[ef_results['max_sharpe']['ret']],
                mode='markers',
                name='Maximum Sharpe Ratio Portfolio',
                marker=dict(color='red', size=15, symbol='star')
            ))
            
            # Add individual assets with error checking
            for i, fund in enumerate(data.columns):
                try:
                    asset_std = np.sqrt(cov_matrix.iloc[i, i])
                    asset_return = mean_returns[i]
                    fig.add_trace(go.Scatter(
                        x=[asset_std],
                        y=[asset_return],
                        mode='markers',
                        name=fund,
                        marker=dict(size=10)
                    ))
                except Exception as e:
                    st.error(f"Error plotting fund {fund}: {e}")
            
            # If available, add optimized portfolio based on user's risk aversion
            if "risk_aversion_score" in st.session_state:
                risk_aversion_score = st.session_state["risk_aversion_score"]
                with st.spinner("Calculating your optimal portfolio..."):
                    optimal = optimal_portfolio(mean_returns, cov_matrix, risk_aversion_score, allow_short)
                
                fig.add_trace(go.Scatter(
                    x=[optimal['std']],
                    y=[optimal['returns']],
                    mode='markers',
                    name='Your Optimal Portfolio',
                    marker=dict(color='purple', size=15, symbol='circle')
                ))
            
            # Chart formatting
            # fig.update_layout(
            #     title="Efficient Frontier",
            #     xaxis_title="Annualized Risk (Standard Deviation)",
            #     yaxis_title="Annualized Return",
            #     legend=dict(
            #         y=0.99,
            #         x=0.01,
            #         yanchor="top",
            #         xanchor="left"
            #     ),
            #     autosize=True,
            #     height=600,
            #     width=800
            # )
            fig.update_layout(
                title="Efficient Frontier",
                xaxis_title="Annualized Risk (Standard Deviation)",
                yaxis_title="Annualized Return",
                legend=dict(
                    orientation="v",
                    y=1,
                    x=1.05,  # Moves legend to the right of the plot
                    xanchor="left",
                    yanchor="top"
                ),
                margin=dict(r=150),  # Add right margin to avoid overlap
                autosize=True,
                height=600,
                width=900  # Optional: widen chart to accommodate right legend
            )
            
            st.plotly_chart(fig)
            
            # Display key portfolio details
            st.subheader("Key Portfolio Compositions")
            
            # Prepare data for display
            key_portfolios = {
                "Minimum Variance Portfolio (GMVP)": ef_results['min_vol']['weights'],
                "Maximum Sharpe Ratio Portfolio": ef_results['max_sharpe']['weights']
            }
            
            if "risk_aversion_score" in st.session_state:
                key_portfolios["Your Optimal Portfolio"] = optimal['weights']
            
            # Create multi-index DataFrame
            multi_index_data = []
            for portfolio_name, weights in key_portfolios.items():
                for i, fund in enumerate(data.columns):
                    multi_index_data.append({
                        "Portfolio": portfolio_name,
                        "Fund ID": fund,
                        "Fund Name": funds[fund],
                        "Allocation (%)": weights[i] * 100
                    })
            
            portfolio_df = pd.DataFrame(multi_index_data)
            
            # Pivot for more readable format with error handling
            try:
                pivot_df = portfolio_df.pivot(index=["Fund ID", "Fund Name"], columns="Portfolio", values="Allocation (%)")
                st.dataframe(pivot_df.round(2))
            except Exception as e:
                st.error(f"Error creating portfolio comparison: {e}")
                st.dataframe(portfolio_df)
                
        elif page == "About":
            st.header("About the Financial Robot Advisor")
            st.markdown("""
            ## BMD5302 Financial Modeling Project
            This financial robot advisor was developed as part of the BMD5302 Financial Modeling course. It allows you to:
            1. Determine your investor profile through a questionnaire
            2. Calculate your risk aversion level
            3. Build an optimized portfolio of funds based on your profile
            4. Visualize the efficient frontier and different optimal portfolios
            
            ### Methodology
            #### The Efficient Frontier
            The model uses Markowitz's Modern Portfolio Theory to construct the efficient frontier, which represents the set of portfolios offering the best return for a given level of risk. Optimization is performed with and without short selling.
            
            #### Risk Aversion
            The investor's utility function is modeled by:
            $U = r - \\sigma^2 \\times \\frac{A}{2}$
            
            where:
            - $r$ is the expected return
            - $\\sigma$ is the standard deviation (risk)
            - $A$ is the risk aversion coefficient
            
            #### The Optimal Portfolio
            The optimal portfolio is determined by maximizing the investor's utility function, taking into account their risk aversion level calculated from the questionnaire responses.
            
            ### Data Used
            The model uses historical NAV data from 15 different funds, including cash funds, bond funds, and equity funds in various currencies and regions.
            
            ### Limitations
            - Past performance does not guarantee future results
            - The model assumes normal distribution of returns
            - Transaction costs and tax implications are not taken into account
            - Currency risk may not be fully captured
            
            ### Technologies Used
            - Python
            - Streamlit for the web interface
            - Pandas for data manipulation
            - Plotly for visualizations
            - NumPy and SciPy for optimization calculations
            """)
            
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.info("Please try refreshing the page. If the problem persists, restart the application.")

# Run the application
if __name__ == "__main__":
    main()