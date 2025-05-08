import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import anthropic
from openai import OpenAI

# Set page configuration
st.set_page_config(
    page_title="Stock Analysis App",
    page_icon="📈",
    layout="wide"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1.2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e0e0e0;
        border-bottom: 2px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">Stock Analysis Dashboard</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Stock Search")
    ticker = st.text_input("Enter Stock Ticker Symbol:", value="AAPL").upper()
    
    st.markdown("### AI Model Configuration")
    api_provider = st.radio("Select API Provider:", ["Claude", "OpenAI"])
    
    if api_provider == "Claude":
        claude_model = st.selectbox(
            "Select Claude Model:", 
            ["claude-3-7-sonnet-latest", "claude-3-5-sonnet-latest"]
        )
        api_key = st.text_input("Enter Claude API Key:", type="password")
    else:
        openai_model = st.selectbox(
            "Select OpenAI Model:", 
            ["o4-mini", "o3", "o3-mini", "o1", "gpt-4.1", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"]
        )
        api_key = st.text_input("Enter OpenAI API Key:", type="password")

# Function to get stock data
@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # Get historical data for last 90 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        hist = stock.history(start=start_date, end=end_date)
        
        # Get financial statements
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cash_flow = stock.cashflow
        
        return {
            'info': info,
            'hist': hist,
            'balance_sheet': balance_sheet,
            'income_stmt': income_stmt,
            'cash_flow': cash_flow
        }
    except Exception as e:
        st.error(f"Error retrieving data for {ticker_symbol}: {str(e)}")
        return None

# Function to analyze stock with Claude
def analyze_with_claude(api_key, model, stock_data):
    client = anthropic.Anthropic(api_key=api_key)
    
    # Prepare data for analysis
    info = stock_data['info']
    
    # Create prompt
    prompt = f"""
You are a financial analyst expert. I'd like you to analyze this stock and provide investment insights.

Company: {info.get('shortName', 'N/A')} ({info.get('symbol', 'N/A')})
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}
Current Price: ${info.get('currentPrice', 'N/A')}
Market Cap: ${info.get('marketCap', 'N/A')}
P/E Ratio: {info.get('trailingPE', 'N/A')}
52-Week Range: ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}

Please analyze this stock and provide:
1. A brief overview of the company
2. Analysis on Income Statement Cash Flow based on last 5 years
3. Analysis on Balance Sheet based on last 5 years
4. Analysis on Cash Flow Statement based on last 5 years
5. Potential risks and opportunities
6. final recommendation (Buy or Sell) with rationale for value investing
"""

    try:
        response = client.messages.create(
            model=model,
            max_completion_tokens=4000,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error during analysis: {str(e)}"

# Function to analyze stock with OpenAI
def analyze_with_openai(api_key, model, stock_data):
    client = OpenAI(api_key=api_key)
    
    # Prepare data for analysis
    info = stock_data['info']
    
    # Create prompt
    prompt = f"""
You are a financial analyst expert. I'd like you to analyze this stock and provide investment insights.

Company: {info.get('shortName', 'N/A')} ({info.get('symbol', 'N/A')})
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}
Current Price: ${info.get('currentPrice', 'N/A')}
Market Cap: ${info.get('marketCap', 'N/A')}
P/E Ratio: {info.get('trailingPE', 'N/A')}
52-Week Range: ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}

Please analyze this stock and provide:
1. A brief overview of the company
2. Analysis on Income Statement based on last 5 years
3. Analysis on Balance Sheet based on last 5 years
4. Analysis on Cash Flow Statement based on last 5 years
5. Potential risks and opportunities
6. final recommendation (Buy or Sell) with rationale for value investing
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a skilled financial analyst who provides concise, accurate stock analyses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during analysis: {str(e)}"

# Main content
if ticker:
    # Get stock data
    data = get_stock_data(ticker)
    
    if data:
        info = data['info']
        hist = data['hist']
        
        # Display company header information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"### {info.get('shortName', 'N/A')} ({info.get('symbol', 'N/A')})")
            st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
            st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
        
        with col2:
            st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}", 
                      f"{round(info.get('regularMarketChangePercent', 0), 2)}%" if 'regularMarketChangePercent' in info else "N/A")
            st.markdown(f"**52-Week Range:** ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}")
        
        with col3:
            st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}" if isinstance(info.get('marketCap'), (int, float)) else "N/A")
            st.markdown(f"**P/E Ratio:** {round(info.get('trailingPE', 0), 2) if 'trailingPE' in info else 'N/A'}")
            st.markdown(f"**Dividend Yield:** {round(info.get('dividendYield', 0) * 100, 2)}%" if 'dividendYield' in info else "N/A")
        
        st.markdown("---")
        
        # Create tabs for different analyses
        tabs = st.tabs(["Price History", "Balance Sheet", "Income Statement", "Cash Flow", "AI Analysis"])
        
        # Tab 1: Price History
        with tabs[0]:
            st.markdown('<div class="section-header">Historical Price Chart (Last 90 Days)</div>', unsafe_allow_html=True)
            
            # Create Plotly figure for stock price
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1E88E5', width=2)
            ))
            
            fig.update_layout(
                title=f"{info.get('shortName', ticker)} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                margin=dict(l=0, r=0, t=50, b=0),
                hovermode="x unified",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display volume as a bar chart
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=hist.index,
                y=hist['Volume'],
                name='Volume',
                marker=dict(color='#0D47A1')
            ))
            
            fig_volume.update_layout(
                title="Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300,
                margin=dict(l=0, r=0, t=50, b=0),
                template="plotly_white"
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
        
        # Tab 2: Balance Sheet
        with tabs[1]:
            st.markdown('<div class="section-header">Balance Sheet</div>', unsafe_allow_html=True)
            if not data['balance_sheet'].empty:
                st.dataframe(data['balance_sheet'], use_container_width=True)
            else:
                st.warning("Balance sheet data not available for this stock.")
        
        # Tab 3: Income Statement
        with tabs[2]:
            st.markdown('<div class="section-header">Income Statement</div>', unsafe_allow_html=True)
            if not data['income_stmt'].empty:
                st.dataframe(data['income_stmt'], use_container_width=True)
            else:
                st.warning("Income statement data not available for this stock.")
        
        # Tab 4: Cash Flow
        with tabs[3]:
            st.markdown('<div class="section-header">Cash Flow Statement</div>', unsafe_allow_html=True)
            if not data['cash_flow'].empty:
                st.dataframe(data['cash_flow'], use_container_width=True)
            else:
                st.warning("Cash flow data not available for this stock.")
        
        # Tab 5: AI Analysis
        with tabs[4]:
            st.markdown('<div class="section-header">AI Stock Analysis</div>', unsafe_allow_html=True)
            
            if not api_key:
                st.warning("Please enter an API key in the sidebar to use AI analysis.")
            else:
                analyze_button = st.button("Analyze Stock")
                
                if analyze_button:
                    with st.spinner("Analyzing the stock... This may take a moment."):
                        if api_provider == "Claude":
                            analysis = analyze_with_claude(api_key, claude_model, data)
                        else:
                            analysis = analyze_with_openai(api_key, openai_model, data)
                        
                        st.markdown("### AI Analysis Results")
                        st.markdown(analysis)
    else:
        st.error(f"Could not find stock data for ticker: {ticker}")
else:
    st.info("Enter a ticker symbol in the sidebar to begin analysis.")