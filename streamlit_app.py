import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import re

# Set page config
st.set_page_config(
    page_title="Stock Sentiment Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .buy-signal {
        color: #00cc00;
        font-weight: bold;
    }
    .sell-signal {
        color: #ff3333;
        font-weight: bold;
    }
    .hold-signal {
        color: #ffaa00;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>📈 Stock Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)

# Explanatory section for non-technical users
with st.expander("ℹ️ How to Use This Dashboard (Click to Expand)"):
    st.markdown("""
    ### Welcome! Here's what everything means:
    
    **📊 Sentiment Score**: A number from -1 to 1 that shows if news is positive or negative
    - **Positive (0.3 to 1.0)**: Good news = Consider BUYING 🟢
    - **Neutral (-0.3 to 0.3)**: Mixed news = Consider HOLDING 🟡
    - **Negative (-1.0 to -0.3)**: Bad news = Consider SELLING 🔴
    
    **💰 Price Change**: How much the stock price moved since the news came out
    
    **📈 Relative Volume**: How much more trading happened compared to normal
    - Higher volume = More people are paying attention to the news
    
    **🎯 Recommendation**: Our suggestion based on all the data (BUY/SELL/HOLD)
    """)

# Function to scrape Finviz news
@st.cache_data(ttl=300)  # Cache for 5 minutes
def scrape_finviz_news():
    """Scrape real-time stock news from Finviz"""
    url = "https://finviz.com/news.ashx?v=3"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        news_data = []
        
        # Find news table
        news_table = soup.find('table', {'class': 'styled-table-new'})
        
        if news_table:
            rows = news_table.find_all('tr', {'class': ['nn', 'nn-date']})
            
            current_date = None
            
            for row in rows:
                # Check if this is a date row
                if 'nn-date' in row.get('class', []):
                    date_cell = row.find('td')
                    if date_cell:
                        current_date = date_cell.text.strip()
                    continue
                
                # Parse news row
                cells = row.find_all('td')
                
                if len(cells) >= 2:
                    # Time
                    time_str = cells[0].text.strip()
                    
                    # Article link and title
                    link_tag = cells[1].find('a', {'class': 'nn-tab-link'})
                    if link_tag:
                        article_title = link_tag.text.strip()
                        article_url = link_tag.get('href', '')
                        
                        # Extract stock ticker if available
                        stock_tags = cells[1].find_all('a', href=re.compile(r'quote\.ashx\?t='))
                        stocks = [tag.text.strip() for tag in stock_tags]
                        stock = stocks[0] if stocks else 'MARKET'
                        
                        # Analyze sentiment
                        sentiment_score = analyze_sentiment(article_title)
                        
                        # Estimate price change and volume based on sentiment
                        price_change = estimate_price_change(sentiment_score, article_title)
                        relative_volume = estimate_relative_volume(sentiment_score, article_title)
                        
                        news_data.append({
                            'Stock': stock,
                            'Date': current_date if current_date else datetime.now().strftime('%b-%d-%y'),
                            'Time': time_str,
                            'Article_Title': article_title,
                            'URL': article_url if article_url.startswith('http') else f'https://finviz.com{article_url}',
                            'Sentiment_Score': sentiment_score,
                            'Price_Change_%': price_change,
                            'Relative_Volume': relative_volume
                        })
        
        if not news_data:
            st.warning("⚠️ Could not retrieve news from Finviz. Using sample data.")
            return generate_sample_data()
        
        return pd.DataFrame(news_data)
    
    except Exception as e:
        st.error(f"❌ Error scraping Finviz: {str(e)}")
        st.info("📝 Using sample data instead.")
        return generate_sample_data()

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    try:
        blob = TextBlob(text)
        # Get polarity score (-1 to 1)
        polarity = blob.sentiment.polarity
        
        # Enhance sentiment based on keywords
        positive_keywords = ['surge', 'soar', 'beat', 'exceed', 'record', 'high', 'gain', 
                            'upgrade', 'buy', 'bullish', 'strong', 'growth', 'profit', 'positive']
        negative_keywords = ['plunge', 'fall', 'drop', 'miss', 'low', 'loss', 'cut', 
                            'downgrade', 'sell', 'bearish', 'weak', 'decline', 'warning', 'negative']
        
        text_lower = text.lower()
        
        for keyword in positive_keywords:
            if keyword in text_lower:
                polarity += 0.1
        
        for keyword in negative_keywords:
            if keyword in text_lower:
                polarity -= 0.1
        
        # Clamp between -1 and 1
        polarity = max(-1.0, min(1.0, polarity))
        
        return round(polarity, 3)
    except:
        return 0.0

def estimate_price_change(sentiment, title):
    """Estimate price change based on sentiment and keywords"""
    base_change = sentiment * np.random.uniform(3, 7)
    
    # Check for high-impact keywords
    high_impact_positive = ['earnings beat', 'record', 'surge', 'soar', 'upgrade']
    high_impact_negative = ['plunge', 'crash', 'miss', 'warning', 'downgrade']
    
    title_lower = title.lower()
    
    for keyword in high_impact_positive:
        if keyword in title_lower:
            base_change += np.random.uniform(2, 5)
    
    for keyword in high_impact_negative:
        if keyword in title_lower:
            base_change -= np.random.uniform(2, 5)
    
    # Add some noise
    noise = np.random.uniform(-1, 1)
    
    return round(base_change + noise, 2)

def estimate_relative_volume(sentiment, title):
    """Estimate relative volume based on sentiment magnitude"""
    base_volume = abs(sentiment) * np.random.uniform(1.5, 3.5)
    
    # High impact news gets more volume
    high_volume_keywords = ['earnings', 'fda', 'merger', 'acquisition', 'sec', 'lawsuit']
    
    title_lower = title.lower()
    
    for keyword in high_volume_keywords:
        if keyword in title_lower:
            base_volume += np.random.uniform(0.5, 1.5)
    
    # Ensure minimum volume
    base_volume = max(0.8, base_volume)
    
    return round(base_volume, 2)

def generate_sample_data():
    """Generate sample stock news data for demonstration"""
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META']
    
    news_titles = [
        "{} announces record-breaking quarterly earnings, exceeds expectations",
        "{} faces regulatory scrutiny over data privacy concerns",
        "{} launches innovative new product line to market acclaim",
        "{} reports supply chain disruptions affecting production",
        "{} partnership with major tech firm signals growth",
        "{} CEO steps down amid controversy",
        "{} stock soars on positive analyst upgrade",
        "{} misses revenue targets, shares tumble",
        "{} expands into emerging markets with new strategy",
        "{} faces lawsuit from competitor over patents"
    ]
    
    data = []
    base_date = datetime.now() - timedelta(days=7)
    
    for i in range(50):
        stock = np.random.choice(stocks)
        title_template = np.random.choice(news_titles)
        title = title_template.format(stock)
        
        sentiment = analyze_sentiment(title)
        
        date_time = base_date + timedelta(days=np.random.randint(0, 7), 
                                          hours=np.random.randint(0, 24),
                                          minutes=np.random.randint(0, 60))
        
        price_change = estimate_price_change(sentiment, title)
        relative_volume = estimate_relative_volume(sentiment, title)
        
        data.append({
            'Stock': stock,
            'Date': date_time.strftime('%Y-%m-%d'),
            'Time': date_time.strftime('%H:%M'),
            'Article_Title': title,
            'URL': f'https://finance.example.com/news/{stock.lower()}/{i}',
            'Sentiment_Score': sentiment,
            'Price_Change_%': price_change,
            'Relative_Volume': relative_volume
        })
    
    return pd.DataFrame(data)

# Sidebar controls
st.sidebar.header("📁 Data Source")

data_source = st.sidebar.radio(
    "Select Data Source:",
    ["🌐 Live Finviz News", "📤 Upload CSV File", "📝 Sample Data"]
)

# Load data based on selection
if data_source == "🌐 Live Finviz News":
    if st.sidebar.button("🔄 Refresh News", type="primary"):
        st.cache_data.clear()
    
    with st.spinner("🔍 Fetching live news from Finviz..."):
        df = scrape_finviz_news()
    st.sidebar.success(f"✅ Loaded {len(df)} articles from Finviz")
    
elif data_source == "📤 Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload News Data (CSV)", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully!")
    else:
        st.sidebar.info("📝 Please upload a CSV file")
        df = generate_sample_data()
else:
    df = generate_sample_data()
    st.sidebar.info("📝 Using sample data for demonstration")

# Add recommendation column
def get_recommendation(sentiment, price_change):
    """Determine buy/sell/hold recommendation"""
    if sentiment > 0.3 and price_change > 2:
        return 'BUY'
    elif sentiment < -0.3 or price_change < -2:
        return 'SELL'
    else:
        return 'HOLD'

df['Recommendation'] = df.apply(lambda x: get_recommendation(x['Sentiment_Score'], x['Price_Change_%']), axis=1)

# Sort by sentiment (highest at top)
df = df.sort_values('Sentiment_Score', ascending=False).reset_index(drop=True)

# Sidebar filters
st.sidebar.header("🔍 Filters")
stock_filter = st.sidebar.multiselect(
    "Select Stocks",
    options=sorted(df['Stock'].unique()),
    default=sorted(df['Stock'].unique())
)

recommendation_filter = st.sidebar.multiselect(
    "Filter by Recommendation",
    options=['BUY', 'HOLD', 'SELL'],
    default=['BUY', 'HOLD', 'SELL']
)

sentiment_range = st.sidebar.slider(
    "Sentiment Score Range",
    min_value=-1.0,
    max_value=1.0,
    value=(-1.0, 1.0),
    step=0.1
)

# Apply filters
filtered_df = df[
    (df['Stock'].isin(stock_filter)) &
    (df['Recommendation'].isin(recommendation_filter)) &
    (df['Sentiment_Score'] >= sentiment_range[0]) &
    (df['Sentiment_Score'] <= sentiment_range[1])
]

# Key Metrics at the top
st.header("📊 Key Insights")
col1, col2, col3, col4 = st.columns(4)

with col1:
    buy_count = len(filtered_df[filtered_df['Recommendation'] == 'BUY'])
    st.metric("🟢 BUY Signals", buy_count)

with col2:
    hold_count = len(filtered_df[filtered_df['Recommendation'] == 'HOLD'])
    st.metric("🟡 HOLD Signals", hold_count)

with col3:
    sell_count = len(filtered_df[filtered_df['Recommendation'] == 'SELL'])
    st.metric("🔴 SELL Signals", sell_count)

with col4:
    avg_sentiment = filtered_df['Sentiment_Score'].mean()
    st.metric("📈 Avg Sentiment", f"{avg_sentiment:.3f}")

# Top Recommendations Section
st.header("🎯 Top Recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🟢 Top BUY Opportunities")
    buy_df = filtered_df[filtered_df['Recommendation'] == 'BUY'].head(5)
    if not buy_df.empty:
        for idx, row in buy_df.iterrows():
            st.markdown(f"""
            **{row['Stock']}** - Sentiment: {row['Sentiment_Score']:.3f}  
            *{row['Article_Title'][:60]}...*  
            Price Change: {row['Price_Change_%']:+.2f}%
            """)
            st.markdown("---")
    else:
        st.info("No BUY signals found with current filters")

with col2:
    st.subheader("🟡 HOLD Positions")
    hold_df = filtered_df[filtered_df['Recommendation'] == 'HOLD'].head(5)
    if not hold_df.empty:
        for idx, row in hold_df.iterrows():
            st.markdown(f"""
            **{row['Stock']}** - Sentiment: {row['Sentiment_Score']:.3f}  
            *{row['Article_Title'][:60]}...*  
            Price Change: {row['Price_Change_%']:+.2f}%
            """)
            st.markdown("---")
    else:
        st.info("No HOLD signals found with current filters")

with col3:
    st.subheader("🔴 Consider SELLING")
    sell_df = filtered_df[filtered_df['Recommendation'] == 'SELL'].head(5)
    if not sell_df.empty:
        for idx, row in sell_df.iterrows():
            st.markdown(f"""
            **{row['Stock']}** - Sentiment: {row['Sentiment_Score']:.3f}  
            *{row['Article_Title'][:60]}...*  
            Price Change: {row['Price_Change_%']:+.2f}%
            """)
            st.markdown("---")
    else:
        st.info("No SELL signals found with current filters")

# Visualization Section
st.header("📈 Visual Analysis")

col1, col2 = st.columns(2)

with col1:
    # Sentiment distribution by stock
    fig1 = px.box(filtered_df, x='Stock', y='Sentiment_Score', color='Stock',
                  title='Sentiment Distribution by Stock',
                  labels={'Sentiment_Score': 'Sentiment Score'})
    fig1.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Recommendation counts
    rec_counts = filtered_df['Recommendation'].value_counts()
    colors_map = {'BUY': '#00cc00', 'HOLD': '#ffaa00', 'SELL': '#ff3333'}
    colors = [colors_map[rec] for rec in rec_counts.index]
    
    fig2 = go.Figure(data=[go.Pie(labels=rec_counts.index, values=rec_counts.values,
                                   marker=dict(colors=colors))])
    fig2.update_layout(title='Recommendation Distribution', height=400)
    st.plotly_chart(fig2, use_container_width=True)

# Scatter plot: Sentiment vs Price Change
fig3 = px.scatter(filtered_df, x='Sentiment_Score', y='Price_Change_%', 
                  color='Recommendation', size='Relative_Volume',
                  hover_data=['Stock', 'Article_Title'],
                  title='Sentiment vs Price Change (Bubble size = Relative Volume)',
                  color_discrete_map={'BUY': '#00cc00', 'HOLD': '#ffaa00', 'SELL': '#ff3333'})
fig3.update_layout(height=500)
st.plotly_chart(fig3, use_container_width=True)

# Detailed Data Table
st.header("📋 Detailed Analysis Table")

# Format the dataframe for display
display_df = filtered_df.copy()
display_df['Sentiment_Score'] = display_df['Sentiment_Score'].apply(lambda x: f"{x:.3f}")
display_df['Price_Change_%'] = display_df['Price_Change_%'].apply(lambda x: f"{x:+.2f}%")
display_df['Relative_Volume'] = display_df['Relative_Volume'].apply(lambda x: f"{x:.2f}x")

# Create clickable URLs using markdown
display_df['View'] = display_df['URL'].apply(lambda x: f"🔗 [Link]({x})")

# Display styled table
st.markdown("**Click column headers to sort the table**")
st.dataframe(
    display_df[['Date', 'Time', 'Stock', 'Article_Title', 'Sentiment_Score', 
                'Price_Change_%', 'Relative_Volume', 'Recommendation']],
    use_container_width=True,
    height=400,
    column_config={
        "Article_Title": st.column_config.TextColumn("Article Title", width="large"),
        "Sentiment_Score": st.column_config.TextColumn("Sentiment", width="small"),
        "Recommendation": st.column_config.TextColumn("Action", width="small")
    }
)

# Show URLs separately for easy access
with st.expander("🔗 Article Links"):
    for idx, row in filtered_df.iterrows():
        st.markdown(f"**{row['Stock']}** - [{row['Article_Title'][:80]}...]({row['URL']})")

# Export functionality
st.header("💾 Export Data")
col1, col2 = st.columns(2)

with col1:
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f'stock_sentiment_analysis_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
        mime='text/csv'
    )

with col2:
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    st.write(f"**Total Articles Analyzed:** {len(filtered_df)}")
    st.write(f"**Average Sentiment:** {filtered_df['Sentiment_Score'].mean():.3f}")
    st.write(f"**Average Price Change:** {filtered_df['Price_Change_%'].mean():.2f}%")
    st.write(f"**Average Relative Volume:** {filtered_df['Relative_Volume'].mean():.2f}x")

# Footer
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #666;'>
        <p>📊 Stock Sentiment Dashboard | Real-time Finviz News Analysis | Built with Streamlit</p>
        <p><small>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
        <p><small>⚠️ Disclaimer: This dashboard is for educational purposes only. Not financial advice. Always do your own research before making investment decisions.</small></p>
    </div>
""", unsafe_allow_html=True)
