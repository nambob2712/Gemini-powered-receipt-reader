"""
AI Receipt OCR - Premium Dark Theme
"""

import os
import csv
import calendar
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from receipt_ocr_system import ReceiptOCRSystem, ProcessingStatus

# Configuration
RECEIPTS_DIR = Path("saved_receipts")
HISTORY_CSV = Path("receipt_history.csv")
CSV_COLUMNS = ["Timestamp", "Date", "Total_Amount", "Currency", "Category", "Merchant", "Image_Path"]
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "gif", "webp"]
CURRENCY_SYMBOLS = {'JPY': '¥', 'USD': '$', 'EUR': '€', 'GBP': '£', 'CNY': '¥', 'KRW': '₩'}

# Theme Colors
COLORS = {
    'bg': '#060D10',
    'surface': '#27403E',
    'primary': '#21A691',
    'accent': '#87DF2C',
    'text': '#FFFFFF',
    'text_secondary': '#B4B4B2',
    'border': '#21A69133'
}

CUSTOM_CSS = """
<style>
    /* Global Styles */
    .stApp {
        background-color: #060D10;
    }
    
    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    h1 {
        background: linear-gradient(90deg, #21A691, #87DF2C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Body text */
    p, span, label, .stMarkdown {
        color: #B4B4B2 !important;
    }
    
    /* Cards/Surfaces */
    .card {
        background: #27403E;
        border: 1px solid #21A69133;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .card-glow {
        box-shadow: 0 0 20px #21A69120;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: #27403E;
        border: 1px solid #21A69133;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    
    [data-testid="stMetricLabel"] {
        color: #B4B4B2 !important;
        font-size: 0.875rem !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #21A691, #1a8a78);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px #21A69140;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #87DF2C, #6bc220);
        box-shadow: 0 6px 25px #87DF2C40;
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: #27403E;
        border: 2px dashed #21A69166;
        border-radius: 12px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #21A691;
        background: #27403E99;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #27403E;
        border: 1px solid #21A69133;
        border-radius: 8px;
        color: #B4B4B2;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #21A691, #1a8a78);
        color: #FFFFFF !important;
        border-color: transparent;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background: #27403E;
        border: 1px solid #21A69133;
        border-radius: 8px;
        color: #FFFFFF;
        padding: 0.75rem 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #21A691;
        box-shadow: 0 0 0 2px #21A69133;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: #27403E;
        border: 1px solid #21A69133;
        border-radius: 8px;
        color: #FFFFFF !important;
        font-weight: 500;
    }
    
    .streamlit-expanderContent {
        background: #27403E;
        border: 1px solid #21A69133;
        border-top: none;
        border-radius: 0 0 8px 8px;
    }
    
    /* DataFrames */
    .stDataFrame {
        background: #27403E;
        border-radius: 12px;
        overflow: hidden;
    }
    
    [data-testid="stDataFrame"] > div {
        background: #27403E;
    }
    
    /* Alerts */
    .stSuccess {
        background: #27403E;
        border-left: 4px solid #87DF2C;
        color: #87DF2C !important;
    }
    
    .stWarning {
        background: #27403E;
        border-left: 4px solid #F59E0B;
    }
    
    .stError {
        background: #27403E;
        border-left: 4px solid #EF4444;
    }
    
    .stInfo {
        background: #27403E;
        border-left: 4px solid #21A691;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #21A691, #87DF2C);
        border-radius: 4px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0a1a1a;
        border-right: 1px solid #21A69133;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    
    /* Divider */
    hr {
        border-color: #21A69133;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: transparent;
        border: 1px solid #21A691;
        color: #21A691;
    }
    
    .stDownloadButton > button:hover {
        background: #21A69120;
        border-color: #87DF2C;
        color: #87DF2C;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #21A691 !important;
    }
    
    /* Image container */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #21A69133;
    }
    
    /* Custom classes */
    .gradient-text {
        background: linear-gradient(90deg, #21A691, #87DF2C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #27403E, #1a2d2b);
        border: 1px solid #21A69133;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: #B4B4B2;
    }
    
    .accent-border {
        border-left: 3px solid #21A691;
        padding-left: 1rem;
    }
    
    .glow-accent {
        box-shadow: 0 0 30px #21A69130;
    }
</style>
"""


def init_storage():
    """Initialize storage directory and CSV file."""
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_CSV.exists():
        with open(HISTORY_CSV, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(CSV_COLUMNS)


def get_api_key() -> Optional[str]:
    """
    Get API key from Streamlit secrets, session state, or environment variables.
    
    Priority order:
    1. Streamlit secrets (secrets.toml) - GEMINI_API_KEY or GOOGLE_API_KEY
    2. Session state (user input in sidebar)
    3. Environment variable - GEMINI_API_KEY or GOOGLE_API_KEY
    
    Returns:
        API key string if found, None otherwise
    """
    # Try Streamlit secrets first (highest priority)
    try:
        if hasattr(st, 'secrets'):
            # Try GEMINI_API_KEY first (newer convention)
            if 'GEMINI_API_KEY' in st.secrets and st.secrets['GEMINI_API_KEY']:
                key = st.secrets['GEMINI_API_KEY']
                if key and key != "your_api_key_here":
                    return key
            # Fall back to GOOGLE_API_KEY (backward compatibility)
            if 'GOOGLE_API_KEY' in st.secrets and st.secrets['GOOGLE_API_KEY']:
                key = st.secrets['GOOGLE_API_KEY']
                if key and key != "your_api_key_here":
                    return key
    except Exception as e:
        # Silently handle secrets access errors (secrets might not be configured)
        pass
    
    # Try session state (user input in sidebar)
    session_key = st.session_state.get('api_key')
    if session_key:
        return session_key
    
    # Try environment variables (last resort)
    env_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if env_key:
        return env_key
    
    return None


@st.cache_resource
def get_ocr_system(api_key: str) -> ReceiptOCRSystem:
    """Get cached OCR system instance."""
    if not api_key:
        raise ValueError("API key is required")
    return ReceiptOCRSystem(api_key=api_key)


def load_history() -> pd.DataFrame:
    """Load receipt history from CSV."""
    if not HISTORY_CSV.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)
    try:
        df = pd.read_csv(HISTORY_CSV, encoding='utf-8')
        if not df.empty:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Total_Amount'] = pd.to_numeric(df['Total_Amount'], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame(columns=CSV_COLUMNS)


def get_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly spending summary from receipt data."""
    if df.empty:
        return pd.DataFrame()
    
    df_valid = df[df['Date'].notna() & df['Total_Amount'].notna()].copy()
    if df_valid.empty:
        return pd.DataFrame()
    
    # Extract month and year
    df_valid['Year'] = df_valid['Date'].dt.year
    df_valid['Month'] = df_valid['Date'].dt.month
    df_valid['Month_Name'] = df_valid['Date'].dt.strftime('%B')
    df_valid['Year_Month'] = df_valid['Date'].dt.to_period('M')
    
    # Group by month
    monthly = df_valid.groupby(['Year', 'Month', 'Month_Name']).agg({
        'Total_Amount': ['sum', 'mean', 'count'],
        'Merchant': 'nunique',
        'Category': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Other'
    }).reset_index()
    
    # Flatten column names
    monthly.columns = ['Year', 'Month', 'Month_Name', 'Total_Spent', 'Avg_Receipt', 'Receipt_Count', 'Unique_Stores', 'Top_Category']
    
    # Sort by date (newest first)
    monthly = monthly.sort_values(['Year', 'Month'], ascending=[False, False])
    
    return monthly


def get_receipts_for_month(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """Get all receipts for a specific month."""
    if df.empty:
        return pd.DataFrame()
    
    df_valid = df[df['Date'].notna()].copy()
    mask = (df_valid['Date'].dt.year == year) & (df_valid['Date'].dt.month == month)
    return df_valid[mask].sort_values('Date', ascending=False)


def get_category_breakdown_for_month(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """Get category-wise spending breakdown for a specific month."""
    month_df = get_receipts_for_month(df, year, month)
    if month_df.empty:
        return pd.DataFrame()
    
    breakdown = month_df.groupby('Category').agg({
        'Total_Amount': 'sum',
        'Merchant': 'count'
    }).reset_index()
    breakdown.columns = ['Category', 'Amount', 'Count']
    breakdown = breakdown.sort_values('Amount', ascending=False)
    return breakdown


def get_available_months(df: pd.DataFrame) -> List[Tuple[int, int, str]]:
    """Get list of available months with data as (year, month, label) tuples."""
    if df.empty:
        return []
    
    df_valid = df[df['Date'].notna()].copy()
    if df_valid.empty:
        return []
    
    df_valid['Year'] = df_valid['Date'].dt.year
    df_valid['Month'] = df_valid['Date'].dt.month
    
    months = df_valid.groupby(['Year', 'Month']).size().reset_index(name='count')
    months = months.sort_values(['Year', 'Month'], ascending=[False, False])
    
    result = []
    for _, row in months.iterrows():
        year, month = int(row['Year']), int(row['Month'])
        label = f"{calendar.month_name[month]} {year}"
        result.append((year, month, label))
    
    return result


def get_ytd_summary(df: pd.DataFrame, year: int) -> Dict:
    """Get year-to-date summary for a specific year."""
    if df.empty:
        return {'total': 0, 'count': 0, 'avg': 0, 'stores': 0}
    
    df_valid = df[(df['Date'].notna()) & (df['Total_Amount'].notna())].copy()
    year_df = df_valid[df_valid['Date'].dt.year == year]
    
    if year_df.empty:
        return {'total': 0, 'count': 0, 'avg': 0, 'stores': 0}
    
    return {
        'total': year_df['Total_Amount'].sum(),
        'count': len(year_df),
        'avg': year_df['Total_Amount'].mean(),
        'stores': year_df['Merchant'].nunique()
    }


def check_duplicate(date, amount, merchant, currency="JPY") -> tuple[bool, Optional[pd.Series]]:
    """Check if receipt already exists in database."""
    df = load_history()
    if df.empty:
        return False, None
    
    mask = pd.Series([True] * len(df))
    
    if date:
        date_str = date.strftime("%Y-%m-%d")
        mask &= df['Date'].dt.strftime('%Y-%m-%d').fillna('') == date_str
    else:
        mask &= df['Date'].isna()
    
    if amount is not None:
        tol = 0.5 if currency == 'JPY' else 0.01
        mask &= df['Total_Amount'].between(amount - tol, amount + tol)
    else:
        mask &= df['Total_Amount'].isna()
    
    if merchant:
        mask &= df['Merchant'].fillna('').str.lower().str.strip() == merchant.lower().strip()
    else:
        mask &= df['Merchant'].fillna('') == ''
    
    mask &= df['Currency'] == currency
    
    matches = df[mask]
    return (True, matches.iloc[0]) if not matches.empty else (False, None)


def save_receipt(uploaded_file, result) -> bool:
    """Save receipt data after checking for duplicates."""
    data = result.data
    
    is_dup, existing = check_duplicate(data.date, data.total_amount, data.merchant_name, data.currency)
    
    if is_dup:
        st.warning("⚠️ **Duplicate Receipt Detected**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="accent-border">', unsafe_allow_html=True)
            st.markdown("**New Receipt**")
            st.write(f"📅 {data.date.strftime('%Y-%m-%d') if data.date else 'N/A'}")
            st.write(f"💰 ¥{data.total_amount:,.0f}" if data.total_amount else "💰 N/A")
            st.write(f"🏪 {data.merchant_name or 'N/A'}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="accent-border">', unsafe_allow_html=True)
            st.markdown("**Existing Record**")
            if existing is not None:
                st.write(f"📅 {pd.to_datetime(existing['Date']).strftime('%Y-%m-%d') if pd.notna(existing['Date']) else 'N/A'}")
                st.write(f"💰 ¥{existing['Total_Amount']:,.0f}" if pd.notna(existing['Total_Amount']) else "💰 N/A")
                st.write(f"🏪 {existing['Merchant'] if pd.notna(existing['Merchant']) else 'N/A'}")
            st.markdown('</div>', unsafe_allow_html=True)
        st.error("❌ Receipt not saved (duplicate)")
        return False
    
    uploaded_file.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        ext = '.jpg'
    image_path = RECEIPTS_DIR / f"{timestamp}{ext}"
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    with open(HISTORY_CSV, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(),
            data.date.strftime("%Y-%m-%d") if data.date else "",
            f"{data.total_amount:.2f}" if data.total_amount else "",
            data.currency,
            data.category.value,
            data.merchant_name or "",
            str(image_path)
        ])
    
    st.success("✅ Receipt saved successfully!")
    return True


def format_amount(amount, currency="JPY"):
    """Format currency amount."""
    if amount is None:
        return "N/A"
    sym = CURRENCY_SYMBOLS.get(currency, currency)
    return f"{sym}{amount:,.0f}" if currency == 'JPY' else f"{sym}{amount:.2f}"


def render_sidebar():
    """Render sidebar."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Get API key using the centralized function
        api_key = get_api_key()
        
        # Check if key is from secrets
        key_from_secrets = False
        try:
            if hasattr(st, 'secrets'):
                if ('GEMINI_API_KEY' in st.secrets and st.secrets['GEMINI_API_KEY'] and 
                    st.secrets['GEMINI_API_KEY'] != "your_api_key_here"):
                    key_from_secrets = True
                elif ('GOOGLE_API_KEY' in st.secrets and st.secrets['GOOGLE_API_KEY'] and 
                      st.secrets['GOOGLE_API_KEY'] != "your_api_key_here"):
                    key_from_secrets = True
        except Exception:
            pass
        
        if api_key:
            if key_from_secrets:
                st.success("✅ API Key Active (from secrets.toml)")
                st.info("💡 Key loaded from `.streamlit/secrets.toml`")
            elif os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'):
                st.success("✅ API Key Active (from environment)")
                st.info("💡 Key loaded from environment variable")
            else:
                st.success("✅ API Key Active")
                st.session_state['api_key'] = api_key
        else:
            st.markdown("**API Key**")
            api_key_input = st.text_input(
                "Enter key",
                value=st.session_state.get('api_key', ''),
                type="password",
                placeholder="AIza...",
                label_visibility="collapsed",
                help="Enter your Google Gemini API key or configure it in .streamlit/secrets.toml"
            )
            st.session_state['api_key'] = api_key_input
            if api_key_input:
                st.success("✅ Key configured")
            else:
                st.info("🔑 [Get free API key](https://aistudio.google.com/apikey)")
                st.info("💡 **Tip:** Add your key to `.streamlit/secrets.toml` to auto-load it")
        
        st.divider()
        
        st.markdown("### 📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            img_count = len(list(RECEIPTS_DIR.glob("*"))) if RECEIPTS_DIR.exists() else 0
            st.metric("Images", img_count)
        with col2:
            df = load_history()
            st.metric("Records", len(df))
        
        if not df.empty:
            total = df['Total_Amount'].sum()
            st.metric("Total Spent", f"¥{total:,.0f}")
        
        st.divider()
        
        st.markdown("### 🤖 AI Engine")
        st.markdown("""
        <div style="background: #27403E; border-radius: 8px; padding: 1rem; border: 1px solid #21A69133;">
            <span style="color: #87DF2C;">●</span> <strong style="color: #FFFFFF;">Gemini 2.0 Flash</strong><br>
            <span style="color: #B4B4B2; font-size: 0.85rem;">High accuracy • Multi-language</span>
        </div>
        """, unsafe_allow_html=True)


def render_upload_tab():
    """Render upload tab."""
    api_key = get_api_key()
    if not api_key:
        st.markdown("""
        <div class="card glow-accent" style="text-align: center; padding: 3rem;">
            <h3 style="color: #FFFFFF;">🔑 API Key Required</h3>
            <p style="color: #B4B4B2;">Configure your Google AI API key in the sidebar to start.</p>
            <a href="https://aistudio.google.com/apikey" target="_blank" 
               style="color: #21A691; text-decoration: none;">Get free API key →</a>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("Upload a receipt image for AI-powered data extraction.")
    
    uploaded = st.file_uploader(
        "Drop receipt image here",
        type=SUPPORTED_FORMATS,
        label_visibility="collapsed"
    )
    
    if uploaded:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("#### 📷 Receipt Image")
            image = Image.open(uploaded)
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔍 Analysis")
            
            if st.button("⚡ Analyze Receipt", type="primary", use_container_width=True):
                with st.spinner(""):
                    st.markdown("""
                    <div style="text-align: center; padding: 2rem;">
                        <p style="color: #21A691;">Processing with AI...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    try:
                        result = get_ocr_system(api_key).process_image(image)
                        
                        if result.status == ProcessingStatus.SUCCESS:
                            st.success("✅ Extraction complete")
                        elif result.status == ProcessingStatus.PARTIAL:
                            st.warning("⚠️ Partial extraction")
                        else:
                            st.error("❌ Extraction failed")
                        
                        if result.data:
                            display_results(result)
                            save_receipt(uploaded, result)
                        
                        for err in result.errors:
                            st.error(err)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.markdown("""
                <div class="card" style="text-align: center;">
                    <p style="color: #B4B4B2; margin: 0;">Click <strong style="color: #21A691;">Analyze Receipt</strong> to extract data</p>
                </div>
                """, unsafe_allow_html=True)


def display_results(result):
    """Display extraction results."""
    d = result.data
    currency = d.currency
    
    # Main metrics
    cols = st.columns(3)
    with cols[0]:
        st.metric("📅 Date", d.date.strftime("%Y-%m-%d") if d.date else "N/A")
    with cols[1]:
        st.metric("💰 Total", format_amount(d.total_amount, currency))
    with cols[2]:
        st.metric("🏷️ Category", d.category.value)
    
    # Confidence bar
    conf = d.confidence_score
    conf_color = "#87DF2C" if conf >= 0.9 else "#21A691" if conf >= 0.7 else "#F59E0B"
    st.markdown(f"""
    <div style="background: #27403E; border-radius: 8px; padding: 0.75rem 1rem; margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #B4B4B2;">AI Confidence</span>
            <span style="color: {conf_color}; font-weight: 600;">{conf:.0%}</span>
        </div>
        <div style="background: #060D10; border-radius: 4px; height: 8px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #21A691, {conf_color}); 
                        width: {conf*100}%; height: 100%; border-radius: 4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Details expander
    with st.expander("📋 Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Merchant:** {d.merchant_name or 'N/A'}")
            st.markdown(f"**Address:** {d.merchant_address or 'N/A'}")
            st.markdown(f"**Phone:** {d.merchant_phone or 'N/A'}")
        with col2:
            st.markdown(f"**Subtotal:** {format_amount(d.subtotal, currency)}")
            st.markdown(f"**Tax:** {format_amount(d.tax_amount, currency)} ({d.tax_rate or 'N/A'})")
            st.markdown(f"**Payment:** {d.payment_method or 'N/A'}")
    
    # Line items
    if d.line_items:
        with st.expander(f"🛒 Items ({len(d.line_items)})", expanded=False):
            items_df = pd.DataFrame([
                {
                    "Item": i.description,
                    "Qty": int(i.quantity),
                    "Price": format_amount(i.total_price, currency)
                } for i in d.line_items
            ])
            st.dataframe(items_df, hide_index=True, use_container_width=True)


def render_analytics_tab():
    """Render analytics tab with monthly categorization integration."""
    df = load_history()
    
    if df.empty:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #FFFFFF;">📊 No Data Yet</h3>
            <p style="color: #B4B4B2;">Upload receipts to see analytics.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # ========== FILTER SECTION ==========
    st.markdown("#### 🗓️ Filter by Period")
    
    col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 2])
    
    # Get available months (based on receipt Date, not Timestamp)
    available_months = get_available_months(df)
    
    with col_filter1:
        filter_options = ["All Time"] + [m[2] for m in available_months]
        selected_filter = st.selectbox(
            "Period Filter",
            filter_options,
            label_visibility="collapsed"
        )
    
    # Apply filter
    if selected_filter == "All Time":
        filtered_df = df.copy()
        filter_label = "All Time"
    else:
        selected_idx = filter_options.index(selected_filter) - 1
        selected_year, selected_month, filter_label = available_months[selected_idx]
        filtered_df = get_receipts_for_month(df, selected_year, selected_month)
    
    df_valid = filtered_df[filtered_df['Total_Amount'].notna()].copy()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== SUMMARY METRICS ==========
    st.markdown(f"#### 📈 Overview — {filter_label}")
    cols = st.columns(4)
    
    total_spent = df_valid['Total_Amount'].sum() if not df_valid.empty else 0
    avg_spent = df_valid['Total_Amount'].mean() if not df_valid.empty else 0
    receipt_count = len(df_valid)
    store_count = df_valid['Merchant'].nunique() if not df_valid.empty else 0
    
    metrics = [
        ("💰 Total Spent", f"¥{total_spent:,.0f}"),
        ("📊 Average", f"¥{avg_spent:,.0f}"),
        ("🧾 Receipts", str(receipt_count)),
        ("🏪 Stores", str(store_count))
    ]
    
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== MONTHLY COMPARISON BAR CHART (only for All Time) ==========
    if selected_filter == "All Time" and available_months:
        st.markdown("#### 📅 Monthly Spending Comparison")
        
        monthly_summary = get_monthly_summary(df)
        
        if not monthly_summary.empty:
            # Sort by year and month for proper display
            monthly_sorted = monthly_summary.sort_values(['Year', 'Month'])
            
            # Create readable labels
            monthly_sorted['Label'] = monthly_sorted.apply(
                lambda x: f"{x['Month_Name'][:3]} {int(x['Year'])}", axis=1
            )
            
            fig = go.Figure(data=[go.Bar(
                x=monthly_sorted['Label'],
                y=monthly_sorted['Total_Spent'],
                marker=dict(
                    color='#21A691',
                    line=dict(color='#87DF2C', width=2)
                ),
                text=[f"¥{v:,.0f}" for v in monthly_sorted['Total_Spent']],
                textposition='outside',
                textfont=dict(color='#B4B4B2', size=10)
            )])
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    gridcolor='rgba(33, 166, 145, 0.13)', 
                    color='#B4B4B2',
                    title=''
                ),
                yaxis=dict(
                    gridcolor='rgba(33, 166, 145, 0.13)', 
                    color='#B4B4B2',
                    title='Amount (¥)'
                ),
                margin=dict(t=40, b=40, l=60, r=20),
                height=320
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly summary cards (compact grid)
            st.markdown("#### 📊 Monthly Breakdown")
            
            months_per_row = 4
            rows = [monthly_summary.iloc[i:i+months_per_row] for i in range(0, len(monthly_summary), months_per_row)]
            
            for row_data in rows:
                cols = st.columns(months_per_row)
                for idx, (_, month_data) in enumerate(row_data.iterrows()):
                    with cols[idx]:
                        st.markdown(f"""
                        <div style="background: #27403E; border: 1px solid #21A69133; border-radius: 10px; 
                                    padding: 1rem; text-align: center; margin-bottom: 0.75rem;">
                            <div style="color: #B4B4B2; font-size: 0.8rem; margin-bottom: 0.25rem;">
                                {month_data['Month_Name'][:3]} {int(month_data['Year'])}
                            </div>
                            <div style="color: #87DF2C; font-size: 1.25rem; font-weight: 700;">
                                ¥{month_data['Total_Spent']:,.0f}
                            </div>
                            <div style="color: #21A691; font-size: 0.75rem; margin-top: 0.25rem;">
                                {int(month_data['Receipt_Count'])} receipts
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== CATEGORY & DAILY CHARTS ==========
    st.markdown("#### 📉 Breakdown")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**By Category**")
        if not df_valid.empty:
            cat_data = df_valid.groupby('Category')['Total_Amount'].sum().reset_index()
            fig = go.Figure(data=[go.Pie(
                labels=cat_data['Category'],
                values=cat_data['Total_Amount'],
                hole=0.5,
                marker=dict(colors=['#21A691', '#87DF2C', '#1a8a78', '#6bc220', '#27403E', '#4fd1c5']),
                textinfo='percent+label',
                textfont=dict(color='#FFFFFF', size=10)
            )])
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
                height=280
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**By Receipt Date**")
        if not df_valid.empty:
            # FIX: Use receipt Date, not Timestamp
            df_chart = df_valid[df_valid['Date'].notna()].copy()
            if not df_chart.empty:
                df_chart['Day'] = df_chart['Date'].dt.date  # FIXED: Was Timestamp
                daily = df_chart.groupby('Day')['Total_Amount'].sum().reset_index()
                daily = daily.sort_values('Day')
                
                fig = go.Figure(data=[go.Bar(
                    x=daily['Day'],
                    y=daily['Total_Amount'],
                    marker=dict(
                        color='#21A691',
                        line=dict(color='#87DF2C', width=1)
                    )
                )])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='rgba(33, 166, 145, 0.13)', color='#B4B4B2'),
                    yaxis=dict(gridcolor='rgba(33, 166, 145, 0.13)', color='#B4B4B2'),
                    margin=dict(t=20, b=40, l=40, r=20),
                    height=280
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No dated receipts available")
    
    # ========== HISTORY TABLE ==========
    st.markdown("#### 📜 Receipt History")
    
    display_df = filtered_df.copy()
    display_df['Upload_Date'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['Receipt_Date'] = display_df['Date'].dt.strftime('%Y-%m-%d').fillna('N/A')
    display_df['Amount'] = display_df['Total_Amount'].apply(
        lambda x: f"¥{x:,.0f}" if pd.notna(x) else "N/A"
    )
    
    # Reorder columns for clarity
    display_cols = ['Receipt_Date', 'Merchant', 'Category', 'Amount', 'Currency', 'Upload_Date']
    display_df = display_df[[c for c in display_cols if c in display_df.columns]]
    
    st.dataframe(
        display_df.iloc[::-1],
        hide_index=True,
        use_container_width=True,
        height=300
    )
    
    st.download_button(
        "📥 Export CSV",
        filtered_df.to_csv(index=False),
        f"receipts_{filter_label.replace(' ', '_')}_{datetime.now():%Y%m%d}.csv",
        "text/csv"
    )



def render_monthly_tab():
    """Render the monthly categorization tab."""
    df = load_history()
    
    if df.empty:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #FFFFFF;">📅 No Monthly Data Yet</h3>
            <p style="color: #B4B4B2;">Upload receipts to see monthly categorization.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Get available months
    available_months = get_available_months(df)
    
    if not available_months:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #FFFFFF;">📅 No Dated Receipts</h3>
            <p style="color: #B4B4B2;">Receipts need valid dates for monthly categorization.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Month/Year Filter Controls
    st.markdown("#### 🗓️ Select Period")
    
    col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 1])
    
    with col_filter1:
        # View mode selector
        view_mode = st.selectbox(
            "View Mode",
            ["Monthly View", "Year-to-Date Overview"],
            label_visibility="collapsed"
        )
    
    current_year = datetime.now().year
    available_years = sorted(set(m[0] for m in available_months), reverse=True)
    
    if view_mode == "Year-to-Date Overview":
        with col_filter2:
            selected_year = st.selectbox(
                "Select Year",
                available_years,
                label_visibility="collapsed"
            )
        
        # YTD Summary
        st.markdown(f"#### 📈 Year-to-Date Summary: {selected_year}")
        ytd = get_ytd_summary(df, selected_year)
        
        cols = st.columns(4)
        ytd_metrics = [
            ("💰 Total Spent", f"¥{ytd['total']:,.0f}"),
            ("🧾 Receipts", str(ytd['count'])),
            ("📊 Average", f"¥{ytd['avg']:,.0f}" if ytd['count'] > 0 else "¥0"),
            ("🏪 Stores", str(ytd['stores']))
        ]
        
        for col, (label, value) in zip(cols, ytd_metrics):
            with col:
                st.markdown(f"""
                <div class="stat-card glow-accent">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value" style="color: #21A691;">{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Monthly Breakdown Cards for the Year
        st.markdown(f"#### 📊 Monthly Breakdown - {selected_year}")
        
        monthly_summary = get_monthly_summary(df)
        year_months = monthly_summary[monthly_summary['Year'] == selected_year]
        
        if year_months.empty:
            st.info(f"No receipt data for {selected_year}")
        else:
            # Create monthly cards in a grid
            months_per_row = 3
            rows = [year_months.iloc[i:i+months_per_row] for i in range(0, len(year_months), months_per_row)]
            
            for row_data in rows:
                cols = st.columns(months_per_row)
                for idx, (_, month_data) in enumerate(row_data.iterrows()):
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="card glow-accent" style="min-height: 180px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                <h4 style="color: #FFFFFF; margin: 0;">{month_data['Month_Name']}</h4>
                                <span style="background: #21A691; color: #FFFFFF; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem;">
                                    {int(month_data['Receipt_Count'])} receipts
                                </span>
                            </div>
                            <div style="font-size: 1.75rem; font-weight: 700; color: #87DF2C; margin-bottom: 0.5rem;">
                                ¥{month_data['Total_Spent']:,.0f}
                            </div>
                            <div style="display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.85rem; color: #B4B4B2;">
                                <span>📊 Avg: ¥{month_data['Avg_Receipt']:,.0f}</span>
                                <span>🏪 {int(month_data['Unique_Stores'])} stores</span>
                            </div>
                            <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #21A69133;">
                                <span style="color: #21A691; font-size: 0.85rem;">
                                    🏷️ Top: {month_data['Top_Category']}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Year trend chart
            st.markdown(f"#### 📉 Spending Trend - {selected_year}")
            
            year_months_sorted = year_months.sort_values('Month')
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=year_months_sorted['Month_Name'],
                y=year_months_sorted['Total_Spent'],
                mode='lines+markers',
                line=dict(color='#21A691', width=3),
                marker=dict(size=10, color='#87DF2C'),
                fill='tozeroy',
                fillcolor='rgba(33, 166, 145, 0.2)'
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(33, 166, 145, 0.13)', color='#B4B4B2', title=''),
                yaxis=dict(gridcolor='rgba(33, 166, 145, 0.13)', color='#B4B4B2', title='Amount (¥)'),
                margin=dict(t=20, b=40, l=60, r=20),
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Monthly View
        with col_filter2:
            # Month selector
            month_options = [m[2] for m in available_months]
            selected_month_label = st.selectbox(
                "Select Month",
                month_options,
                label_visibility="collapsed"
            )
            
            # Get corresponding year and month
            selected_idx = month_options.index(selected_month_label)
            selected_year, selected_month, _ = available_months[selected_idx]
        
        # Month Header
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 1rem; margin: 1.5rem 0;">
            <h3 style="color: #FFFFFF; margin: 0;">📅 {selected_month_label}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Get month data
        month_receipts = get_receipts_for_month(df, selected_year, selected_month)
        month_total = month_receipts['Total_Amount'].sum() if not month_receipts.empty else 0
        month_count = len(month_receipts)
        month_avg = month_receipts['Total_Amount'].mean() if not month_receipts.empty else 0
        
        # Summary stats
        cols = st.columns(4)
        month_metrics = [
            ("💰 Total Spent", f"¥{month_total:,.0f}"),
            ("🧾 Receipts", str(month_count)),
            ("📊 Average", f"¥{month_avg:,.0f}"),
            ("🏪 Stores", str(month_receipts['Merchant'].nunique()) if not month_receipts.empty else "0")
        ]
        
        for col, (label, value) in zip(cols, month_metrics):
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Two-column layout: Category breakdown and Receipt list
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("#### 🏷️ Category Breakdown")
            
            cat_breakdown = get_category_breakdown_for_month(df, selected_year, selected_month)
            
            if not cat_breakdown.empty:
                # Pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=cat_breakdown['Category'],
                    values=cat_breakdown['Amount'],
                    hole=0.5,
                    marker=dict(colors=['#21A691', '#87DF2C', '#1a8a78', '#6bc220', '#27403E', '#4fd1c5']),
                    textinfo='percent+label',
                    textfont=dict(color='#FFFFFF', size=11),
                    insidetextorientation='horizontal'
                )])
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=280
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Category list
                for _, cat_row in cat_breakdown.iterrows():
                    pct = (cat_row['Amount'] / month_total * 100) if month_total > 0 else 0
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; 
                                padding: 0.75rem 1rem; background: #27403E; border-radius: 8px; 
                                margin-bottom: 0.5rem; border-left: 3px solid #21A691;">
                        <div>
                            <span style="color: #FFFFFF; font-weight: 500;">{cat_row['Category']}</span>
                            <span style="color: #B4B4B2; font-size: 0.85rem; margin-left: 0.5rem;">
                                ({int(cat_row['Count'])} items)
                            </span>
                        </div>
                        <div style="text-align: right;">
                            <span style="color: #87DF2C; font-weight: 600;">¥{cat_row['Amount']:,.0f}</span>
                            <span style="color: #B4B4B2; font-size: 0.85rem; margin-left: 0.5rem;">
                                {pct:.1f}%
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No categorized receipts for this month")
        
        with col2:
            st.markdown("#### 🧾 Receipt Details")
            
            if not month_receipts.empty:
                for _, receipt in month_receipts.iterrows():
                    date_str = receipt['Date'].strftime('%Y-%m-%d') if pd.notna(receipt['Date']) else 'N/A'
                    amount = receipt['Total_Amount'] if pd.notna(receipt['Total_Amount']) else 0
                    merchant = receipt['Merchant'] if pd.notna(receipt['Merchant']) else 'Unknown'
                    category = receipt['Category'] if pd.notna(receipt['Category']) else 'Other'
                    
                    st.markdown(f"""
                    <div class="card" style="padding: 1rem; margin-bottom: 0.75rem;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <div style="color: #FFFFFF; font-weight: 600; font-size: 1.1rem;">
                                    {merchant}
                                </div>
                                <div style="color: #B4B4B2; font-size: 0.85rem; margin-top: 0.25rem;">
                                    📅 {date_str} • 🏷️ {category}
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="color: #87DF2C; font-weight: 700; font-size: 1.25rem;">
                                    ¥{amount:,.0f}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No receipts for this month")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Month comparison with previous month
        prev_month = selected_month - 1 if selected_month > 1 else 12
        prev_year = selected_year if selected_month > 1 else selected_year - 1
        prev_receipts = get_receipts_for_month(df, prev_year, prev_month)
        prev_total = prev_receipts['Total_Amount'].sum() if not prev_receipts.empty else 0
        
        if prev_total > 0:
            change = ((month_total - prev_total) / prev_total) * 100
            change_color = "#87DF2C" if change <= 0 else "#EF4444"
            change_icon = "📉" if change <= 0 else "📈"
            prev_month_name = calendar.month_name[prev_month]
            
            st.markdown(f"""
            <div class="card glow-accent" style="padding: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #B4B4B2;">Compared to {prev_month_name} {prev_year}</span>
                    </div>
                    <div>
                        <span style="color: {change_color}; font-weight: 600; font-size: 1.1rem;">
                            {change_icon} {change:+.1f}%
                        </span>
                        <span style="color: #B4B4B2; margin-left: 0.5rem;">
                            (¥{prev_total:,.0f} → ¥{month_total:,.0f})
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="AI Receipt OCR",
        page_icon="🧾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_storage()
    
    # Header
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1>🧾 AI Receipt OCR</h1>
        <p style="color: #B4B4B2; font-size: 1.1rem;">
            Intelligent receipt scanning powered by <span style="color: #21A691;">Google Gemini</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    render_sidebar()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["⚡ Scan", "� Monthly", "�📊 Analytics"])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_monthly_tab()
    
    with tab3:
        render_analytics_tab()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; margin-top: 2rem; border-top: 1px solid #21A69133;">
        <p style="color: #B4B4B2; font-size: 0.85rem; margin: 0;">
            Built with <span style="color: #21A691;">Streamlit</span> • 
            Powered by <span style="color: #87DF2C;">Gemini AI</span>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
