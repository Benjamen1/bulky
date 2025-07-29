import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

# Import our PDF extractor
from pdf_extractor import LifesumPDFExtractor

# Page config
st.set_page_config(
    page_title="Bulking Analysis Dashboard",
    page_icon="💪",
    layout="wide"
)

def load_weight_data():
    """Load weight data from data/daily_weight.csv"""
    weight_file = Path("data/daily_weight.csv")
    
    if not weight_file.exists():
        st.error(f"Weight file not found: {weight_file}")
        st.info("Please create daily_weight.csv in the data folder")
        return None
    
    try:
        df = pd.read_csv(weight_file)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Remove duplicates (keep most recent entry for each date)
        df = df.drop_duplicates(subset=['date'], keep='last')
        
        st.success(f"✅ Loaded {len(df)} weight records")
        return df
    except Exception as e:
        st.error(f"Error loading weight data: {e}")
        return None

def load_nutrition_data(nutrition_file=None):
    """Load nutrition data from CSV or extract from PDFs"""
    if nutrition_file:
        try:
            df = pd.read_csv(nutrition_file)
            df['date'] = pd.to_datetime(df['date']).dt.date
            return df.sort_values('date').reset_index(drop=True)
        except Exception as e:
            st.error(f"Error loading nutrition data: {e}")
            return None
    return None

def extract_pdfs_to_nutrition_data():
    """Extract nutrition data from PDFs in data/lifesum folder"""
    lifesum_folder = Path("data/lifesum")
    
    if not lifesum_folder.exists():
        st.error(f"Lifesum folder not found: {lifesum_folder}")
        st.info("Please create the folder and add your Lifesum PDF files")
        return None
    
    pdf_files = list(lifesum_folder.glob("*.pdf"))
    if not pdf_files:
        st.warning(f"No PDF files found in {lifesum_folder}")
        return None
    
    try:
        # Extract data
        extractor = LifesumPDFExtractor()
        extractor.process_folder(lifesum_folder)
        
        nutrition_df = extractor.get_daily_nutrition_df()
        
        if not nutrition_df.empty:
            # Remove duplicates (in case same date appears in multiple PDFs)
            nutrition_df = nutrition_df.drop_duplicates(subset=['date'], keep='last')
            st.success(f"✅ Extracted data from {len(pdf_files)} PDF files ({len(nutrition_df)} unique days)")
        
        return nutrition_df
        
    except Exception as e:
        st.error(f"Error extracting PDF data: {e}")
        return None

def merge_data(weight_df, nutrition_df):
    """Merge weight and nutrition data"""
    merged = pd.merge(nutrition_df, weight_df, on='date', how='inner')
    merged = merged.sort_values('date').reset_index(drop=True)
    
    # Calculate next-day weight change
    merged['next_day_weight'] = merged['weight_kg'].shift(-1)
    merged['weight_change'] = merged['next_day_weight'] - merged['weight_kg']
    
    # Remove last row (no next day data)
    merged = merged[:-1].copy()
    
    return merged

def calculate_rolling_averages(df, window=7):
    """Calculate rolling averages"""
    df = df.copy()
    df['weight_avg'] = df['weight_kg'].rolling(window=window, min_periods=1).mean()
    df['calories_avg'] = df['calories'].rolling(window=window, min_periods=1).mean()
    return df

def find_maintenance_calories(df):
    """Estimate maintenance calories from scatter plot data"""
    # Filter out extreme outliers
    q1_weight = df['weight_change'].quantile(0.25)
    q3_weight = df['weight_change'].quantile(0.75)
    iqr = q3_weight - q1_weight
    
    # Keep data within reasonable weight change range
    filtered_df = df[
        (df['weight_change'] >= q1_weight - 1.5 * iqr) & 
        (df['weight_change'] <= q3_weight + 1.5 * iqr)
    ].copy()
    
    # Find calories where weight change is closest to 0
    filtered_df['abs_weight_change'] = abs(filtered_df['weight_change'])
    
    # Get records with smallest weight changes
    stable_weight_data = filtered_df.nsmallest(
        max(5, len(filtered_df) // 4), 
        'abs_weight_change'
    )
    
    maintenance_estimate = stable_weight_data['calories'].mean()
    return maintenance_estimate

def create_scatter_plot(df):
    """Create calories vs weight change scatter plot"""
    fig = px.scatter(
        df, 
        x='calories', 
        y='weight_change',
        hover_data=['date'],
        title="Calories vs Next-Day Weight Change",
        labels={
            'calories': 'Daily Calories',
            'weight_change': 'Weight Change (kg)'
        },
        opacity=0.7
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", 
                  annotation_text="No Weight Change")
    
    # Estimate and show maintenance calories
    maintenance_cals = find_maintenance_calories(df)
    fig.add_vline(x=maintenance_cals, line_dash="dash", line_color="green",
                  annotation_text=f"Est. Maintenance: {maintenance_cals:.0f} cal")
    
    fig.update_layout(height=500)
    return fig, maintenance_cals

def create_rolling_average_plot(df):
    """Create rolling averages plot"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Weight trend
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['weight_avg'],
            mode='lines',
            name='7-Day Avg Weight',
            line=dict(color='red', width=3)
        ),
        secondary_y=False,
    )
    
    # Calories trend
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['calories_avg'],
            mode='lines',
            name='7-Day Avg Calories',
            line=dict(color='blue', width=2)
        ),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Weight (kg)", secondary_y=False)
    fig.update_yaxes(title_text="Calories", secondary_y=True)
    
    fig.update_layout(
        title="7-Day Rolling Averages - Weight vs Calories",
        height=500
    )
    
    return fig

def calculate_weekly_weight_change(df):
    """Calculate weekly weight change rate"""
    if len(df) < 7:
        return None
    
    # Get first and last weeks
    first_week = df.head(7)['weight_kg'].mean()
    last_week = df.tail(7)['weight_kg'].mean()
    
    weeks_elapsed = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 7
    
    if weeks_elapsed > 0:
        weekly_change = (last_week - first_week) / weeks_elapsed
        return weekly_change
    return None

def main():
    st.title("💪 Bulking Analysis Dashboard")
    st.markdown("Automatically loads your weight and nutrition data for analysis.")
    
    # Show data status in sidebar
    st.sidebar.header("📊 Data Status")
    
    # Auto-refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Load data automatically
    st.sidebar.write("**Weight Data:**")
    weight_df = load_weight_data()
    
    st.sidebar.write("**Nutrition Data:**")
    nutrition_data = extract_pdfs_to_nutrition_data()
    
    # Show file structure info
    st.sidebar.markdown("""
    **Expected file structure:**
    ```
    📁 Project Root
    ├── app.py
    └── data/
        ├── daily_weight.csv
        └── lifesum/
            ├── report1.pdf
            ├── report2.pdf
            └── ...
    ```
    """)
    
    # Main analysis
    if weight_df is not None and nutrition_data is not None and not nutrition_data.empty:
        # Load weight data
        if weight_df is not None and not nutrition_data.empty:
            # Merge datasets
            merged_df = merge_data(weight_df, nutrition_data)
            
            if len(merged_df) == 0:
                st.error("No matching dates found between weight and nutrition data. Check your date formats.")
                return
            
            # Calculate rolling averages
            rolling_df = calculate_rolling_averages(merged_df)
            
            # Display data overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Days", len(merged_df))
            
            with col2:
                avg_calories = merged_df['calories'].mean()
                st.metric("Avg Daily Calories", f"{avg_calories:.0f}")
            
            with col3:
                weekly_change = calculate_weekly_weight_change(merged_df)
                if weekly_change:
                    st.metric("Weekly Weight Change", f"{weekly_change:.2f} kg")
            
            # Create visualizations
            st.header("📊 Analysis Charts")
            
            # Scatter plot
            scatter_fig, maintenance_cals = create_scatter_plot(merged_df)
            st.plotly_chart(scatter_fig, use_container_width=True)
            
            # Rolling averages
            rolling_fig = create_rolling_average_plot(rolling_df)
            st.plotly_chart(rolling_fig, use_container_width=True)
            
            # Recommendations
            st.header("🎯 Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Maintenance Calories")
                st.metric("Estimated Maintenance", f"{maintenance_cals:.0f} calories/day")
                
                st.subheader("Bulking Targets")
                lean_bulk = maintenance_cals + 300
                aggressive_bulk = maintenance_cals + 500
                
                st.write(f"**Lean Bulk (slow gain):** {lean_bulk:.0f} calories/day")
                st.write(f"**Moderate Bulk:** {aggressive_bulk:.0f} calories/day")
            
            with col2:
                st.subheader("Current Progress")
                current_avg = merged_df['calories'].mean()
                if current_avg > maintenance_cals + 200:
                    st.success("✅ You're in a caloric surplus - should be gaining weight")
                elif current_avg < maintenance_cals - 200:
                    st.warning("⚠️ You're in a caloric deficit - may lose weight")
                else:
                    st.info("ℹ️ You're near maintenance - weight should be stable")
                
                if weekly_change:
                    if 0.2 <= weekly_change <= 0.5:  # 0.2-0.5 kg per week is ideal
                        st.success(f"✅ Good rate of gain: {weekly_change:.2f} kg/week")
                    elif weekly_change > 0.5:
                        st.warning(f"⚠️ Gaining too fast: {weekly_change:.2f} kg/week - consider reducing calories")
                    elif weekly_change < 0:
                        st.error(f"❌ Losing weight: {weekly_change:.2f} kg/week - increase calories")
                    else:
                        st.info(f"ℹ️ Slow gain: {weekly_change:.2f} kg/week - consider increasing calories")
            
            # Data preview
            with st.expander("📋 View Raw Data"):
                st.dataframe(merged_df[['date', 'weight_kg', 'calories', 'protein', 'weight_change']].head(10))
                
                if st.button("Download Merged Data as CSV"):
                    csv = merged_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="bulking_analysis_data.csv",
                        mime="text/csv"
                    )
    
    else:
        # Show what's missing
        missing_items = []
        if weight_df is None:
            missing_items.append("❌ data/daily_weight.csv")
        else:
            missing_items.append("✅ data/daily_weight.csv")
            
        if nutrition_data is None or nutrition_data.empty:
            missing_items.append("❌ Lifesum PDFs in data/lifesum/")
        else:
            missing_items.append("✅ Lifesum PDFs")
        
        st.info("📋 **Data Status:**\n" + "\n".join(missing_items))
        
        if weight_df is None or nutrition_data is None or nutrition_data.empty:
            st.markdown("**To get started:**")
            st.markdown("1. Create `data/daily_weight.csv`")
            st.markdown("2. Add your Lifesum PDFs to `data/lifesum/`")
            st.markdown("3. Click '🔄 Refresh Data' in the sidebar")
        
        # Show sample data format
        st.header("📋 Sample Data Formats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("data/daily_weight.csv")
            st.code("""date,weight_kg
2025-07-29,82.0
2025-07-30,82.2
2025-07-31,81.8""")
        
        with col2:
            st.subheader("Lifesum PDFs")
            st.write("Place your weekly PDF exports in:")
            st.code("data/lifesum/")
            st.write("The app will automatically:")
            st.write("- Extract daily calories & macros")
            st.write("- Remove duplicate dates")
            st.write("- Merge with your weight data")

if __name__ == "__main__":
    main()