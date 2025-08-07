import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta, date
import os
from pathlib import Path
import tempfile

# Import our modules
try:
    from pdf_extractor import LifesumPDFExtractor
    from sheets_handler import SheetsHandler
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure sheets_handler.py and pdf_extractor.py are in your repository")
    st.stop()

def get_sheets_handler():
    """Get SheetsHandler instance"""
    try:
        return SheetsHandler()
    except Exception as e:
        st.error(f"Failed to initialize Google Sheets connection: {e}")
        st.info("Check your Google Sheets credentials in Streamlit secrets")
        return None
def process_uploaded_pdf(uploaded_file):
    """Process uploaded PDF and extract nutrition data"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        # Extract data
        extractor = LifesumPDFExtractor()
        extractor.extract_pdf_data(tmp_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Get extracted data
        nutrition_df = extractor.get_daily_nutrition_df()
        
        return nutrition_df
        
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return pd.DataFrame()
    
# Page config
st.set_page_config(
    page_title="Bulking Analysis Dashboard",
    page_icon="💪",
    layout="wide"
)

def save_weight_data(weight_df):
    """Save weight data back to CSV"""
    try:
        weight_file = Path("data/daily_weight.csv")
        weight_df.to_csv(weight_file, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving weight data: {e}")
        return False

def add_weight_entry(weight_df, new_date, new_weight):
    """Add or update a weight entry"""
    # Convert date to match existing format
    if isinstance(new_date, date):
        new_date = new_date.strftime('%Y-%m-%d')
    elif isinstance(new_date, str):
        # Ensure it's in YYYY-MM-DD format
        try:
            parsed_date = datetime.strptime(new_date, '%Y-%m-%d').date()
            new_date = parsed_date.strftime('%Y-%m-%d')
        except:
            return None, "Invalid date format"
    
    # Check if date already exists
    existing_dates = weight_df['date'].astype(str).values
    
    if new_date in existing_dates:
        # Update existing entry
        weight_df.loc[weight_df['date'].astype(str) == new_date, 'weight_kg'] = new_weight
        return weight_df, f"Updated weight for {new_date}"
    else:
        # Add new entry
        new_row = pd.DataFrame({
            'date': [new_date], 
            'weight_kg': [new_weight]
        })
        weight_df = pd.concat([weight_df, new_row], ignore_index=True)
        # Sort by date
        weight_df['date'] = pd.to_datetime(weight_df['date']).dt.date
        weight_df = weight_df.sort_values('date').reset_index(drop=True)
        return weight_df, f"Added new weight entry for {new_date}"

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

def load_nutrition_data():
    """Load nutrition data from Google Sheets"""
    sheets = get_sheets_handler()
    
    if sheets is None:
        st.error("Cannot load nutrition data - Google Sheets connection failed")
        return pd.DataFrame()
    
    df = sheets.load_nutrition_data()
    
    if not df.empty:
        st.success(f"✅ Loaded {len(df)} nutrition records from Google Sheets")
    else:
        st.warning("No nutrition data found in Google Sheets")
    
    return df

def merge_data(weight_df, nutrition_df):
    """Merge weight and nutrition data"""
    merged = pd.merge(nutrition_df, weight_df, on='date', how='inner')
    merged = merged.sort_values('date').reset_index(drop=True)
    
    # Calculate macro-based calories
    merged['macro_calories'] = (merged['protein'] * 4) + (merged.get('fat_total', merged.get('fat_saturated', 0)) * 9) + (merged.get('carbs_total', 0) * 4)
    
    # Calculate next-day weight change
    # First, let's see what dates we have after the initial merge
    print(f"After initial merge: {len(merged)} rows, dates {merged['date'].min()} to {merged['date'].max()}")
    
    # Create a copy of weight data shifted by one day for easier merging
    weight_next_day = weight_df.copy()
    weight_next_day['date'] = weight_next_day['date'] - pd.Timedelta(days=1)
    weight_next_day = weight_next_day.rename(columns={'weight_kg': 'next_day_weight'})
    
    # Merge with next day weights
    merged = pd.merge(merged, weight_next_day[['date', 'next_day_weight']], on='date', how='left')
    
    # Calculate weight change
    merged['weight_change'] = merged['next_day_weight'] - merged['weight_kg']
    
    # Only keep rows where we have next day weight data
    merged = merged.dropna(subset=['next_day_weight']).copy()
    
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
    
    # Weight Entry Form
    st.sidebar.header("⚖️ Add Weight Entry")
    
    with st.sidebar.form("weight_entry_form"):
        # Default to today's date
        entry_date = st.date_input(
            "Date",
            value=date.today(),
            help="Select the date for this weight entry"
        )
        
        entry_weight = st.number_input(
            "Weight (kg)",
            min_value=40.0,
            max_value=200.0,
            value=80.0,
            step=0.1,
            help="Enter your weight in kilograms"
        )
        
        submitted = st.form_submit_button("💾 Save Weight")
        
        if submitted:
            result = save_weight_entry(entry_date, entry_weight)
            st.success(result)
            st.info("📝 Data saved to Google Sheets! Click 'Refresh Data' to see updates.")
    
    # PDF Upload Section
    st.sidebar.header("📄 Upload Nutrition PDF")
    
    uploaded_file = st.sidebar.file_uploader(
        "Drop your Lifesum PDF here",
        type=['pdf'],
        help="Upload your weekly Lifesum PDF export"
    )
    
    if uploaded_file is not None:
        # Show processing message
        with st.spinner("Processing PDF..."):
            nutrition_df = process_uploaded_pdf(uploaded_file)
            
            if not nutrition_df.empty:
                st.sidebar.success(f"✅ Extracted {len(nutrition_df)} days of nutrition data")
                
                # Show preview
                st.sidebar.subheader("📋 Extracted Data Preview")
                preview_df = nutrition_df[['date', 'calories', 'protein']].copy()
                st.sidebar.dataframe(preview_df, hide_index=True)
                
                # Save button
                if st.sidebar.button("💾 Save to Google Sheets"):
                    with st.spinner("Saving to Google Sheets..."):
                        result = save_nutrition_data(nutrition_df)
                        st.sidebar.success(result)
                        st.sidebar.info("📝 Click 'Refresh Data' to see updates in analysis")
            else:
                st.sidebar.error("Failed to extract data from PDF")
    
    # Show recent weight entries
    weight_df = load_weight_data()
    if weight_df is not None and len(weight_df) > 0:
        st.sidebar.subheader("📋 Recent Weight Entries")
        recent_entries = weight_df.tail(5)[['date', 'weight_kg']].copy()
        recent_entries['date'] = recent_entries['date'].astype(str)
        st.sidebar.dataframe(recent_entries, hide_index=True)
    
    # Load data automatically
    st.sidebar.write("**Data Status:**")
    nutrition_data = load_nutrition_data()
    
    # Debug information
    if weight_df is not None:
        st.sidebar.write(f"Weight records: {len(weight_df)} days")
        if not weight_df.empty:
            st.sidebar.write(f"Weight dates: {weight_df['date'].min()} to {weight_df['date'].max()}")
    
    if nutrition_data is not None and not nutrition_data.empty:
        st.sidebar.write(f"Nutrition records: {len(nutrition_data)} days") 
        st.sidebar.write(f"Nutrition dates: {nutrition_data['date'].min()} to {nutrition_data['date'].max()}")
    
    # Show connection status
    sheets = get_sheets_handler()
    if sheets is not None:
        sheet_info = sheets.get_sheet_info()
        
        if sheet_info['connected']:
            st.sidebar.success("🔗 Connected to Google Sheets")
        else:
            st.sidebar.error("❌ Google Sheets connection failed")
            st.sidebar.error(f"Error: {sheet_info.get('error', 'Unknown error')}")
    else:
        st.sidebar.error("❌ Could not initialize Google Sheets connection")
    
    # Show file structure info
    st.sidebar.markdown("""
    **Data Storage:**
    ```
    📊 Google Sheets: "Bulking Tracker"
    ├── daily_weight (sheet)
    └── daily_nutrition (sheet)
    ```
    **To update:** Use forms above or edit sheets directly
    """)
    
    # Main analysis
    if weight_df is not None and nutrition_data is not None and not nutrition_data.empty:
        # Load weight data
        if weight_df is not None and not nutrition_data.empty:
            # Merge datasets
            merged_df = merge_data(weight_df, nutrition_data)
            
            # Debug the merge process
            st.sidebar.write(f"**After merge:** {len(merged_df)} days")
            if not merged_df.empty:
                st.sidebar.write(f"Merged dates: {merged_df['date'].min()} to {merged_df['date'].max()}")
            
            if len(merged_df) == 0:
                st.error("No matching dates found between weight and nutrition data. Check your date formats.")
                return
            
            # Calculate rolling averages
            rolling_df = calculate_rolling_averages(merged_df)
            
            # Display data overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Days", len(merged_df))
            
            with col2:
                avg_calories = merged_df['calories'].mean()
                st.metric("Avg Daily Calories (Lifesum)", f"{avg_calories:.0f}")
            
            with col3:
                if 'macro_calories' in merged_df.columns:
                    avg_macro_calories = merged_df['macro_calories'].mean()
                    st.metric("Avg Daily Calories (Macro Calc)", f"{avg_macro_calories:.0f}")
                else:
                    st.metric("Macro Calculated Calories", "N/A")
            
            with col4:
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
                
                st.subheader("Recommended Macros")
                target_calories = lean_bulk
                
                # Updated macro targets
                protein_g = 250
                fat_g = 200
                carbs_g = 231
                
                # Calculate total calories from macros
                actual_target_calories = (protein_g * 4) + (fat_g * 9) + (carbs_g * 4)
                
                st.write(f"**Target Macros ({actual_target_calories:.0f} calories/day):**")
                st.write(f"• **Protein:** {protein_g}g ({protein_g*4} cal, {protein_g*4/actual_target_calories*100:.0f}%)")
                st.write(f"• **Fat:** {fat_g}g ({fat_g*9} cal, {fat_g*9/actual_target_calories*100:.0f}%)")
                st.write(f"• **Carbs:** {carbs_g}g ({carbs_g*4} cal, {carbs_g*4/actual_target_calories*100:.0f}%)")
                
                st.info("💡 **Simple Rules:**\n• 100g meat ≈ 25g protein, 15g fat\n• 1 egg ≈ 6g protein, 6g fat\n• 1 tbsp honey ≈ 15g carbs")
            
            with col2:
                st.subheader("Current Progress")
                current_avg = merged_df['calories'].mean()
                current_protein = merged_df['protein'].mean()
                
                # Calculate actual calories from macros if available
                if 'carbs_total' in merged_df.columns and 'fat_total' in merged_df.columns:
                    actual_calories = (current_protein * 4) + (merged_df['fat_total'].mean() * 9) + (merged_df['carbs_total'].mean() * 4)
                    st.write(f"**Lifesum shows:** {current_avg:.0f} calories/day")
                    st.write(f"**Macro calculation:** {actual_calories:.0f} calories/day")
                    st.info("💡 Trust macro calculation over Lifesum's calorie display")
                    comparison_calories = actual_calories
                else:
                    comparison_calories = current_avg
                
                if comparison_calories > maintenance_cals + 200:
                    st.success("✅ You're in a caloric surplus - should be gaining weight")
                elif comparison_calories < maintenance_cals - 200:
                    st.warning("⚠️ You're in a caloric deficit - may lose weight")
                else:
                    st.info("ℹ️ You're near maintenance - weight should be stable")
                
                if weekly_change:
                    if 0.2 <= weekly_change <= 0.5:  # 0.2-0.5 kg per week is ideal
                        st.success(f"✅ Good rate of gain: {weekly_change:.2f} kg/week")
                    elif weekly_change > 0.5:
                        st.warning(f"⚠️ Gaining too fast: {weekly_change:.2f} kg/week - consider reducing calories")
                    elif weekly_change < -0.1:
                        st.error(f"❌ Losing weight: {weekly_change:.2f} kg/week - increase calories")
                    else:
                        st.info(f"ℹ️ Slow gain: {weekly_change:.2f} kg/week - consider increasing calories slightly")
                
                st.subheader("Macro Comparison")
                if not merged_df.empty:
                    current_protein_avg = merged_df['protein'].mean()
                    
                    # Show current vs target macros
                    st.write(f"**Current Protein:** {current_protein_avg:.0f}g vs Target: {protein_g}g")
                    
                    protein_diff = current_protein_avg - protein_g
                    if abs(protein_diff) <= 20:
                        st.write("✅ Protein intake is on target")
                    elif protein_diff > 20:
                        st.write(f"⬆️ Protein is {protein_diff:.0f}g above target (good for bulking)")
                    else:
                        st.write(f"⬇️ Protein is {abs(protein_diff):.0f}g below target - add more meat/eggs")
                    
                    # Show macro breakdown if available
                    if 'carbs_total' in merged_df.columns and 'fat_total' in merged_df.columns:
                        current_fat = merged_df['fat_total'].mean()
                        current_carbs = merged_df['carbs_total'].mean()
                        
                        st.write(f"**Current Fat:** {current_fat:.0f}g vs Target: {fat_g}g")
                        st.write(f"**Current Carbs:** {current_carbs:.0f}g vs Target: {carbs_g}g")
            
            # Data preview
            with st.expander("📋 View Raw Data"):
                # Show key columns including macro calculation
                display_columns = ['date', 'weight_kg', 'calories', 'macro_calories', 'protein', 'weight_change']
                available_columns = [col for col in display_columns if col in merged_df.columns]
                
                # Add fat and carbs if available
                if 'fat_total' in merged_df.columns:
                    available_columns.insert(-1, 'fat_total')
                if 'carbs_total' in merged_df.columns:
                    available_columns.insert(-1, 'carbs_total')
                
                # Round the macro_calories for better display
                display_df = merged_df[available_columns].copy()
                if 'macro_calories' in display_df.columns:
                    display_df['macro_calories'] = display_df['macro_calories'].round(0)
                
                st.dataframe(display_df.head(10))
                
                # Show the difference between Lifesum calories and macro calculation
                if 'macro_calories' in merged_df.columns:
                    avg_lifesum = merged_df['calories'].mean()
                    avg_macro = merged_df['macro_calories'].mean()
                    difference = avg_macro - avg_lifesum
                    
                    st.info(f"📊 **Calorie Comparison:**\n"
                           f"• Lifesum Average: {avg_lifesum:.0f} cal/day\n"
                           f"• Macro Calculation: {avg_macro:.0f} cal/day\n"
                           f"• Difference: {difference:.0f} cal/day ({difference/avg_lifesum*100:.1f}%)")
                
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