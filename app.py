# streamlit/app.py
"""
Enhanced Bulking Analysis Dashboard
Now with automated PDF processing from Google Drive
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta, date
import gspread
from google.oauth2.service_account import Credentials

# Page config
st.set_page_config(
    page_title="Bulking Analysis Dashboard",
    page_icon="💪",
    layout="wide"
)

# Cache the Google Sheets connection
@st.cache_resource
def get_sheets_client():
    """Initialize Google Sheets client"""
    try:
        credentials_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        gc = gspread.authorize(creds)
        return gc
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {e}")
        return None

# Cache data with TTL
@st.cache_data(ttl=300)  # 5 minute cache
def load_all_data():
    """Load all data from Google Sheets"""
    gc = get_sheets_client()
    if not gc:
        return None, None, None, None, None
    
    try:
        spreadsheet = gc.open("Bulking Tracker")
        
        # Load each sheet
        weight_df = pd.DataFrame(spreadsheet.worksheet('daily_weight').get_all_records())
        nutrition_df = pd.DataFrame(spreadsheet.worksheet('daily_nutrition').get_all_records())
        weekly_df = pd.DataFrame(spreadsheet.worksheet('weekly_averages').get_all_records())
        processed_df = pd.DataFrame(spreadsheet.worksheet('processed_files').get_all_records())
        settings_df = pd.DataFrame(spreadsheet.worksheet('settings').get_all_records())
        
        # Process dates
        if not weight_df.empty:
            weight_df['date'] = pd.to_datetime(weight_df['date']).dt.date
            weight_df = weight_df.sort_values('date').reset_index(drop=True)
        
        if not nutrition_df.empty:
            nutrition_df['date'] = pd.to_datetime(nutrition_df['date']).dt.date
            nutrition_df = nutrition_df.sort_values('date').reset_index(drop=True)
        
        return weight_df, nutrition_df, weekly_df, processed_df, settings_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

def save_weight_entry(date_val, weight_kg, body_fat=None, notes=""):
    """Save a new weight entry"""
    gc = get_sheets_client()
    if not gc:
        return "Failed to connect to Google Sheets"
    
    try:
        spreadsheet = gc.open("Bulking Tracker")
        worksheet = spreadsheet.worksheet('daily_weight')
        
        # Format date
        if isinstance(date_val, date):
            date_str = date_val.strftime('%Y-%m-%d')
        else:
            date_str = str(date_val)
        
        # Check if date already exists
        existing_data = worksheet.get_all_records()
        existing_dates = [r['date'] for r in existing_data]
        
        row_data = [
            date_str,
            float(weight_kg),
            float(body_fat) if body_fat else '',
            notes,
            datetime.now().isoformat()
        ]
        
        if date_str in existing_dates:
            # Update existing row
            for i, row in enumerate(existing_data, start=2):
                if row['date'] == date_str:
                    worksheet.update(f'A{i}:E{i}', [row_data])
                    return f"Updated weight for {date_str}"
        else:
            # Add new row
            worksheet.append_row(row_data)
            return f"Added weight entry for {date_str}"
            
    except Exception as e:
        return f"Error: {e}"

def merge_data(weight_df, nutrition_df):
    """Merge weight and nutrition data with next-day weight change"""
    if weight_df.empty or nutrition_df.empty:
        return pd.DataFrame()
    
    # Merge on date
    merged = pd.merge(nutrition_df, weight_df, on='date', how='inner')
    merged = merged.sort_values('date').reset_index(drop=True)
    
    # Calculate macro-based calories
    merged['macro_calories'] = (
        (merged['protein'] * 4) + 
        (merged['fat_total'] * 9) + 
        (merged['carbs_total'] * 4)
    )
    
    # Calculate next-day weight change
    weight_next = weight_df.copy()
    weight_next['date'] = weight_next['date'] - pd.Timedelta(days=1)
    weight_next = weight_next.rename(columns={'weight_kg': 'next_day_weight'})
    
    merged = pd.merge(merged, weight_next[['date', 'next_day_weight']], on='date', how='left')
    merged['weight_change'] = merged['next_day_weight'] - merged['weight_kg']
    
    # Only keep rows with next day weight
    merged = merged.dropna(subset=['next_day_weight']).copy()
    
    return merged

def calculate_rolling_averages(df, window=7):
    """Calculate rolling averages"""
    df = df.copy()
    df['weight_avg'] = df['weight_kg'].rolling(window=window, min_periods=1).mean()
    df['calories_avg'] = df['calories'].rolling(window=window, min_periods=1).mean()
    df['protein_avg'] = df['protein'].rolling(window=window, min_periods=1).mean()
    return df

def find_maintenance_calories(df):
    """Estimate maintenance calories from data"""
    if df.empty:
        return None
    
    # Filter outliers
    q1 = df['weight_change'].quantile(0.25)
    q3 = df['weight_change'].quantile(0.75)
    iqr = q3 - q1
    
    filtered = df[
        (df['weight_change'] >= q1 - 1.5 * iqr) & 
        (df['weight_change'] <= q3 + 1.5 * iqr)
    ].copy()
    
    # Find calories where weight change is near zero
    filtered['abs_change'] = abs(filtered['weight_change'])
    stable_data = filtered.nsmallest(max(5, len(filtered) // 4), 'abs_change')
    
    return stable_data['calories'].mean() if not stable_data.empty else None

def create_dashboard_header():
    """Create the dashboard header with key metrics"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("💪 Bulking Analysis Dashboard")
        st.markdown("*Automated tracking with Google Drive + Sheets*")

def create_sidebar(settings_df):
    """Create sidebar with controls and data entry"""
    st.sidebar.header("⚙️ Dashboard Controls")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Get settings
    settings = {row['setting_name']: row['value'] for row in settings_df.to_dict('records')} if not settings_df.empty else {}
    
    # Display targets
    st.sidebar.markdown("### 🎯 Current Targets")
    st.sidebar.info(f"""
    **Calories:** {settings.get('target_calories', '3724')} kcal  
    **Protein:** {settings.get('target_protein', '250')}g  
    **Carbs:** {settings.get('target_carbs', '231')}g  
    **Fat:** {settings.get('target_fat', '200')}g  
    **Weekly Goal:** {settings.get('weekly_goal_kg', '0.35')} kg/week
    """)
    
    # Weight entry form
    st.sidebar.markdown("---")
    st.sidebar.header("⚖️ Add Weight Entry")
    
    with st.sidebar.form("weight_entry"):
        entry_date = st.date_input("Date", value=date.today())
        
        col1, col2 = st.columns(2)
        with col1:
            weight = st.number_input("Weight (kg)", 60.0, 120.0, 80.0, 0.1)
        with col2:
            body_fat = st.number_input("Body Fat %", 5.0, 30.0, 15.0, 0.1)
        
        notes = st.text_input("Notes (optional)")
        
        if st.form_submit_button("💾 Save Weight", use_container_width=True):
            result = save_weight_entry(entry_date, weight, body_fat, notes)
            st.success(result)
            st.cache_data.clear()
            st.rerun()

def create_metrics_row(merged_df, weekly_df, settings):
    """Create the main metrics row"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "📊 Total Days", 
            len(merged_df),
            f"{len(merged_df.tail(7))} this week"
        )
    
    with col2:
        if not merged_df.empty:
            current_weight = merged_df['weight_kg'].iloc[-1]
            week_ago_weight = merged_df['weight_kg'].iloc[-7] if len(merged_df) >= 7 else merged_df['weight_kg'].iloc[0]
            weight_change = current_weight - week_ago_weight
            st.metric(
                "⚖️ Current Weight",
                f"{current_weight:.1f} kg",
                f"{weight_change:+.2f} kg/week"
            )
    
    with col3:
        if not merged_df.empty:
            avg_cals = merged_df.tail(7)['calories'].mean()
            target_cals = float(settings.get('target_calories', 3724))
            diff = avg_cals - target_cals
            st.metric(
                "🔥 Avg Calories (7d)",
                f"{avg_cals:.0f}",
                f"{diff:+.0f} vs target"
            )
    
    with col4:
        if not merged_df.empty:
            avg_protein = merged_df.tail(7)['protein'].mean()
            target_protein = float(settings.get('target_protein', 250))
            diff = avg_protein - target_protein
            st.metric(
                "🥩 Avg Protein (7d)",
                f"{avg_protein:.0f}g",
                f"{diff:+.0f}g vs target"
            )
    
    with col5:
        maintenance = find_maintenance_calories(merged_df)
        if maintenance:
            current_avg = merged_df.tail(7)['calories'].mean() if len(merged_df) >= 7 else merged_df['calories'].mean()
            surplus = current_avg - maintenance
            st.metric(
                "📈 Daily Surplus",
                f"{surplus:+.0f} kcal",
                f"Maint: {maintenance:.0f}"
            )

def create_scatter_plot(df):
    """Create calories vs weight change scatter plot"""
    fig = px.scatter(
        df, 
        x='calories', 
        y='weight_change',
        hover_data=['date', 'protein', 'carbs_total', 'fat_total'],
        title="Calories vs Next-Day Weight Change",
        labels={
            'calories': 'Daily Calories',
            'weight_change': 'Weight Change (kg)'
        },
        color='protein',
        color_continuous_scale='Viridis',
        opacity=0.7
    )
    
    # Add trend line
    z = np.polyfit(df['calories'], df['weight_change'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['calories'].min(), df['calories'].max(), 100)
    y_trend = p(x_trend)
    
    fig.add_trace(go.Scatter(
        x=x_trend, 
        y=y_trend,
        mode='lines',
        name='Trend',
        line=dict(color='red', dash='dash')
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="No Weight Change")
    
    # Add maintenance calories line
    maintenance = find_maintenance_calories(df)
    if maintenance:
        fig.add_vline(x=maintenance, line_dash="dash", line_color="green",
                      annotation_text=f"Est. Maintenance: {maintenance:.0f} cal")
    
    fig.update_layout(height=500)
    return fig

def create_trends_plot(df):
    """Create multi-panel trends plot"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Weight Trend', 'Calorie Intake', 'Protein Intake', 'Weekly Rate of Change'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Weight trend
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['weight_kg'], mode='markers', 
                  name='Daily Weight', opacity=0.3),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['weight_avg'], mode='lines',
                  name='7-Day Avg', line=dict(width=3)),
        row=1, col=1
    )
    
    # Calorie intake
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['calories'], mode='markers',
                  name='Daily Calories', opacity=0.3),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['calories_avg'], mode='lines',
                  name='7-Day Avg', line=dict(width=3)),
        row=1, col=2
    )
    
    # Protein intake
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['protein'], mode='markers',
                  name='Daily Protein', opacity=0.3),
        row=2, col=1
    )
    if 'protein_avg' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['protein_avg'], mode='lines',
                      name='7-Day Avg', line=dict(width=3)),
            row=2, col=1
        )
    
    # Weekly rate of change
    weekly_changes = []
    dates = []
    for i in range(7, len(df)):
        week_change = (df.iloc[i]['weight_kg'] - df.iloc[i-7]['weight_kg'])
        weekly_changes.append(week_change)
        dates.append(df.iloc[i]['date'])
    
    if weekly_changes:
        fig.add_trace(
            go.Bar(x=dates, y=weekly_changes, name='Weekly Change'),
            row=2, col=2
        )
        
        # Add target zone
        fig.add_hrect(y0=0.2, y1=0.5, fillcolor="green", opacity=0.2,
                     line_width=0, row=2, col=2)
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Weight (kg)", row=1, col=1)
    fig.update_yaxes(title_text="Calories", row=1, col=2)
    fig.update_yaxes(title_text="Protein (g)", row=2, col=1)
    fig.update_yaxes(title_text="Weight Change (kg)", row=2, col=2)
    
    fig.update_layout(height=800, showlegend=False)
    return fig

def main():
    # Create header
    create_dashboard_header()
    
    # Load all data
    weight_df, nutrition_df, weekly_df, processed_df, settings_df = load_all_data()
    
    if weight_df is None:
        st.error("❌ Failed to connect to Google Sheets. Check your credentials.")
        st.stop()
    
    # Create sidebar with settings
    if settings_df is not None:
        create_sidebar(settings_df)
    else:
        st.sidebar.error("Could not load settings")
    
    # Get settings dict
    settings = {row['setting_name']: row['value'] for row in settings_df.to_dict('records')} if settings_df is not None and not settings_df.empty else {}
    
    # Data status
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**📊 Weight Records:** {len(weight_df) if not weight_df.empty else 0} days")
    
    with col2:
        st.info(f"**🥗 Nutrition Records:** {len(nutrition_df) if not nutrition_df.empty else 0} days")
    
    with col3:
        if processed_df is not None and not processed_df.empty:
            successful = len(processed_df[processed_df['status'] == 'success'])
            st.info(f"**📄 PDFs Processed:** {successful} files")
        else:
            st.info("**📄 PDFs Processed:** 0 files")
    
    # Check if we have data to analyze
    if weight_df.empty or nutrition_df.empty:
        st.warning("⚠️ Not enough data for analysis. Please ensure you have both weight and nutrition data.")
        
        st.markdown("### 📝 Getting Started")
        st.markdown("""
        1. **Add weight data**: Use the sidebar form to add daily weight entries
        2. **Upload PDFs to Google Drive**: Place your Lifesum PDFs in the configured folder
        3. **Wait for processing**: PDFs are automatically processed daily at 8 AM UTC
        4. **Manual trigger**: You can manually trigger processing from GitHub Actions
        """)
        
        if settings:
            folder_id = settings.get('drive_folder_id', 'Not configured')
            if folder_id != 'YOUR_FOLDER_ID_HERE':
                st.success(f"✅ Google Drive folder configured: `{folder_id}`")
            else:
                st.error("❌ Please configure your Google Drive folder ID in the settings sheet")
        
        st.stop()
    
    # Merge data
    merged_df = merge_data(weight_df, nutrition_df)
    
    if merged_df.empty:
        st.warning("⚠️ No matching dates between weight and nutrition data")
        st.stop()
    
    # Calculate rolling averages
    merged_df = calculate_rolling_averages(merged_df)
    
    # Display metrics
    create_metrics_row(merged_df, weekly_df, settings)
    
    # Main visualizations
    st.markdown("---")
    st.header("📊 Analysis")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🎯 Correlation", "📋 Data", "🔍 Insights"])
    
    with tab1:
        trends_fig = create_trends_plot(merged_df)
        st.plotly_chart(trends_fig, use_container_width=True)
    
    with tab2:
        scatter_fig = create_scatter_plot(merged_df)
        st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Correlation Analysis")
        corr_cols = ['calories', 'protein', 'carbs_total', 'fat_total', 'weight_change']
        available_cols = [col for col in corr_cols if col in merged_df.columns]
        if len(available_cols) > 1:
            corr_matrix = merged_df[available_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           color_continuous_scale='RdBu', zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Recent Data")
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            days_to_show = st.slider("Days to display", 7, len(merged_df), min(30, len(merged_df)))
        with col2:
            show_macros = st.checkbox("Show macro breakdown", True)
        
        # Prepare display dataframe
        display_cols = ['date', 'weight_kg', 'calories', 'protein']
        if show_macros:
            display_cols.extend(['carbs_total', 'fat_total'])
        display_cols.append('weight_change')
        
        display_df = merged_df[display_cols].tail(days_to_show).copy()
        display_df['date'] = display_df['date'].astype(str)
        display_df = display_df.round(2)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = merged_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv,
            file_name=f"bulking_data_{date.today()}.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.subheader("🔍 Insights & Recommendations")
        
        # Calculate key insights
        maintenance = find_maintenance_calories(merged_df)
        current_avg = merged_df.tail(7)['calories'].mean()
        current_protein = merged_df.tail(7)['protein'].mean()
        
        if len(merged_df) >= 14:
            week1_weight = merged_df.tail(14)['weight_kg'].iloc[0:7].mean()
            week2_weight = merged_df.tail(7)['weight_kg'].mean()
            weekly_rate = week2_weight - week1_weight
        else:
            weekly_rate = None
        
        # Display insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Current Status")
            
            if maintenance:
                surplus = current_avg - maintenance
                st.metric("Estimated Maintenance", f"{maintenance:.0f} kcal/day")
                st.metric("Current Surplus", f"{surplus:+.0f} kcal/day")
                
                if surplus > 500:
                    st.warning("⚠️ Large surplus - may lead to excessive fat gain")
                elif surplus > 200:
                    st.success("✅ Good surplus for lean bulking")
                elif surplus > 0:
                    st.info("ℹ️ Small surplus - very lean gains")
                else:
                    st.error("❌ In a deficit - need more calories to bulk")
            
            if weekly_rate is not None:
                st.metric("Weekly Weight Change", f"{weekly_rate:.2f} kg/week")
                
                target_rate = float(settings.get('weekly_goal_kg', 0.35))
                if 0.2 <= weekly_rate <= 0.5:
                    st.success(f"✅ Good rate of gain")
                elif weekly_rate > 0.5:
                    st.warning(f"⚠️ Gaining too fast - reduce calories by ~200")
                elif weekly_rate < 0.2:
                    st.info(f"ℹ️ Slow gain - increase calories by ~200")
        
        with col2:
            st.markdown("### 🎯 Recommendations")
            
            # Protein check
            target_protein = float(settings.get('target_protein', 250))
            protein_diff = current_protein - target_protein
            
            if abs(protein_diff) <= 20:
                st.success(f"✅ Protein on target ({current_protein:.0f}g/day)")
            elif protein_diff < -20:
                st.warning(f"⬇️ Increase protein by {abs(protein_diff):.0f}g/day")
                st.caption("Add: 1 scoop whey (~25g) or 100g chicken (~30g)")
            else:
                st.info(f"⬆️ Protein {protein_diff:.0f}g above target (OK for bulking)")
            
            # Calorie recommendation
            if maintenance:
                ideal_surplus = 300  # For lean bulk
                ideal_calories = maintenance + ideal_surplus
                calorie_adjustment = ideal_calories - current_avg
                
                if abs(calorie_adjustment) > 100:
                    if calorie_adjustment > 0:
                        st.warning(f"📈 Increase calories by {calorie_adjustment:.0f}/day")
                        st.caption(f"Target: {ideal_calories:.0f} kcal/day")
                    else:
                        st.warning(f"📉 Reduce calories by {abs(calorie_adjustment):.0f}/day")
                        st.caption(f"Target: {ideal_calories:.0f} kcal/day")
                else:
                    st.success("✅ Calories are well-calibrated")
        
        # Weekly summary if available
        if not weekly_df.empty:
            st.markdown("---")
            st.markdown("### 📅 Weekly Averages")
            
            weekly_display = weekly_df.tail(4)[['week_ending', 'avg_weight', 'weight_change', 
                                               'avg_calories', 'avg_protein']].copy()
            weekly_display = weekly_display.round(2)
            st.dataframe(weekly_display, use_container_width=True, hide_index=True)
    
    # Recent PDFs processed
    if processed_df is not None and not processed_df.empty:
        st.markdown("---")
        with st.expander("📄 Recently Processed PDFs"):
            recent_files = processed_df.tail(10)[['file_name', 'processed_at', 'status', 'rows_extracted']].copy()
            st.dataframe(recent_files, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()