# 🔥 Wildfire alarm system with KumoRFM

**AI-Powered Wildfire Risk Prediction and Assessment Platform**

An intelligent wildfire risk assessment system that combines KumoRFM machine learning predictions with real-time sensor data, weather conditions, and comprehensive emergency response planning.

---

## 🌟 Key Features

- 🧠 KumoRFM Prediction Engine: Identifies top 5 highest-risk areas using temporal machine learning  
- 📹 Dynamic Camera Network: Generates and analyzes camera detection data (fire/smoke/normal)  
- 🌤️ Real-time Weather Integration: Live weather data from OpenWeatherMap API  
- 🚒 Emergency Resource Mapping: Comprehensive fire station coverage analysis  
- 🤖 AI Risk Assessment: OpenAI-powered comprehensive risk analysis  
- 📊 Professional Reports: Automated HTML and JSON report generation  
- 📱 Responsive Design: Clean, professional web-ready reports  

---

## 🏗️ System Architecture

```
KumoRFM Prediction → Top 5 Risk Areas → Data Integration → AI Analysis → Report Generation
         ↓                    ↓                   ↓              ↓              ↓
   Time-series ML      Geographic Points    Weather + Camera   OpenAI GPT    HTML + JSON
     Modeling           + Coordinates        + Fire Stations   Assessment     Reports
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install kumoai pandas openai requests jinja2
```

### API Keys needed

- OpenAI API Key  
- OpenWeatherMap API Key  
- KumoRFM API Key  

---

### Setup

1. Clone the repository

```bash
git clone [repository-url]
cd KumoRFM-wildfire
```

2. Configure API Keys

```bash
export OPENAI_API_KEY="your_openai_api_key"
export OPENWEATHER_API_KEY="your_openweather_api_key"
export KUMO_API_KEY="your_kumo_api_key"
```

3. Prepare Data

- Place `nested_california_fire_risk_enhanced_dataset.jsonl` in `/data/` directory  
- Get the camera detection data for high-risk areas  

---

### Basic Usage

```python
# Initialize KumoRFM model
import kumoai.experimental.rfm as rfm
model = rfm.KumoRFM(graph)

# Run integrated assessment
from kumo_integration import run_kumo_integration

success = run_kumo_integration(
    kumo_model=model,
    users_df=users_df,
    openai_api_key=OPENAI_API_KEY,
    time_window=30,
    anchor_time="2024-07-01"  # Optional
)
```

---

## 📁 Project Structure

```
KumoRFM-wildfire/
├── src/
│   ├── kumo_integration.py      # Main integration script
│   ├── data_loader.py           # Data processing utilities
│   ├── report_generator.py      # Report generation engine
│   └── api_utils/
│       └── weather_fetcher.py   # Weather API integration
├── data/
│   ├── nested_california_fire_risk_enhanced_dataset.jsonl
│   └── camera_monitoring_dataset.jsonl
├── reports/
│   ├── json_reports             # Generated JSON assessments
│   └── html_reports/           # Generated HTML reports
    └── reports/read_reports/
    ├── html_report.py           # JSON to HTML converter
    └── report_template.html     # HTML template
```

---

## 🔥 Core Workflow

### 1. KumoRFM Prediction
- Samples 1,000 geographic points from California dataset  
- Uses temporal regression to predict fire risk scores  
- Identifies top 5 highest-risk areas with coordinates  

### 2. Dynamic Camera Detection Data
- Camera Network: Multiple cameras per high-risk area  
- Detection Status: Fire/smoke/normal detection states  
- Geographic Positioning: ±500m radius around prediction points  

### 3. Comprehensive Data Integration
- Weather Data: Real-time temperature, humidity, wind conditions  
- Historical Analysis: Fire incident history and patterns  
- Resource Mapping: Nearest fire stations with response distances  
- Land Classification: FHSZ risk levels and land types  

### 4. AI-Powered Assessment
- Multi-factor Analysis: Combines all data sources  
- Risk Scoring: 1–5 scale with detailed reasoning  
- Emergency Recommendations: Actionable response plans  
- Monitoring Requirements: Ongoing surveillance needs  

### 5. Professional Reporting
- JSON Format: Structured data for API integration  
- HTML Reports: Professional web-ready presentations  
- Risk Visualization: Color-coded risk levels  
- Resource Tables: Emergency contact information  

---

## 🎯 Use Cases

- 🚨 Emergency Response Planning  
- 📊 Insurance Risk Assessment  
- 🏛️ Government Decision Support  
- 🔬 Research Applications  
- 📱 Public Safety Alerts  

---

## ⚙️ Configuration Options

### Time-based Queries

```python
# Current conditions
run_kumo_integration(model, users_df, api_key)

# Historical analysis
run_kumo_integration(model, users_df, api_key, anchor_time="2023-08-15")

# Custom prediction window
run_kumo_integration(model, users_df, api_key, time_window=60)
```

### Sampling Configuration

```python
# Modify sampling size in kumo_integration.py
sample_size = min(1000, total_points)  # Adjust for performance vs accuracy
```

---

## 📊 Sample Output

```
Location: Tulare County, GridPoint_17419
KumoRFM Ranking: #1 out of 5 (Risk Score: 0.007864)
AI Risk Level: Very High (4/5)
Camera Status: 2/5 cameras detecting fire
Weather: 87°F, 40% humidity, 11mph winds
Nearest Station: Smartville FS (6.6km)
```

---

## 🛡️ Technical Specifications

- ML Framework: KumoRFM temporal regression  
- AI Engine: OpenAI GPT-4o-mini  
- Data Processing: Pandas, NumPy  
- Geospatial: Haversine distance calculations  
- Weather API: OpenWeatherMap integration  
- Report Engine: Jinja2 templating  
- File Formats: JSONL, JSON, HTML  

---

## 🔧 Advanced Features

### Camera Clustering Algorithm
- Automatic grouping by 500m radius  
- Graph-based connectivity clustering  
- Multi-camera detection aggregation  

### Weather Integration
- Real-time API calls with fallback handling  
- Multi-unit temperature and wind speed conversion  
- Weather condition impact analysis  

### Emergency Resource Optimization
- Distance-based fire station ranking  
- Response time categorization  
- Coverage gap identification  

---

## 🚨 Important Notes

- ⚠️ Disclaimer: This system is for informational purposes only. All AI-generated recommendations should be verified by qualified fire safety professionals before implementation.  
- 🔐 Security: Ensure all API keys are properly secured and never committed to version control.  
- 📞 Emergency: In case of immediate fire danger, contact emergency services (911) immediately regardless of system assessment.  

---

## 🤝 Contributing

1. Fork the repository  
2. Create feature branch (`git checkout -b feature/enhancement`)  
3. Commit changes (`git commit -am 'Add new feature'`)  
4. Push to branch (`git push origin feature/enhancement`)  
5. Create Pull Request  

---

## 📄 License

This project is licensed under the MIT License – see the `LICENSE` file for details.

---

## 🙏 Acknowledgments

- KumoAI: Temporal machine learning platform  
- OpenAI: AI-powered risk assessment  
- OpenWeatherMap: Real-time weather data  
- California Fire & Resource Assessment Program: Geographic fire risk data