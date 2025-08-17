import json
from jinja2 import Environment, FileSystemLoader
import os
import glob

# Get current script directory
script_dir = '/content/drive/MyDrive/myResearch/SFSU/KumoRFM-wildfire/reports/read_reports'

# Batch read JSON reports
json_reports_dir = "/content/drive/MyDrive/myResearch/SFSU/KumoRFM-wildfire/reports/json_reports"
json_files = glob.glob(os.path.join(json_reports_dir, "fire_risk_assessment_*.json"))

if not json_files:
    print(f"Error: No JSON report files found in {json_reports_dir}")
    exit()

print(f"Found {len(json_files)} JSON report files")

# Set Jinja2 template directory to script directory
env = Environment(loader=FileSystemLoader(script_dir))

# Load template
try:
    template = env.get_template('report_template.html')
except Exception as e:
    print(f"Error: Cannot find HTML template file. Please check path: {os.path.join(script_dir, 'report_template.html')}")
    print(f"Detailed error: {e}")
    exit()

# Set output directory
final_reports_dir = "/content/drive/MyDrive/myResearch/SFSU/KumoRFM-wildfire/reports/html_reports"
os.makedirs(final_reports_dir, exist_ok=True)

# Batch process each JSON file
successful_count = 0
for json_file_path in json_files:
    try:
        print(f"Processing: {os.path.basename(json_file_path)}")
        
        # Load JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        # Render HTML content
        output_html = template.render(report=report_data)
        
        # Build output file name
        output_file_name = f"report_{report_data['report_header']['report_id']}.html"
        output_file_path = os.path.join(final_reports_dir, output_file_name)
        
        # Write HTML file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(output_html)
        
        print(f"✅ Successfully generated: {output_file_name}")
        successful_count += 1
        
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(json_file_path)}: {e}")
        continue

print(f"\n🎉 Batch conversion completed! Successfully processed {successful_count}/{len(json_files)} files")
print(f"📁 HTML reports saved in: {final_reports_dir}")