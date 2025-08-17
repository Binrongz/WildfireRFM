#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KumoRFM Integration with Fire Risk Assessment System

"""

import pandas as pd
import json
import os
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import sys

# OpenAI API
from openai import OpenAI

from data_loader import DataLoader
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAIFireRiskAgent:
    """Fire risk assessment Agent based on OpenAI API"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        """
        Initialize OpenAI Agent
        
        Args:
            api_key: OpenAI API key
            model_name: Model name to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        
    def _load_prompt_template(self) -> str:
        """Load prompt template"""
        try:
            # Try to load from multiple possible paths
            possible_paths = [
                '/content/drive/MyDrive/myResearch/SFSU/KumoRFM-wildfire/src/prompts/fire_assessment_prompt.txt',
                '/content/fire_assessment_prompt.txt',
                './fire_assessment_prompt.txt'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        return f.read()
            
            # If file not found, use default template
            logger.warning("Prompt template file not found, using default template")
            return self._get_default_prompt_template()
            
        except Exception as e:
            logger.error(f"Failed to load prompt template: {e}")
            return self._get_default_prompt_template()
    
    def _get_default_prompt_template(self) -> str:
        """Default prompt template"""
        return """
You are a wildfire risk assessment expert. Analyze the following data and provide a comprehensive fire risk assessment.

LOCATION INFORMATION:
Location: {location_info}
Coordinates: {coordinates}
Administrative: {admin_info}
Land Type: {land_type}

FIRE RISK DATA:
FHSZ Risk Level: {fhsz_risk_level}
Fire History: {fire_history}

CURRENT CONDITIONS:
Weather: {weather_data}
Camera Analysis: {camera_analysis}

EMERGENCY RESOURCES:
Fire Stations: {resources_info}

KUMO PREDICTION DATA:
Risk Score: {kumo_risk_score}
Risk Ranking: #{kumo_rank} out of {kumo_total}
Prediction Window: {kumo_window} days

Please provide your assessment in the following JSON format:
{{
    "ai_assessment": {{
        "risk_score": [1-5 integer],
        "risk_level": "[Low/Low-Moderate/Moderate/High/Very High/Extreme]",
        "reasoning": "detailed analysis considering all factors including KumoRFM prediction",
        "confidence": "[Low/Medium/High]"
    }},
    "emergency_recommendations": [
        "specific actionable recommendation 1",
        "specific actionable recommendation 2",
        "specific actionable recommendation 3"
    ],
    "monitoring_recommendations": [
        "monitoring requirement 1",
        "monitoring requirement 2",
        "monitoring requirement 3"
    ]
}}
"""
    
    def _format_prompt_parameters(self, assessment_data: Dict, kumo_prediction: Dict) -> Dict[str, str]:
        """Format prompt parameters"""
        location = assessment_data.get('location', {})
        coordinates = location.get('coordinates', {})
        camera_assessment = assessment_data.get('camera_assessment', {})
        weather = assessment_data.get('weather', {})
        fire_risk_data = assessment_data.get('fire_risk_data', {})
        fire_stations = assessment_data.get('fire_stations', {})
        
        # Format location information
        location_info = f"{location.get('county', 'Unknown')} - {location.get('display_name', 'Unknown')}"
        
        # Format coordinates
        coordinates_str = f"{coordinates.get('lat', 0):.6f}, {coordinates.get('lon', 0):.6f}"
        
        # Format administrative information
        admin_info = f"City: {location.get('city', 'Unknown')}, County: {location.get('county', 'Unknown')}, State: California"
        
        # Format land type
        land_type = location.get('land_type', 'Unknown')
        
        # Format FHSZ risk
        fhsz_risk_level = f"{fire_risk_data.get('fhsz_risk_level', 'Unknown')} (Level {fire_risk_data.get('fhsz_risk', 'Unknown')})"
        
        # Format fire history
        fire_history = f"{fire_risk_data.get('fire_count', 0)} historical incidents, Dates: {', '.join(fire_risk_data.get('fire_dates', []) or ['None'])}"
        
        # Format weather data
        temp = weather.get('temperature', {})
        wind = weather.get('wind_speed', {})
        weather_data = f"Temperature: {temp.get('fahrenheit', 'N/A')}°F ({temp.get('celsius', 'N/A')}°C), Humidity: {weather.get('humidity', 'N/A')}%, Wind: {wind.get('mph', 'N/A')}mph ({wind.get('mps', 'N/A')}m/s), Conditions: {weather.get('weather_description', 'Unknown')}"
        
        # Format camera analysis
        camera_analysis = f"{camera_assessment.get('total_cameras', 0)} cameras deployed, {camera_assessment.get('cameras_detecting_fire', 0)} detecting fire, {camera_assessment.get('cameras_detecting_smoke', 0)} detecting smoke, Summary: {camera_assessment.get('detection_summary', 'No data')}"
        
        # Format fire resources
        stations_detail = fire_stations.get('stations_detail', [])
        if stations_detail:
            nearest_stations = []
            for station in stations_detail[:3]:
                name = station.get('name', 'Unknown')
                distance = station.get('distance_km', 'N/A')
                nearest_stations.append(f"{name} ({distance}km)")
            resources_info = f"{fire_stations.get('station_count', 0)} fire stations available. Nearest: {'; '.join(nearest_stations)}"
        else:
            resources_info = "No fire station data available"
        
        return {
            'location_info': location_info,
            'coordinates': coordinates_str,
            'admin_info': admin_info,
            'land_type': land_type,
            'fhsz_risk_level': fhsz_risk_level,
            'fire_history': fire_history,
            'weather_data': weather_data,
            'camera_analysis': camera_analysis,
            'resources_info': resources_info,
            'kumo_risk_score': kumo_prediction.get('risk_score', 'N/A'),
            'kumo_rank': kumo_prediction.get('rank', 'N/A'),
            'kumo_total': kumo_prediction.get('total', 'N/A'),
            'kumo_window': kumo_prediction.get('time_window', 30)
        }
    
    def assess_fire_risk(self, assessment_data: Dict, kumo_prediction: Dict) -> Dict:
        """Execute fire risk assessment"""
        location_name = assessment_data.get('location', {}).get('display_name', 'Unknown')
        logger.info(f"Starting OpenAI risk assessment for location: {location_name}")
        
        # Format parameters
        params = self._format_prompt_parameters(assessment_data, kumo_prediction)
        
        # Create prompt
        prompt = self.prompt_template.format(**params)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a wildfire risk assessment expert. Provide accurate, actionable analysis based on the data provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            response_content = response.choices[0].message.content
            
            # Parse response
            assessment_result = self._parse_assessment_response(response_content)

            # Process numpy types in kumo_prediction
            processed_kumo = {}
            for key, value in kumo_prediction.items():
                if hasattr(value, 'item'):  # numpy type
                    processed_kumo[key] = value.item()
                else:
                    processed_kumo[key] = value
            
            # Build complete result
            complete_result = {
                'ai_assessment': assessment_result.get('ai_assessment', {}),
                'emergency_recommendations': assessment_result.get('emergency_recommendations', []),
                'monitoring_recommendations': assessment_result.get('monitoring_recommendations', []),
                'kumo_prediction': processed_kumo,
                'timestamp': datetime.now().isoformat(),
                'sensor_data': assessment_data,
                'raw_ai_response': response_content
            }

            return complete_result  # Important return
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            # Return default result
            return {
                'ai_assessment': {
                    'risk_score': 3,
                    'risk_level': 'Assessment Failed',
                    'reasoning': f'Unable to complete AI assessment: {str(e)}',
                    'confidence': 'Low'
                },
                'emergency_recommendations': ['Contact emergency services if immediate danger'],
                'monitoring_recommendations': ['Manual monitoring required'],
                'kumo_prediction': kumo_prediction,
                'timestamp': datetime.now().isoformat(),
                'sensor_data': assessment_data,
                'raw_ai_response': ''
            }

    def _get_fallback_result(self, kumo_prediction):
        """Fallback result when assessment_data is None"""
        return {
            'ai_assessment': {
                'risk_score': 3,
                'risk_level': 'Data Error',
                'reasoning': 'Assessment data was None',
                'confidence': 'Low'
            },
            'emergency_recommendations': ['Data error occurred'],
            'monitoring_recommendations': ['Manual verification required'],
            'kumo_prediction': kumo_prediction,
            'timestamp': datetime.now().isoformat(),
            'sensor_data': {},
            'raw_ai_response': ''
        }
    
    def _parse_assessment_response(self, response: str) -> Dict:
        """Parse AI response"""
        try:
            # Try to extract JSON
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_part = response[start:end]
                return json.loads(json_part)
        except:
            pass
        
        # If JSON parsing fails, return default result
        return {
            'ai_assessment': {
                'risk_score': 3,
                'risk_level': 'Moderate',
                'reasoning': response[:500] if response else 'No assessment available',
                'confidence': 'Medium'
            },
            'emergency_recommendations': ['Monitor conditions closely'],
            'monitoring_recommendations': ['Regular monitoring required']
        }


class KumoFireRiskIntegrator:
    """Integrator for KumoRFM and fire risk assessment system"""
    
    def __init__(self, openai_api_key: str, data_dir: str = "/content/drive/MyDrive/myResearch/SFSU/KumoRFM-wildfire/data"):
        """
        Initialize integrator
        
        Args:
            openai_api_key: OpenAI API key
            data_dir: Data file directory
        """
        self.data_dir = data_dir
        
        # Initialize components
        self.data_loader = DataLoader(data_dir)
        self.ai_agent = OpenAIFireRiskAgent(openai_api_key)
        self.report_generator = ReportGenerator("/content/drive/MyDrive/myResearch/SFSU/KumoRFM-wildfire/reports/json_reports")
        
        logger.info("KumoFireRiskIntegrator initialized")
    
    def get_top5_risk_areas_from_kumo(self, kumo_model, users_df, time_window=30, anchor_time=None):
        """Get top 5 highest risk areas from KumoRFM"""
        logger.info(f"Starting KumoRFM risk assessment, time window: {time_window} days")
        
        risk_predictions = []
        total_points = len(users_df)
        
        # Batch prediction (can sample to save time)
        sample_size = min(1000, total_points)  # Limit sample size
        sampled_df = users_df.sample(n=sample_size, random_state=42)
        
        logger.info(f"Performing risk prediction on {len(sampled_df)} grid points...")
        
        for idx, row in sampled_df.iterrows():
            gridpoint = row['location_name']
            try:
                query = f"PREDICT SUM(orders.order_amount, 0, {time_window}, days) FOR users.location_name='{gridpoint}'"
                if anchor_time:
                    prediction = kumo_model.predict(query, anchor_time=pd.Timestamp(anchor_time))
                else:
                    prediction = kumo_model.predict(query)
                
                risk_score = prediction['TARGET_PRED'].iloc[0]
                
                # Only keep areas with risk
                if risk_score > 0:
                    risk_predictions.append({
                        'gridpoint': gridpoint,
                        'risk_score': risk_score,
                        'lat': row['location_lat'],
                        'lon': row['location_lon'],
                        'county': row['admin_county'],
                        'city': row['admin_city'],
                        'land_type': row['land_land_type'],
                        'fhsz_risk': row['fire_risk_fhsz_risk_level'],
                        'historical_fires': row['fire_risk_fire_count'],
                        'prediction_timestamp': prediction['ANCHOR_TIMESTAMP'].iloc[0]
                    })
                    
            except Exception as e:
                logger.warning(f"Prediction failed for {gridpoint}: {e}")
                continue
        
        # Sort to get top 5
        top5_areas = sorted(risk_predictions, key=lambda x: x['risk_score'], reverse=True)[:5]
        
        logger.info(f"Found {len(risk_predictions)} areas with risk, returning top {len(top5_areas)}")
        
        # Print results
        print("\n🔥 Top 5 highest risk areas predicted by KumoRFM:")
        for i, area in enumerate(top5_areas, 1):
            print(f"{i}. {area['gridpoint']} ({area['county']})")
            print(f"   Risk score: {area['risk_score']:.6f}")
            print(f"   Coordinates: ({area['lat']:.4f}, {area['lon']:.4f})")
            print(f"   Land type: {area['land_type']}")
            print()

        # Add camera generation logic
        self._add_cameras_for_high_risk_areas(top5_areas, anchor_time)
        
        return top5_areas

    def _add_cameras_for_high_risk_areas(self, top5_areas, anchor_time=None):
      """Dynamically generate camera data for high-risk areas"""
      import random
      
      camera_file_path = os.path.join(self.data_dir, "camera_monitoring_dataset.jsonl")
      new_cameras = []
      
      # Set timestamp
      if anchor_time:
          base_time = pd.Timestamp(anchor_time)
      else:
          base_time = pd.Timestamp.now()
      
      for area in top5_areas:
          # Generate 3-5 cameras for each area
          camera_count = random.randint(3, 5)
          county = area['county'].replace(' ', '_').replace('-', '_')
          
          for i in range(1, camera_count + 1):
              # Location fine-tuning (±0.005 degrees, approximately 500m range)
              lat_offset = random.uniform(-0.005, 0.005)
              lon_offset = random.uniform(-0.005, 0.005)
              
              # Random detection status
              status = random.choice(['fire', 'smoke', 'normal', 'normal', 'normal'])  # Higher probability for normal status
              
              # Timestamp with random offset
              time_offset = random.randint(0, 1800)  # 0-30 minutes random offset
              timestamp = (base_time + pd.Timedelta(seconds=time_offset)).isoformat()
              
              camera = {
                  "camera_id": f"CAM_{county}_{area['gridpoint'].replace(' ', '_')}_{i:02d}",
                  "lat": round(area['lat'] + lat_offset, 6),
                  "lon": round(area['lon'] + lon_offset, 6),
                  "detection_status": status,
                  "timestamp": timestamp
              }
              new_cameras.append(camera)
      
      # Append to file
      try:
          with open(camera_file_path, 'a', encoding='utf-8') as f:
              for camera in new_cameras:
                  f.write(json.dumps(camera) + '\n')
          
          logger.info(f"Generated {len(new_cameras)} camera data for {len(top5_areas)} high-risk areas")
      except Exception as e:
          logger.error(f"Failed to write camera data: {e}")
    
    def find_cameras_near_area(self, area_info: Dict, radius_km: float = 10.0) -> List[Dict]:
        """Find nearby real camera data for specified area"""
        lat, lon = area_info['lat'], area_info['lon']
        gridpoint = area_info['gridpoint']
        
        # Load camera dataset
        camera_data = self.data_loader.load_camera_jsonl_data()
        if not camera_data:
            logger.warning(f"Cannot load camera data, creating default camera data for {gridpoint}")
            # If no camera data, create a default camera
            return [{
                'camera_id': f"CAM_{gridpoint}_DEFAULT",
                'lat': lat,
                'lon': lon,
                'detection_status': 'normal',
                'timestamp': datetime.now().isoformat()
            }]
        
        # Find cameras within specified radius
        nearby_cameras = []
        for camera in camera_data:
            try:
                camera_lat = camera.get('lat', 0)
                camera_lon = camera.get('lon', 0)
                
                # Calculate distance
                distance = self.data_loader.haversine_distance(
                    lat, lon, camera_lat, camera_lon
                ) / 1000  # Convert to kilometers
                
                if distance <= radius_km:
                    nearby_cameras.append(camera)
                    
            except Exception as e:
                logger.warning(f"Error processing camera data: {e}")
                continue
        
        if not nearby_cameras:
            logger.warning(f"No cameras found within {radius_km}km, expanding search range")
            # If no nearby cameras found, take the closest few cameras
            camera_distances = []
            for camera in camera_data:
                try:
                    camera_lat = camera.get('lat', 0)
                    camera_lon = camera.get('lon', 0)
                    distance = self.data_loader.haversine_distance(
                        lat, lon, camera_lat, camera_lon
                    ) / 1000
                    camera_distances.append((distance, camera))
                except:
                    continue
            
            # Take the closest 5 cameras
            camera_distances.sort(key=lambda x: x[0])
            nearby_cameras = [cam for dist, cam in camera_distances[:5]]
        
        logger.info(f"Found {len(nearby_cameras)} nearby cameras for {gridpoint}")
        return nearby_cameras
    
    def create_assessment_data_for_area(self, area_info: Dict, rank: int, total: int, time_window: int) -> Dict:
        """Create assessment data for specified area"""
        # Find real camera data nearby this area
        cameras = self.find_cameras_near_area(area_info)
        
        # Cluster analysis of cameras
        camera_clusters = self.data_loader.cluster_cameras_by_distance(cameras, 500)
        main_cluster = max(camera_clusters, key=len) if camera_clusters else cameras
        cluster_analysis = self.data_loader.analyze_cluster_detection(main_cluster)
        
        # Match static geographic data
        nearest_grid = self.data_loader.match_nearest_grid_point(
            area_info['lat'], 
            area_info['lon']
        )
        
        if not nearest_grid:
            logger.warning(f"Cannot find matching geographic data for {area_info['gridpoint']}")
            # Create basic geographic data
            nearest_grid = {
                'location': {'name': area_info['gridpoint']},
                'display_name': f"{area_info['county']} - {area_info['land_type']}",
                'admin': {'city': area_info['city'], 'county': area_info['county']},
                'land': {'land_type': area_info['land_type']},
                'fire_risk': {
                    'fhsz_risk_level': area_info['fhsz_risk'],
                    'fire_count': area_info['historical_fires'],
                    'fire_dates': []
                },
                'resources': {'station_count': 0, 'stations_detail': []}
            }
        
        # Process fire station data
        resources = nearest_grid.get("resources", {})
        if "nearby_stations" in resources and "stations_detail" not in resources:
            resources["stations_detail"] = self.data_loader._parse_nearby_stations(resources["nearby_stations"])
        
        # Get weather data
        weather_data = self.data_loader.integrate_weather_data(
            area_info['lat'], 
            area_info['lon']
        )
        
        # Build unified assessment data
        assessment_data = {
            'cluster_id': f"KUMO_HIGH_RISK_{area_info['gridpoint']}",
            'location': {
                'coordinates': {'lat': area_info['lat'], 'lon': area_info['lon']},
                'display_name': nearest_grid.get('display_name', f"{area_info['county']} - {area_info['gridpoint']}"),
                'city': area_info['city'],
                'county': area_info['county'],
                'land_type': area_info['land_type'],
                'nearest_grid_point': area_info['gridpoint']
            },
            'fire_risk_data': nearest_grid.get('fire_risk', {}),
            'land_info': nearest_grid.get('land', {}),
            'camera_assessment': cluster_analysis,
            'fire_stations': resources,
            'weather': weather_data
        }
        
        return assessment_data
    
    def run_integrated_assessment(self, kumo_model, users_df, time_window=30, anchor_time=None):
        """Run complete integrated assessment"""
        logger.info("🚀 Starting KumoRFM integrated fire risk assessment...")
        
        try:
            # Load static geographic data
            self.data_loader.load_static_geo_data()
            
            # 1. Get Top5 high-risk areas from KumoRFM
            top5_areas = self.get_top5_risk_areas_from_kumo(kumo_model, users_df, time_window, anchor_time)
            
            if not top5_areas:
                logger.error("No high-risk areas obtained")
                return False
            
            # 2. Generate comprehensive assessment for each high-risk area
            successful_reports = 0
            
            for i, area in enumerate(top5_areas, 1):
                try:
                    logger.info(f"🔍 Processing {i}th high-risk area: {area['gridpoint']}")
                    
                    # Create assessment data
                    assessment_data = self.create_assessment_data_for_area(
                        area, i, len(top5_areas), time_window
                    )
                    
                    # KumoRFM prediction information
                    kumo_prediction = {
                        'risk_score': area['risk_score'],
                        'rank': i,
                        'total': len(top5_areas),
                        'time_window': time_window,
                        'prediction_timestamp': area['prediction_timestamp']
                    }
                    
                    # AI assessment
                    ai_result = self.ai_agent.assess_fire_risk(assessment_data, kumo_prediction)
                    
                    # Generate report
                    report_file = self.report_generator.generate_complete_report(
                        ai_result, 
                        self.ai_agent.model_name
                    )
                    
                    if report_file:
                        successful_reports += 1
                        logger.info(f"✅ Report generated: {report_file}")
                    
                except Exception as e:
                    logger.error(f"Error processing area {area['gridpoint']}: {e}")
                    continue
            
            # Summary
            print(f"\n🎉 KumoRFM integrated assessment completed!")
            print(f"📊 Assessed {len(top5_areas)} high-risk areas")
            print(f"📄 Successfully generated {successful_reports} technical reports")
            print(f"📁 Reports saved at: {os.path.abspath(self.report_generator.output_dir)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Integrated assessment failed: {str(e)}")
            print(f"\n❌ Assessment failed: {str(e)}")
            return False


# Example usage function
def run_kumo_integration(kumo_model, users_df, openai_api_key, time_window=30, anchor_time=None):
    """
    Main function to run KumoRFM integrated assessment
    
    Args:
        kumo_model: KumoRFM model instance
        users_df: User data DataFrame
        openai_api_key: OpenAI API key
        time_window: Prediction time window (days)
    """
    # Create integrator
    integrator = KumoFireRiskIntegrator(openai_api_key)
    
    # Run integrated assessment
    success = integrator.run_integrated_assessment(kumo_model, users_df, time_window, anchor_time)
    
    return success


if __name__ == "__main__":
    print("KumoRFM Fire Risk Integration Script")
    print("Please call run_kumo_integration() function in Colab to run integrated assessment")