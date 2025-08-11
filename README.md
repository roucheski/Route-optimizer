# Route-optimizer

## Overview
This app helps plan and optimize employee transport routes from various pick-up points to company plants.  
It calculates the most efficient paths, displays route distances, and marks each pick-up in sequence.  
By identifying detours and inefficiencies, it supports cost reduction and better scheduling.

## Features
- Upload employee location dataset (CSV)
- Generate optimal routes from pick-up points to plant
- Display drop order numbers and route distances

## Sample Dataset

The dataset used in the app should follow this exact format:
| Address         | Employee Number | Distance before(km) | Latitudes  | Longitudes |
|-----------------|-----------------|---------------------|------------|------------|
| Point 1         | 0               | 0                   | 7.731897   | 80.133021  |
| N-Aluthgama_2   | MG - 7431       | 4.9407               | 7.768698   | 80.111381  |

## Requirements
- Python 3.10 or later
- Install dependencies:
```bash
pip install streamlit pandas folium geopy

