# """
# Flask REST API for ICD code retrieval system.
# Provides endpoints for model inference and health checks.
# """
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import sys
# from typing import Dict, Any

# # Add project root to path
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# from inference import ICDRetriever

# app = Flask(__name__)
# CORS(app)  # Enable CORS for Streamlit frontend

# # Global retriever instance (loaded once at startup)
# retriever = None


# def initialize_retriever():
#     """Initialize the ICD retriever on app startup."""
#     global retriever
    
#     # Configuration - adjust these paths as needed
#     config = {
#         'artifacts_dir': os.getenv('ARTIFACTS_DIR', './artifacts'),
#         'icd_csv_path': os.getenv('ICD_CSV_PATH', 'Data/icd_codes_8k.csv'),
#         'device': os.getenv('DEVICE', 'cpu'),  # 'cuda', 'mps', or 'cpu'
#         'txt_backbone': os.getenv('TXT_BACKBONE', 'emilyalsentzer/Bio_ClinicalBERT'),
#     }
    
#     print("Initializing ICD Retriever...")
#     print(f"Configuration: {config}")
    
#     try:
#         retriever = ICDRetriever(**config)
#         print("✓ ICD Retriever initialized successfully!")
#         return True
#     except Exception as e:
#         print(f"✗ Failed to initialize retriever: {e}")
#         return False


# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint."""
#     return jsonify({
#         'status': 'healthy' if retriever is not None else 'initializing',
#         'model_loaded': retriever is not None,
#     })


# @app.route('/api/predict', methods=['POST'])
# def predict():
#     """
#     Main prediction endpoint.
    
#     Expects JSON body:
#     {
#         "clinical_note": "Patient presents with...",
#         "lab_values": {
#             "a1c": 6.5,
#             "glucose": 120,
#             "creatinine": 1.1,
#             ...
#         },
#         "age": 65,
#         "sex": "M",
#         "top_k": 20
#     }
    
#     Returns:
#     {
#         "success": true,
#         "results": [
#             {
#                 "rank": 1,
#                 "code": "E11.9",
#                 "title": "Type 2 diabetes mellitus",
#                 "description": "...",
#                 "score": 0.85,
#                 "confidence": "High"
#             },
#             ...
#         ]
#     }
#     """
#     if retriever is None:
#         return jsonify({
#             'success': False,
#             'error': 'Model not initialized'
#         }), 503
    
#     try:
#         data = request.get_json()
        
#         # Validate required fields
#         required_fields = ['clinical_note', 'lab_values', 'age', 'sex']
#         missing_fields = [f for f in required_fields if f not in data]
        
#         if missing_fields:
#             return jsonify({
#                 'success': False,
#                 'error': f'Missing required fields: {missing_fields}'
#             }), 400
        
#         # Extract parameters
#         clinical_note = data['clinical_note']
#         lab_values = data['lab_values']
#         age = float(data['age'])
#         sex = data['sex']
#         top_k = int(data.get('top_k', 20))
        
#         # Validate lab values are numeric
#         for lab_key, lab_val in lab_values.items():
#             if lab_val is not None and not isinstance(lab_val, (int, float)):
#                 return jsonify({
#                     'success': False,
#                     'error': f'Lab value for {lab_key} must be numeric'
#                 }), 400
        
#         # Run prediction
#         results = retriever.predict(
#             clinical_note=clinical_note,
#             lab_values=lab_values,
#             age=age,
#             sex=sex,
#             top_k=top_k,
#         )
        
#         return jsonify({
#             'success': True,
#             'results': results,
#             'count': len(results),
#         })
        
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500


# @app.route('/api/lab_keys', methods=['GET'])
# def get_lab_keys():
#     """Return list of expected lab keys."""
#     if retriever is None:
#         return jsonify({
#             'success': False,
#             'error': 'Model not initialized'
#         }), 503
    
#     return jsonify({
#         'success': True,
#         'lab_keys': retriever.get_lab_keys()
#     })


# @app.route('/api/example', methods=['GET'])
# def get_example():
#     """Return an example patient case for testing."""
#     example = {
#         "clinical_note": (
#             "Patient is a 65-year-old male with a history of type 2 diabetes mellitus "
#             "and hypertension. He presents with complaints of increased thirst, frequent "
#             "urination, and fatigue over the past 2 weeks. Patient reports poor medication "
#             "adherence. Physical examination reveals blood pressure of 145/92 mmHg. "
#             "Laboratory studies show elevated hemoglobin A1c at 9.2%, fasting glucose "
#             "of 245 mg/dL, and creatinine of 1.4 mg/dL. Patient counseled on medication "
#             "compliance and lifestyle modifications."
#         ),
#         "lab_values": {
#             "a1c": 9.2,
#             "glucose": 245,
#             "creatinine": 1.4,
#             "egfr": 55,
#             "ldl": 135,
#             "hdl": 42,
#             "triglycerides": 180,
#             "wbc": 7.5,
#             "hgb": 13.2,
#             "platelets": 250,
#             "crp": 3.5,
#             "troponin": 0.01,
#             "bnp": 85,
#             "alt": 28,
#             "ast": 32
#         },
#         "age": 65,
#         "sex": "M",
#         "top_k": 10
#     }
    
#     return jsonify({
#         'success': True,
#         'example': example
#     })


# if __name__ == '__main__':
#     # Initialize retriever before starting server
#     if initialize_retriever():
#         print("\n" + "="*60)
#         print("Flask API Server Starting...")
#         print("="*60)
#         print("\nEndpoints:")
#         print("  GET  /health          - Health check")
#         print("  POST /api/predict     - ICD code prediction")
#         print("  GET  /api/lab_keys    - Get expected lab keys")
#         print("  GET  /api/example     - Get example patient case")
#         print("\n" + "="*60 + "\n")
        
#         app.run(
#             host='0.0.0.0',
#             port=5000,
#             debug=False,  # Set to True for development
#         )
#     else:
#         print("Failed to initialize retriever. Exiting.")
#         sys.exit(1)



"""
Flask REST API for ICD code retrieval system.
Provides endpoints for model inference and health checks.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from inference import ICDRetriever

app = Flask(__name__)
CORS(app)  # Enable CORS for Streamlit frontend

# Global retriever instance (loaded once at startup)
retriever = None


def initialize_retriever():
    """Initialize the ICD retriever on app startup."""
    global retriever
    
    # Configuration - adjust these paths as needed
    config = {
        'artifacts_dir': os.getenv('ARTIFACTS_DIR', './artifacts'),
        'icd_csv_path': os.getenv('ICD_CSV_PATH', 'Data/icd_codes_8k.csv'),
        'device': os.getenv('DEVICE', 'cpu'),  # 'cuda', 'mps', or 'cpu'
        'txt_backbone': os.getenv('TXT_BACKBONE', 'distilbert-base-uncased'),  # Changed to match your trained model
    }


    
    
    print("Initializing ICD Retriever...")
    print(f"Configuration: {config}")
    
    try:
        retriever = ICDRetriever(**config)
        print("✓ ICD Retriever initialized successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize retriever: {e}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if retriever is not None else 'initializing',
        'model_loaded': retriever is not None,
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Expects JSON body:
    {
        "clinical_note": "Patient presents with...",
        "lab_values": {
            "a1c": 6.5,
            "glucose": 120,
            "creatinine": 1.1,
            ...
        },
        "age": 65,
        "sex": "M",
        "top_k": 20
    }
    
    Returns:
    {
        "success": true,
        "results": [
            {
                "rank": 1,
                "code": "E11.9",
                "title": "Type 2 diabetes mellitus",
                "description": "...",
                "score": 0.85,
                "confidence": "High"
            },
            ...
        ]
    }
    """
    if retriever is None:
        return jsonify({
            'success': False,
            'error': 'Model not initialized'
        }), 503
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['clinical_note', 'lab_values', 'age', 'sex']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Extract parameters
        clinical_note = data['clinical_note']
        lab_values = data['lab_values']
        age = float(data['age'])
        sex = data['sex']
        top_k = int(data.get('top_k', 20))
        
        # Validate lab values are numeric
        for lab_key, lab_val in lab_values.items():
            if lab_val is not None and not isinstance(lab_val, (int, float)):
                return jsonify({
                    'success': False,
                    'error': f'Lab value for {lab_key} must be numeric'
                }), 400
        
        # Run prediction
        results = retriever.predict(
            clinical_note=clinical_note,
            lab_values=lab_values,
            age=age,
            sex=sex,
            top_k=top_k,
        )
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results),
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/lab_keys', methods=['GET'])
def get_lab_keys():
    """Return list of expected lab keys."""
    if retriever is None:
        return jsonify({
            'success': False,
            'error': 'Model not initialized'
        }), 503
    
    return jsonify({
        'success': True,
        'lab_keys': retriever.get_lab_keys()
    })


@app.route('/api/example', methods=['GET'])
def get_example():
    """Return an example patient case for testing."""
    example = {
        "clinical_note": (
            "Patient is a 65-year-old male with a history of type 2 diabetes mellitus "
            "and hypertension. He presents with complaints of increased thirst, frequent "
            "urination, and fatigue over the past 2 weeks. Patient reports poor medication "
            "adherence. Physical examination reveals blood pressure of 145/92 mmHg. "
            "Laboratory studies show elevated hemoglobin A1c at 9.2%, fasting glucose "
            "of 245 mg/dL, and creatinine of 1.4 mg/dL. Patient counseled on medication "
            "compliance and lifestyle modifications."
        ),
        "lab_values": {
            "a1c": 9.2,
            "glucose": 245,
            "creatinine": 1.4,
            "egfr": 55,
            "ldl": 135,
            "hdl": 42,
            "triglycerides": 180,
            "wbc": 7.5,
            "hgb": 13.2,
            "platelets": 250,
            "crp": 3.5,
            "troponin": 0.01,
            "bnp": 85,
            "alt": 28,
            "ast": 32
        },
        "age": 65,
        "sex": "M",
        "top_k": 10
    }
    
    return jsonify({
        'success': True,
        'example': example
    })


if __name__ == '__main__':
    # Initialize retriever before starting server
    if initialize_retriever():
        print("\n" + "="*60)
        print("Flask API Server Starting...")
        print("="*60)
        print("\nEndpoints:")
        print("  GET  /health          - Health check")
        print("  POST /api/predict     - ICD code prediction")
        print("  GET  /api/lab_keys    - Get expected lab keys")
        print("  GET  /api/example     - Get example patient case")
        print("\n" + "="*60 + "\n")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to True for development
        )
    else:
        print("Failed to initialize retriever. Exiting.")
        sys.exit(1)