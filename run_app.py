#!/usr/bin/env python3
"""
Startup script for the Skin Disease Classification System.
This script checks dependencies and starts the Flask application.
"""

import os
import sys
import subprocess
import importlib

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'flask',
        'tensorflow', 
        'opencv-python',
        'numpy',
        'PIL',
        'werkzeug'
    ]
    
    missing_packages = []
    
    print("🔍 Checking dependencies...")
    
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'opencv-python':
                importlib.import_module('cv2')
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_model_file():
    """Check if the model file exists"""
    model_file = 'skindiseasemodel.h5'
    
    if os.path.exists(model_file):
        print(f"✅ Model file found: {model_file}")
        return True
    else:
        print(f"❌ Model file not found: {model_file}")
        print("Please ensure the trained model file is in the project directory.")
        return False

def start_application():
    """Start the Flask application"""
    print("\n🚀 Starting Skin Disease Classification System...")
    print("📱 Web interface will be available at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 60)
    
    try:
        # Import and run the Flask app
        from APP import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

def main():
    """Main function"""
    print("🏥 Skin Disease Classification System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model file
    if not check_model_file():
        sys.exit(1)
    
    # Start application
    start_application()

if __name__ == "__main__":
    main() 