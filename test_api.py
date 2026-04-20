#!/usr/bin/env python3
"""
Test script for Enzyme Activity Prediction API endpoints.
Run after starting the server with: uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

import requests
import json
import sys
from time import sleep

BASE_URL = "http://127.0.0.1:8000"

def test_root():
    """Test root endpoint."""
    print("Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    print("✓ Root endpoint working")
    return True

def test_models_list():
    """Test models list endpoint."""
    print("\nTesting models list endpoint...")
    response =requests.get(f"{BASE_URL}/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3
    for model in data:
        assert model["available"] == True
        print(f"  ✓ Model '{model['model_name']}' available")
    print("✓ Models list endpoint working")
    return True

def test_single_prediction():
    """Test single prediction endpoint."""
    print("\nTesting single prediction endpoint...")
    
    # Use a short sequence for faster testing
    payload = {
        "enzyme": {
            "sequence": "MKALSKLKAEEGIWMTDVPVPELGHNDLLIKIRKTAICGTDVHIYNWDEWSQKTIPVPMVVGHEYVGEVVGIGQEVKGFK",
            "ec": "1.1.1.103",
            "organism": "Cupriavidus necator",
            "n_measurements": 1,
            "ph_opt": 7.25,
            "temp_opt": 37.0,
            "kmValue": 5.2,
            "molecularWeight": 45000
        },
        "dataset_name": "gst"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/predict/single",
        json=payload,
        timeout=60
    )
    
    if response.status_code != 200:
        print(f"  ✗ Error: {response.status_code}")
        print(f"  Response: {response.text}")
        return False
    
    data = response.json()
    assert "predicted_log_kcat" in data
    assert "model_name" in data
    assert data["model_name"] == "gst"
    print(f"  ✓ Predicted log(kcat): {data['predicted_log_kcat']:.3f}")
    print("✓ Single prediction endpoint working")
    return True

def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\nTesting batch prediction endpoint...")
    
    payload = {
        "enzymes": [
            {
                "sequence": "MKALSKLKAEEGIWMTDVPVPELGHNDLLIKIRKTAICGTDVHIYNWDEWSQKTIPVPMVVGHEYVGEVVGIGQEVKGFK",
                "ec": "1.1.1.103",
                "organism": "Organism A",
                "kmValue": 5.2,
                "molecularWeight": 45000
            },
            {
                "sequence": "MKALSKLKAEEGIWMTDVPVPELGHNDLLIKIRKTAICGTDVHIYNWDEWSQKTIPVPMVVGHEYVGEVVGIGQEVKGFKIGDRV",
                "ec": "1.1.1.103",
                "organism": "Organism B",
                "kmValue": 6.1,
                "molecularWeight": 46000
            }
        ],
        "dataset_name": "gst"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/predict/batch",
        json=payload,
        timeout=90
    )
    
    if response.status_code != 200:
        print(f"  ✗ Error: {response.status_code}")
        print(f"  Response: {response.text}")
        return False
    
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    for i, pred in enumerate(data, 1):
        print(f"  ✓ Enzyme {i}: log(kcat) = {pred['predicted_log_kcat']:.3f}")
    print("✓ Batch prediction endpoint working")
    return True

def main():
    """Run all tests."""
    print("="  * 60)
    print("Enzyme Activity Prediction API - Endpoint Tests")
    print("=" * 60)
    
    tests = [
        ("Root", test_root),
        ("Models List", test_models_list),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} test failed with error: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASSED" if passed_test else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
