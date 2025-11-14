import requests
from pathlib import Path

API_URL = 'http://localhost:5000'

def test_health_check():
    print("\n[TEST 1] Health Check")
    
    response = requests.get(f'{API_URL}/')
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200, "Health check failed"
    print("Health check passed")


def test_get_classes():
    print("\n[TEST 2] Get Classes")
    
    response = requests.get(f'{API_URL}/classes')
    
    print(f"Status code: {response.status_code}")
    data = response.json()
    print(f"Classes: {data['classes']}")
    print(f"Count: {data['count']}")
    
    assert response.status_code == 200, "Get classes failed"
    assert len(data['classes']) == 4, "Should have 4 classes"
    print("Get classes passed")


def test_predict():
    print("\n[TEST 3] Predict")
    
    # Find a test image
    test_image_dir = Path('../data/processed/test')
    
    # Try to find any image
    test_image = None
    for class_dir in test_image_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg'))
            if images:
                test_image = images[0]
                break
    
    if test_image is None:
        print("No test image found")
        return
    
    print(f"Using test image: {test_image.name}")
    
    # Send request
    with open(test_image, 'rb') as f:
        files = {'file': f}
        response = requests.post(f'{API_URL}/predict', files=files)
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Prediction: {data['prediction']}")
        print(f"Confidence: {data['confidence']:.2%}")
        print("\nProbabilities:")
        for cls, prob in data['probabilities'].items():
            print(f"  {cls:15s}: {prob:.2%}")
        print("Prediction passed")
    else:
        print(f"Prediction failed: {response.json()}")


def test_invalid_file():
    print("\n[TEST 4] Invalid File")
    
    # Create a text file
    files = {'file': ('test.txt', b'not an image', 'text/plain')}
    response = requests.post(f'{API_URL}/predict', files=files)
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 400, "Should reject invalid file"
    print("Invalid file rejected correctly")


if __name__ == '__main__':
    print("="*60)
    print("TESTING BRAIN TUMOR API")
    print("="*60)
    
    try:
        test_health_check()
        test_get_classes()
        test_predict()
        test_invalid_file()
        
        print("ALL TESTS PASSED")
    
    except Exception as e:
        print(f"\nTest failed: {e}")