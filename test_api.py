from fastapi.testclient import TestClient
from api.main import app, model, metadata

client = TestClient(app)


def test_model_loaded():
    assert model is not None


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_version():
    response = client.get("/health")

    assert response.json()["model_version"] == "1.0.0"


def test_valid_prediction():

    response = client.post(
        "/predict",
        json={
            "mean_radius": 17.99,
            "mean_texture": 10.38,
            "mean_perimeter": 122.8,
            "mean_area": 1001,
            "mean_smoothness": 0.1184,
            "mean_compactness": 0.2776,
            "mean_concavity": 0.3001,
            "mean_concave_points": 0.1471,
            "mean_symmetry": 0.2419,
            "mean_fractal_dimension": 0.07871,
            "radius_error": 1.095,
            "texture_error": 0.9053,
            "perimeter_error": 8.589,
            "area_error": 153.4,
            "smoothness_error": 0.006399,
            "compactness_error": 0.04904,
            "concavity_error": 0.05373,
            "concave_points_error": 0.01587,
            "symmetry_error": 0.03003,
            "fractal_dimension_error": 0.006193,
            "worst_radius": 25.38,
            "worst_texture": 17.33,
            "worst_perimeter": 184.6,
            "worst_area": 2019,
            "worst_smoothness": 0.1622,
            "worst_compactness": 0.6656,
            "worst_concavity": 0.7119,
            "worst_concave_points": 0.2654,
            "worst_symmetry": 0.4601,
            "worst_fractal_dimension": 0.1189
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert result["class"] in ["Benign", "Malignant"]
    assert 0 <= result["malignant_probability"] <= 1
    assert result["model_version"] == "1.0.0"


def test_invalid_prediction():

    response = client.post(
        "/predict",
        json={"mean_radius": -1}
    )

    assert response.status_code == 422
