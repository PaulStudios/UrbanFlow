---

# UrbanFlow

UrbanFlow is a smart traffic light system designed to prioritize emergency vehicles and optimize traffic flow in urban environments. This project aims to improve urban mobility, reduce congestion, and enhance safety using dynamic Priority Management and a user-friendly Android application.

## Features

- **Mobile Application (UrbanFlow):**
  - Secure user registration and login for vehicle owners, commuters, and emergency service drivers.
  - Real-time updates on traffic light statuses and estimated wait times.
  - GPS tracking and route prediction for emergency vehicles using Google Maps API.

- **Priority Management System:**
  - Dynamic calculation of priority scores based on real-time data and traffic conditions.
  - Integration with traffic data sources and computer vision for accurate traffic flow analysis.
  - Future enhancement with machine learning models for predictive traffic pattern analysis.

## Vehicle Route Prediction

### Overview

This project aims to predict vehicle routes using Kalman Filter and LSTM models. The data is preprocessed, normalized, and divided into training and testing sets. The Kalman Filter is applied for initial predictions, followed by training an LSTM model to make more accurate predictions.

### Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- filterpy
- tensorflow
- matplotlib
- concurrent.futures
- logging

Install the required packages using:
```bash
pip install pandas numpy scikit-learn filterpy tensorflow matplotlib
```

### Usage

#### Prepare the Data

1. Ensure the data is in a CSV file with columns: `upload_id`, `timestamp`, `longitude`, and `latitude`.
2. Place the CSV file in the project directory.

#### Run the Main Script

The main function in the script will load the data, preprocess it, apply the Kalman Filter, train the LSTM model, and evaluate the predictions.

```bash
python prediction_ai.py
```

#### Model Training

The LSTM model is trained with early stopping and model checkpointing. The trained model is saved as `final_lstm_model.keras`.

#### Making Predictions

Use the `load_and_predict` function to load a trained model and make predictions on new data.

```python
predictions = load_and_predict('final_lstm_model.keras', test_data)
```

### Logging

The script uses Python's logging module to provide detailed logs of the process. Logs include data loading, preprocessing, model training, evaluation, and prediction steps.

## Contributing

Contributions to UrbanFlow are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was developed as part of the ISEF project by Indradip Paul.
- Special thanks to my mentor for guidance and support.

## Contact

For questions or support, please contact [me](mailto:indradip.paul@outlook.com).
