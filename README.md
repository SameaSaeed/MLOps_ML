# End-to-End Machine Learning

Welcome to the End-to-End Machine Learning project. This repository provides a comprehensive solution for building, training, evaluating, and deploying machine learning models. It covers the full data science workflow from data preprocessing through to model deployment and monitoring, making it an ideal starting point for both experimentation and production-level ML pipelines.

## Features

- Complete machine learning pipeline: data preprocessing, feature engineering, model selection, training, evaluation, and deployment.
- Modular architecture for easy extension and customization.
- Automated data validation and cleaning.
- Support for multiple machine learning algorithms.
- Integrated model evaluation and reporting.
- Scalable deployment options for various environments.
- Configurable through YAML or JSON files.
- Logging and monitoring for production readiness.

## Requirements

To run this project, make sure your environment meets the following requirements:

- Python 3.7 or above
- pip (latest version recommended)
- Common ML/data science libraries:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - joblib
  - pyyaml or json (for configuration)
- (Optional) Flask or FastAPI if using the deployment APIs
- (Optional) Docker for containerized deployment

You can install all core dependencies using the provided requirements file.

## Project Architecture

Below is an overview of the pipeline structure and data flow:

<img width="450" height="450" alt="579441890-2ecba885-612b-4069-9ee9-7412dae0433f" src="https://github.com/user-attachments/assets/fd419de7-31b8-4256-9bd0-62a348e08c90" />

## Installation
Follow these steps to set up the project on your local machine:

1. **Clone the repository**

    ```bash
    git clone https://github.com/SameaSaeed/End-to_End_ML.git
    cd End-to_End_ML
    ```

2. **Create a virtual environment (optional but recommended)**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the required packages**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up configuration files**

    - Copy and modify the example configuration as needed.

## Usage

This project is structured to be run step-by-step or as an integrated pipeline. The typical workflow is:

1. **Prepare your data:** Place your dataset in the `data/` directory.
2. **Configure your pipeline:** Edit the configuration file (`config.yaml` or `config.json`) to specify data sources, preprocessing, model parameters, and outputs.
3. **Run the pipeline:** Execute the main script to launch the pipeline.

```bash
python main.py --config config.yaml
```

4. **Results and outputs:** Processed data, models, and reports will be saved to the `outputs/` directory.

### Example Workflow

```bash
python main.py --config configs/classification_example.yaml
```

#### Data Processing

- Automatic cleaning, missing value handling, and feature engineering.

#### Model Training

- Choose from multiple algorithms by editing the configuration.

#### Evaluation

- View accuracy, confusion matrix, and other metrics in the console and saved reports.

#### Deployment

- Serve the best model using the provided API scripts or Dockerfile.

#### AB testing
Run both models in separate terminals:

Terminal 1
python3 app1.py

Terminal 2
python3 app2.py

Configure Nginx for A/B Traffic Split sudo nano /etc/nginx/sites-available/ab_testing
Enable the config: sudo ln -s /etc/nginx/sites-available/ab_testing /etc/nginx/sites-enabled/ sudo nginx -t sudo systemctl restart nginx

Test A/B traffic routing: curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{}'

## Output

<img width="1734" height="780" alt="Batch_predict" src="https://github.com/user-attachments/assets/a1c7a06c-b5af-42e2-a278-fcffbd126545" />

## Contact

For questions or support, please open an issue on GitHub or contact the maintainer directly through their GitHub profile.
