from src.data_loading import load_data
from src.data_preprocessing import preprocess_data
from src.eda import plot_loan_approval_distribution
from src.model_training import train_r_model, cross_validate_rf
from src.model_training import split_data, scale_data, train_logistic_model

from src.model_evaluation import evaluate_model
from src.utils import setup_logging, log_info, log_error
from pprint import pformat


def main():
    setup_logging()
    
    try:
        # Load data
        df = load_data('src/dataset/credit.csv')
        log_info('Data loaded successfully.')
        
        # Preprocess data
        df = preprocess_data(df)
        log_info('Data preprocessed successfully.')
        
        # Exploratory Data Analysis
        plot_loan_approval_distribution(df)
        log_info('EDA completed')
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(df)
        log_info('Data split into train and test sets.')
        
        # Scale data
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
        log_info('Data scaled successfully.')
        
        # Train Logistic Regression model
        log_model = train_logistic_model(X_train_scaled, y_train)
        log_info('Logistic Regression model trained successfully.')
        
        # Train Random Forest model
        rf_model = train_r_model(X_train_scaled, y_train)
        log_info('Random Forest model trained successfully.')
        
        # Cross-validate Random Forest model
        rf_scores = cross_validate_rf(X_train_scaled, y_train)
        log_info(f'Random Forest Cross-Validation Scores: {rf_scores}')
        log_info(f'Random Forest Mean Accuracy: {rf_scores.mean()}')
        log_info(f'Random Forest Standard Deviation: {rf_scores.std()}')
        
        # Evaluate Logistic Regression model
        laccuracy, lcmatrix = evaluate_model(log_model, X_test_scaled, y_test)
        log_info(f'Logistic Regression evaluation completed.')
        log_info(f'Accuracy: {laccuracy}')
        log_info(f'Confusion Matrix:\n{pformat(lcmatrix)}')
        
        # Evaluate Random Forest model
        rf_accuracy, rf_conf_matrix = evaluate_model(rf_model, X_test_scaled, y_test)
        log_info(f'Random Forest evaluation completed. Accuracy: {rf_accuracy}')
        log_info(f'Random Forest Confusion Matrix:\n{pformat(rf_conf_matrix)}')
        
    except Exception as e:
        log_error(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
