# Synthetic Healthcare Data Generation with tVAE

## Project Overview

This project implements a Tabular Variational Autoencoder (tVAE) for generating synthetic healthcare data that preserves the statistical properties and relationships of the original dataset while ensuring privacy. It was developed as part of my External Pharmacy Internship at the CHU (University Hospital) of Strasbourg.

The implementation focuses on the Right Heart Catheterization (RHC) dataset with an emphasis on data quality, proper handling of missing values, and utility preservation.

## Key Features

- **Miniaturized tVAE Implementation**: Lightweight version of the tVAE architecture optimized for tabular medical data
- **Advanced Data Preprocessing**: Specialized handling of categorical and continuous variables
- **Missing Values Management**: Proper handling of null values using the 'from_column' approach
- **Privacy Preservation**: Generated synthetic data that protects patient privacy while maintaining statistical utility (in coming)

## About the Project

This project represents an exploration into privacy-preserving synthetic data generation methods for healthcare applications at CHU Strasbourg. The goal is to create high-quality synthetic data that can be safely shared and used for research purposes without compromising patient privacy.

By using a Variational Autoencoder approach, we can generate new synthetic patient records that maintain the complex relationships between medical variables while ensuring no actual patient data is revealed.

## Next Steps

Further development will focus on enhancing the model's performance, expanding the evaluation metrics, and testing with additional healthcare datasets.

---

*This project is part of an External Pharmacy Internship at CHU Strasbourg.*
