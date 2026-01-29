"""
Simple Example: Quick Model Inference
======================================
A minimal example showing how to load and use the saved SVM model
"""

import joblib
import json
import pandas as pd
from pathlib import Path


def quick_predict(variant_data, model_dir='models'):
    """
    Quick prediction function - minimal code
    
    Parameters:
    -----------
    variant_data : dict
        e.g., {'chrom': 'chr7', 'pos': 12345, 'ref': 'A', 'alt': 'T', 
               'gene': 'CFTR', 'sfari_score': 3.0}
    
    Returns:
    --------
    dict with prediction and confidence
    """
    
    # 1. Load model and scaler
    model = joblib.load(f'{model_dir}/svm_model.joblib')
    scaler = joblib.load(f'{model_dir}/svm_scaler.joblib')
    
    with open(f'{model_dir}/svm_features.json', 'r') as f:
        features = json.load(f)['feature_names']
    
    # 2. Preprocess variant
    def encode_chrom(chrom):
        chrom_str = str(chrom).replace('chr', '').upper()
        mapping = {'X': 23, 'Y': 24, 'M': 25, 'MT': 25}
        return mapping.get(chrom_str, int(chrom_str) if chrom_str.isdigit() else -1)
    
    allele_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Create feature vector
    feature_values = {
        'chrom_enc': encode_chrom(variant_data['chrom']),
        'ref_enc': allele_map.get(variant_data['ref'].upper(), -1),
        'alt_enc': allele_map.get(variant_data['alt'].upper(), -1),
        'gene_freq': variant_data.get('gene_freq', 0.01),
        'gene_count': variant_data.get('gene_count', 10),
        'sfari_score': variant_data.get('sfari_score', 1.0)
    }
    
    # 3. Create DataFrame with correct feature order
    X = pd.DataFrame([feature_values])[features]
    
    # 4. Scale and predict
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    
    return {
        'prediction': int(prediction),
        'label': 'Pathogenic' if prediction == 1 else 'Non-pathogenic',
    }


# Example usage
if __name__ == "__main__":
    
    # Example 1: Single variant
    variant = {
        'chrom': 'chr7',
        'pos': 117199563,
        'ref': 'C',
        'alt': 'T',
        'gene': 'CFTR',
        'sfari_score': 3.0
    }
    
    result = quick_predict(variant)
    print(f"\nVariant: {variant['chrom']}:{variant['pos']} {variant['ref']}>{variant['alt']}")
    print(f"Gene: {variant['gene']}")
    print(f"Prediction: {result['label']}")
    
    # Example 2: Another variant
    variant2 = {
        'chrom': 'chr1',
        'pos': 12345678,
        'ref': 'A',
        'alt': 'G',
        'gene': 'EXAMPLE_GENE',
        'sfari_score': 1.5
    }
    
    result2 = quick_predict(variant2)
    print(f"\nVariant: {variant2['chrom']}:{variant2['pos']} {variant2['ref']}>{variant2['alt']}")
    print(f"Gene: {variant2['gene']}")
    print(f"Prediction: {result2['label']}")
    
    # Example 3: Batch prediction
    variants = [
        {'chrom': 'chr1', 'pos': 100000, 'ref': 'A', 'alt': 'T', 'gene': 'GENE1', 'sfari_score': 2.0},
        {'chrom': 'chr2', 'pos': 200000, 'ref': 'C', 'alt': 'G', 'gene': 'GENE2', 'sfari_score': 1.5},
        {'chrom': 'chrX', 'pos': 300000, 'ref': 'G', 'alt': 'A', 'gene': 'GENE3', 'sfari_score': 3.5},
    ]
    
    print("\n" + "="*60)
    print("BATCH PREDICTIONS:")
    print("="*60)
    
    for var in variants:
        res = quick_predict(var)
        print(f"{var['gene']:15s} | {res['label']:15s} |")