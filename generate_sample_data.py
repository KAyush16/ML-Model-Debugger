import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_sample_dataset(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=3,
    class_sep=0.8,
    random_state=42
):
    """
    Generate a sample classification dataset.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Total number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        n_classes: Number of classes
        class_sep: Class separation (lower = harder)
        random_state: Random seed
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=random_state,
        flip_y=0.05  # Add 5% label noise
    )
    
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    df['target'] = y
    
    np.random.seed(random_state)
    df['category_A'] = np.random.choice(['Type1', 'Type2', 'Type3'], size=n_samples)
    df['category_B'] = np.random.choice(['High', 'Medium', 'Low'], size=n_samples)
    
    for col in feature_names[:3]:  # Add missing values to first 3 features
        missing_idx = np.random.choice(
            n_samples, 
            size=int(0.05 * n_samples), 
            replace=False
        )
        df.loc[missing_idx, col] = np.nan
    
    return df


if __name__ == "__main__":
    
    print("Generating easy 3-class dataset...")
    df_easy = generate_sample_dataset(
        n_samples=800,
        n_features=8,
        n_informative=6,
        n_classes=3,
        class_sep=1.2,
        random_state=42
    )
    df_easy.to_csv('sample_data_easy.csv', index=False)
    print(f"Saved: sample_data_easy.csv (shape: {df_easy.shape})")
    
    print("\nGenerating medium binary dataset...")
    df_medium = generate_sample_dataset(
        n_samples=1000,
        n_features=12,
        n_informative=7,
        n_classes=2,
        class_sep=0.8,
        random_state=123
    )
    df_medium.to_csv('sample_data_medium.csv', index=False)
    print(f"Saved: sample_data_medium.csv (shape: {df_medium.shape})")
    
    print("\nGenerating hard 4-class dataset...")
    df_hard = generate_sample_dataset(
        n_samples=1200,
        n_features=15,
        n_informative=8,
        n_classes=4,
        class_sep=0.5,
        random_state=456
    )
    df_hard.to_csv('sample_data_hard.csv', index=False)
    print(f"Saved: sample_data_hard.csv (shape: {df_hard.shape})")
    
    print("\n Sample datasets generated successfully!")
    print("\nYou can now upload these files to the ML Model Debugger app.")