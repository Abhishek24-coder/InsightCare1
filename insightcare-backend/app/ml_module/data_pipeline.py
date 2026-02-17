"""
Data Pipeline - Load, Clean, and Prepare Disease-Symptom Data
Author: Abhishek
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class DataPipeline:
    """
    Handles loading and preprocessing of disease-symptom datasets
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the data pipeline
        
        Args:
            data_dir: Path to directory containing CSV files
        """
        if data_dir is None:
            # Default to the data folder in ml_module
            self.data_dir = Path(__file__).parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        # Initialize data containers
        self.df_disease_symptoms = None
        self.df_symptom_severity = None
        self.df_descriptions = None
        self.df_precautions = None
        
        # Processed data
        self.symptoms_list = []
        self.diseases_list = []
        self.severity_dict = {}
        
    def load_data(self) -> bool:
        """
        Load all CSV files
        
        Returns:
            bool: True if all files loaded successfully
        """
        try:
            # Load main dataset
            self.df_disease_symptoms = pd.read_csv(
                self.data_dir / "dataset.csv"
            )
            print(f"✓ Loaded dataset.csv: {self.df_disease_symptoms.shape}")
            
            # Load symptom severity
            self.df_symptom_severity = pd.read_csv(
                self.data_dir / "Symptom-severity.csv"
            )
            print(f"✓ Loaded Symptom-severity.csv: {self.df_symptom_severity.shape}")
            
            # Load descriptions
            self.df_descriptions = pd.read_csv(
                self.data_dir / "symptom_Description.csv"
            )
            print(f"✓ Loaded symptom_Description.csv: {self.df_descriptions.shape}")
            
            # Load precautions
            self.df_precautions = pd.read_csv(
                self.data_dir / "symptom_precaution.csv"
            )
            print(f"✓ Loaded symptom_precaution.csv: {self.df_precautions.shape}")
            
            return True
            
        except FileNotFoundError as e:
            print(f"✗ Error loading data: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False
    
    def clean_symptom_name(self, symptom: str) -> str:
        """
        Clean and normalize symptom names
        
        Args:
            symptom: Raw symptom string
            
        Returns:
            Cleaned symptom name
        """
        if pd.isna(symptom) or symptom == '':
            return None
        
        # Remove leading/trailing spaces and underscores
        symptom = str(symptom).strip().strip('_').strip()
        
        # Replace spaces with underscores
        symptom = symptom.replace(' ', '_')
        
        # Convert to lowercase
        symptom = symptom.lower()
        
        return symptom if symptom else None
    
    def prepare_data(self) -> pd.DataFrame:
        """
        Clean and prepare the disease-symptoms dataset
        
        Returns:
            Cleaned dataframe with Disease and Symptoms columns
        """
        if self.df_disease_symptoms is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        df = self.df_disease_symptoms.copy()
        
        # Get all symptom columns (Symptom_1 through Symptom_17)
        symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]
        
        # Clean all symptom columns
        for col in symptom_cols:
            df[col] = df[col].apply(self.clean_symptom_name)
        
        print(f"\n✓ Cleaned {len(symptom_cols)} symptom columns")
        print(f"✓ Total diseases: {df['Disease'].nunique()}")
        print(f"✓ Total records: {len(df)}")
        
        return df
    
    def get_unique_symptoms(self, df: pd.DataFrame) -> List[str]:
        """
        Extract all unique symptoms from the dataset
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            List of unique symptoms
        """
        symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]
        
        # Collect all symptoms
        all_symptoms = set()
        for col in symptom_cols:
            symptoms = df[col].dropna().unique()
            all_symptoms.update(symptoms)
        
        self.symptoms_list = sorted(list(all_symptoms))
        print(f"\n✓ Found {len(self.symptoms_list)} unique symptoms")
        
        return self.symptoms_list
    
    def get_unique_diseases(self, df: pd.DataFrame) -> List[str]:
        """
        Extract all unique diseases from the dataset
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            List of unique diseases
        """
        self.diseases_list = sorted(df['Disease'].unique().tolist())
        print(f"✓ Found {len(self.diseases_list)} unique diseases")
        
        return self.diseases_list
    
    def create_severity_dict(self) -> Dict[str, int]:
        """
        Create a dictionary mapping symptoms to their severity weights
        
        Returns:
            Dictionary with symptom: severity_weight pairs
        """
        if self.df_symptom_severity is None:
            raise ValueError("Severity data not loaded. Call load_data() first.")
        
        # Clean symptom names in severity dataframe
        df_severity = self.df_symptom_severity.copy()
        df_severity['Symptom'] = df_severity['Symptom'].apply(self.clean_symptom_name)
        
        # Create dictionary
        self.severity_dict = dict(zip(
            df_severity['Symptom'], 
            df_severity['weight']
        ))
        
        print(f"\n✓ Created severity dictionary with {len(self.severity_dict)} symptoms")
        
        return self.severity_dict
    
    def get_disease_info(self, disease: str) -> Dict:
        """
        Get description and precautions for a disease
        
        Args:
            disease: Disease name
            
        Returns:
            Dictionary with description and precautions
        """
        info = {
            'disease': disease,
            'description': None,
            'precautions': []
        }
        
        # Get description
        if self.df_descriptions is not None:
            desc_row = self.df_descriptions[
                self.df_descriptions['Disease'] == disease
            ]
            if not desc_row.empty:
                info['description'] = desc_row.iloc[0]['Description']
        
        # Get precautions
        if self.df_precautions is not None:
            prec_row = self.df_precautions[
                self.df_precautions['Disease'] == disease
            ]
            if not prec_row.empty:
                precautions = []
                for i in range(1, 5):
                    col = f'Precaution_{i}'
                    if col in prec_row.columns:
                        prec = prec_row.iloc[0][col]
                        if pd.notna(prec) and str(prec).strip():
                            precautions.append(str(prec).strip())
                info['precautions'] = precautions
        
        return info
    
    def summarize_data(self):
        """Print summary statistics of the loaded data"""
        print("\n" + "="*60)
        print("DATA PIPELINE SUMMARY")
        print("="*60)
        
        if self.df_disease_symptoms is not None:
            print(f"\n📊 Main Dataset:")
            print(f"   • Total records: {len(self.df_disease_symptoms)}")
            print(f"   • Diseases: {self.df_disease_symptoms['Disease'].nunique()}")
            print(f"   • Columns: {len(self.df_disease_symptoms.columns)}")
        
        if self.symptoms_list:
            print(f"\n🔬 Symptoms:")
            print(f"   • Unique symptoms: {len(self.symptoms_list)}")
            print(f"   • Sample: {', '.join(self.symptoms_list[:5])}...")
        
        if self.diseases_list:
            print(f"\n🏥 Diseases:")
            print(f"   • Unique diseases: {len(self.diseases_list)}")
            print(f"   • Sample: {', '.join(self.diseases_list[:3])}...")
        
        if self.severity_dict:
            print(f"\n⚖️  Severity Weights:")
            print(f"   • Symptoms with severity: {len(self.severity_dict)}")
            print(f"   • Weight range: {min(self.severity_dict.values())} - {max(self.severity_dict.values())}")
        
        print("\n" + "="*60 + "\n")


def main():
    """Test the data pipeline"""
    print("Testing Data Pipeline...\n")
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Load data
    if not pipeline.load_data():
        print("Failed to load data!")
        return
    
    # Prepare data
    df_clean = pipeline.prepare_data()
    
    # Extract unique symptoms and diseases
    symptoms = pipeline.get_unique_symptoms(df_clean)
    diseases = pipeline.get_unique_diseases(df_clean)
    
    # Create severity dictionary
    severity_dict = pipeline.create_severity_dict()
    
    # Print summary
    pipeline.summarize_data()
    
    # Test getting disease info
    test_disease = diseases[0]
    info = pipeline.get_disease_info(test_disease)
    print(f"Sample Disease Info - {info['disease']}:")
    print(f"Description: {info['description'][:100]}...")
    print(f"Precautions: {info['precautions']}")


if __name__ == "__main__":
    main()
