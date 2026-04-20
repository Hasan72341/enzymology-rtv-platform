"""Example Python client for the Enzyme API."""
import requests
from typing import Dict, List


class EnzymeAPIClient:
    """Client for interacting with the Enzyme Activity Prediction API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.api_v1 = f"{base_url}/api/v1"
    
    def predict_single(
        self,
        sequence: str,
        dataset_name: str,
        ec: str = None,
        organism: str = None,
        **kwargs
    ) -> Dict:
        """Predict enzyme activity for a single sequence.
        
        Args:
            sequence: Protein sequence
            dataset_name: Model to use (gst, laccase, or lactase)
            ec: EC number (optional)
            organism: Organism name (optional)
            **kwargs: Additional enzyme parameters
            
        Returns:
            Prediction response
        """
        enzyme_data = {
            "sequence": sequence,
            "ec": ec,
            "organism": organism,
            **kwargs
        }
        
        payload = {
            "enzyme": enzyme_data,
            "dataset_name": dataset_name
        }
        
        response = requests.post(
            f"{self.api_v1}/predict/single",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(
        self,
        enzymes: List[Dict],
        dataset_name: str
    ) -> List[Dict]:
        """Predict enzyme activity for multiple sequences.
        
        Args:
            enzymes: List of enzyme data dictionaries
            dataset_name: Model to use
            
        Returns:
            List of prediction responses
        """
        payload = {
            "enzymes": enzymes,
            "dataset_name": dataset_name
        }
        
        response = requests.post(
            f"{self.api_v1}/predict/batch",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    
    def rank_enzymes(
        self,
        enzymes: List[Dict],
        dataset_name: str
    ) -> Dict:
        """Rank enzymes by predicted activity.
        
        Args:
            enzymes: List of enzyme data dictionaries
            dataset_name: Model to use
            
        Returns:
            Ranking response with top enzymes
        """
        payload = {
            "enzymes": enzymes,
            "dataset_name": dataset_name
        }
        
        response = requests.post(
            f"{self.api_v1}/rank",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    
    def get_models(self) -> List[Dict]:
        """Get list of available models.
        
        Returns:
            List of model information
        """
        response = requests.get(f"{self.api_v1}/models")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict:
        """Check API health.
        
        Returns:
            Health status
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


#  Example usage
if __name__ == "__main__":
    # Initialize client
    client = EnzymeAPIClient()
    
    # Check health
    print("Health check:")
    print(client.health_check())
    print()
    
    # Get available models
    print("Available models:")
    models = client.get_models()
    for model in models:
        print(f"  - {model['model_name']}: {'available' if model['available'] else 'not available'}")
    print()
    
    # Single prediction example
    print("Single prediction:")
    sequence = "MKALSKLKAEEGIWMTDVPVPELGHNDLLIKIRKTAICGTDVHIYNWDEWSQKTIPVPMVVGHEYVGEVVGIGQEVKGFKIGDRVSGEGHITCGHCRNCRGGRTHLCRNTIGVGVNRPGCFAEYLVIPAFNAFKIPDNISDDLAAIFDPFGNAVHTALSFDLVGEDVLVSGAGPIGIMAAAVAKHVGARNVVITDVNEYRLELARKMGITRAVNVAKENLNDVMAELGMTEGFDVGLEMSGAPPAFRTMLDTMNHGGRIAMLGIPPSDMSIDWTKVIFKGLFIKGIYGREMFETWYKMAALIQSGLDLSPIITHRFSIDDFQKGFDAMRSGQSGKVILSWD"
    
    try:
        result = client.predict_single(
            sequence=sequence,
            dataset_name="gst",
            ec="1.1.1.103",
            organism="Cupriavidus necator",
            ph_opt=7.25,
            temp_opt=37.0
        )
        print(f"  Predicted log(kcat): {result['predicted_log_kcat']:.3f}")
        print(f"  Model: {result['model_name']}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    # Batch prediction example
    print("Batch prediction:")
    enzymes = [
        {
            "sequence": sequence[:200],  # Shortened for example
            "ec": "1.1.1.103",
            "organism": "Organism A"
        },
        {
            "sequence": sequence[50:250],  # Another variant
            "ec": "1.1.1.103",
            "organism": "Organism B"
        }
    ]
    
    try:
        results = client.predict_batch(enzymes, "gst")
        for i, result in enumerate(results, 1):
            print(f"  Enzyme {i}: log(kcat) = {result['predicted_log_kcat']:.3f}")
    except Exception as e:
        print(f"  Error: {e}")
