"""
AASIST Inference Wrapper for Streamlit App
Provides clean interface for AASIST model predictions and XAI explanations
"""

import torch
import librosa
import numpy as np
from pathlib import Path
import sys

# File nằm tại vispoofdb/models/aasist/ — import model từ cùng thư mục
AASIST_PKG = Path(__file__).resolve().parent
BASE_DIR = AASIST_PKG.parents[2]  # project root (3 levels up)
sys.path.insert(0, str(AASIST_PKG))

from models.baseline import Full_AASIST_Model


class AASISTDetector:
    """AASIST Deep Learning Model Wrapper for inference"""
    
    def __init__(self, model_path):
        """
        Args:
            model_path: Path to AASIST model checkpoint (.pth file)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        self.model = Full_AASIST_Model().to(self.device)
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        self.model.eval()
    
    def _preprocess_audio(self, y, sr):
        """Preprocess audio for AASIST model"""
        # Resample to 16kHz if needed
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        
        # AASIST expects specific input shape
        # Convert to tensor and add batch dimension
        y_tensor = torch.FloatTensor(y).to(self.device)
        return y_tensor.unsqueeze(0)  # Add batch dimension
    
    def predict_audio(self, file_path):
        """
        Predict on audio file
        
        Returns:
            dict with keys: success, is_fake, confidence_ai, probs
        """
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=16000)
            
            # Preprocess
            audio_tensor = self._preprocess_audio(y, sr)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(audio_tensor)
                probs = torch.softmax(outputs, dim=1)
                
                # probs: [batch_size, 2] where [0]=real, [1]=fake
                prob_real = probs[0, 0].item()
                prob_fake = probs[0, 1].item()
            
            is_fake = prob_fake >= 0.5
            
            return {
                "success": True,
                "is_fake": is_fake,
                "confidence_ai": prob_fake,
                "confidence_real": prob_real,
                "probs": [prob_real, prob_fake],
                "details": [prob_fake],  # For compatibility with app
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_model_info(self):
        """Return model information for UI"""
        return {
            "name": "AASIST (Deep Learning)",
            "description": "Anti-Spoofing with Automatic Speaker Verification using Integrated Spectro-Temporal features",
            "device": str(self.device),
            "model_path": str(self.model_path),
        }


class AASISTXAIExplainer:
    """XAI explainer for AASIST using gradient-based methods"""
    
    def __init__(self, detector):
        self.detector = detector
        self.model = detector.model
        self.device = detector.device
    
    def get_feature_importance(self, audio_tensor):
        """
        Compute feature importance using gradient-based saliency
        
        Args:
            audio_tensor: Input audio tensor [1, audio_length]
        
        Returns:
            importance_scores: Saliency map
        """
        audio_tensor = audio_tensor.clone().detach().to(self.device)
        audio_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(audio_tensor)
        loss = output[0, 1]  # Gradient w.r.t fake class
        
        # Backward pass
        loss.backward()
        
        # Gradient magnitude = importance
        saliency = torch.abs(audio_tensor.grad).squeeze().cpu().detach().numpy()
        
        return saliency
    
    def explain(self, y_audio, sr_audio):
        """
        Generate explanation for audio sample
        
        Args:
            y_audio: Waveform array
            sr_audio: Sample rate
        
        Returns:
            dict with explanation data
        """
        try:
            # Resample if needed
            if sr_audio != 16000:
                y_audio = librosa.resample(y_audio, orig_sr=sr_audio, target_sr=16000)
            
            # Get saliency
            audio_tensor = torch.FloatTensor(y_audio).unsqueeze(0).to(self.device)
            saliency = self.get_feature_importance(audio_tensor)
            
            # Normalize to [0, 1]
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
            
            # Compute statistics
            top_k = min(10, len(saliency))
            top_indices = np.argsort(saliency)[-top_k:]
            
            return {
                "success": True,
                "saliency": saliency,
                "top_features": top_indices,
                "top_values": saliency[top_indices],
                "method": "Gradient-based Saliency",
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "Gradient-based Saliency",
            }
