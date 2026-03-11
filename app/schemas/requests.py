"""Request schemas for API endpoints."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Tuple
import re


class EnzymeData(BaseModel):
    """Enzyme data for prediction."""
    
    sequence: str = Field(
        ...,
        description="Protein sequence (amino acids)",
        min_length=10,
        max_length=2000
    )
    ec: Optional[str] = Field(
        None,
        description="EC number (e.g., '1.1.1.1')",
        pattern=r"^\d+\.\d+\.\d+\.\d+$"
    )
    organism: Optional[str] = Field(
        None,
        description="Organism name",
        max_length=200
    )
    n_measurements: int = Field(
        1,
        description="Number of measurements",
        ge=1
    )
    kcat_std: float = Field(
        0.0,
        description="Standard deviation of kcat",
        ge=0.0
    )
    kmValue: Optional[float] = Field(
        None,
        description="Michaelis constant (Km)",
        gt=0.0
    )
    ph_opt: Optional[float] = Field(
        None,
        description="Optimal pH",
        ge=0.0,
        le=14.0
    )
    temp_opt: Optional[float] = Field(
        None,
        description="Optimal temperature (°C)",
        ge=0.0,
        le=150.0
    )
    molecularWeight: Optional[float] = Field(
        None,
        description="Molecular weight (Da)",
        gt=0.0
    )
    
    @field_validator('sequence')
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        """Validate protein sequence contains only valid amino acids."""
        v = v.upper().strip()
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        invalid_chars = set(v) - valid_aa
        if invalid_chars:
            raise ValueError(f"Invalid amino acids in sequence: {invalid_chars}")
        return v


class EnzymePredictionRequest(BaseModel):
    """Request for single enzyme prediction."""
    
    enzyme: EnzymeData = Field(..., description="Enzyme data")
    dataset_name: str = Field(
        ...,
        description="Dataset/model name to use",
        pattern=r"^(gst|laccase|lactase)$"
    )


class BatchPredictionRequest(BaseModel):
    """Request for batch enzyme predictions."""
    
    enzymes: List[EnzymeData] = Field(
        ...,
        description="List of enzyme data",
        min_length=1,
        max_length=100
    )
    dataset_name: str = Field(
        ...,
        description="Dataset/model name to use",
        pattern=r"^(gst|laccase|lactase)$"
    )


class BioprocessOptimizationRequest(BaseModel):
    """Request for bioprocess optimization."""
    
    enzyme: EnzymeData = Field(..., description="Enzyme data")
    dataset_name: str = Field(
        ...,
        description="Dataset/model name to use",
        pattern=r"^(gst|laccase|lactase)$"
    )
    ph_range: Optional[Tuple[float, float]] = Field(
        None,
        description="pH range for optimization (min, max)"
    )
    temp_range: Optional[Tuple[float, float]] = Field(
        None,
        description="Temperature range for optimization (min, max) in °C"
    )
    
    @field_validator('ph_range')
    @classmethod
    def validate_ph_range(cls, v: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """Validate pH range."""
        if v is not None:
            min_ph, max_ph = v
            if not (0 <= min_ph <= 14 and 0 <= max_ph <= 14):
                raise ValueError("pH values must be between 0 and 14")
            if min_ph >= max_ph:
                raise ValueError("min_ph must be less than max_ph")
        return v
    
    @field_validator('temp_range')
    @classmethod
    def validate_temp_range(cls, v: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """Validate temperature range."""
        if v is not None:
            min_temp, max_temp = v
            if not (0 <= min_temp <= 150 and 0 <= max_temp <= 150):
                raise ValueError("Temperature values must be between 0 and 150°C")
            if min_temp >= max_temp:
                raise ValueError("min_temp must be less than max_temp")
        return v
