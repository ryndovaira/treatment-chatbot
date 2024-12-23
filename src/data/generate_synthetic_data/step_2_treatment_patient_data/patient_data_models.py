from typing import List, Optional

from pydantic import BaseModel, Field


class Medication(BaseModel):
    name: str = Field(..., description="Name of the medication")
    dosage: str = Field(..., description="Dosage of the medication (e.g., '500 mg')")
    frequency: str = Field(..., description="Frequency of intake (e.g., 'Once daily')")
    duration: str = Field(..., description="Duration of usage (e.g., '6 months', 'Ongoing')")


class TreatmentHistoryEntry(BaseModel):
    date_started: str = Field(
        ..., description="Start date of the treatment (ISO format YYYY-MM-DD)"
    )
    medications: List[Medication] = Field(
        ..., description="List of medications used during this period"
    )
    reason_for_change: Optional[str] = Field(
        None, description="Reason for changing medications, if applicable"
    )


class PatientData(BaseModel):
    current_medications: List[Medication] = Field(
        ..., description="List of current medications with details"
    )
    treatment_history: List[TreatmentHistoryEntry] = Field(
        ..., description="Patient's treatment history"
    )
    lifestyle_recommendations: List[str] = Field(
        ..., description="List of lifestyle recommendations for the patient"
    )
