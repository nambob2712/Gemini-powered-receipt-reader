"""
Receipt OCR System - Google Gemini Vision Backend
"""

import os
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from io import BytesIO

from PIL import Image


class ExpenseCategory(Enum):
    FOOD = "Food"
    HOUSEHOLD = "Household"
    TRANSPORTATION = "Transportation"
    ENTERTAINMENT = "Entertainment"
    HEALTHCARE = "Healthcare"
    CLOTHING = "Clothing"
    ELECTRONICS = "Electronics"
    UTILITIES = "Utilities"
    OFFICE = "Office"
    OTHER = "Other"


class ProcessingStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class LineItem:
    description: str
    quantity: float = 1.0
    unit_price: float = 0.0
    total_price: float = 0.0
    discount: float = 0.0


@dataclass
class ExtractedData:
    merchant_name: Optional[str] = None
    merchant_address: Optional[str] = None
    merchant_phone: Optional[str] = None
    date: Optional[datetime] = None
    time: Optional[str] = None
    total_amount: Optional[float] = None
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    tax_rate: Optional[str] = None
    discount_amount: Optional[float] = None
    payment_method: Optional[str] = None
    currency: str = "JPY"
    line_items: List[LineItem] = field(default_factory=list)
    raw_text: str = ""
    confidence_score: float = 0.0
    category: ExpenseCategory = ExpenseCategory.OTHER
    detected_language: str = "ja"
    ai_notes: str = ""


@dataclass
class ProcessingResult:
    status: ProcessingStatus
    data: Optional[ExtractedData] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


EXTRACTION_PROMPT = """Analyze this receipt image and extract all information.

Return ONLY valid JSON:
{
    "merchant_name": "Store name",
    "merchant_address": "Address or null",
    "merchant_phone": "Phone or null",
    "date": "YYYY-MM-DD",
    "time": "HH:MM or null",
    "currency": "JPY/USD/EUR",
    "line_items": [{"description": "Item", "quantity": 1, "unit_price": 0, "total_price": 0, "discount": 0}],
    "subtotal": 0,
    "tax_amount": 0,
    "tax_rate": "8% or null",
    "discount_amount": 0,
    "total_amount": 0,
    "payment_method": "Cash/Card or null",
    "detected_language": "ja/en",
    "category": "Food/Household/Transportation/Entertainment/Healthcare/Clothing/Electronics/Utilities/Office/Other",
    "confidence": 0.95,
    "notes": "Any observations"
}

Rules:
- Japanese: 合計=total, 小計=subtotal, 税=tax, 割引=discount
- Be precise with numbers
- Return ONLY JSON, no markdown"""


class ReceiptOCRSystem:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        # Try provided key first, then environment variables
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable, "
                "configure it in .streamlit/secrets.toml, or pass api_key parameter."
            )
        self.model_name = model
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            if not self.api_key:
                raise ValueError("API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            try:
                self._model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                raise ValueError(f"Failed to initialize model '{self.model_name}': {str(e)}. Try 'gemini-2.0-flash' or 'gemini-2.5-flash'.")
        return self._model
    
    def process_image(self, image: Union[str, Path, Image.Image]) -> ProcessingResult:
        start = time.time()
        errors, warnings = [], []
        
        try:
            # Load image
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            
            # Convert to RGB if needed
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Call Gemini API
            response = self.model.generate_content([EXTRACTION_PROMPT, img])
            
            # Parse response
            if not hasattr(response, 'text') or not response.text:
                raise ValueError("Empty response from API")
            
            text = response.text.strip()
            
            # Handle markdown code blocks
            if '```' in text:
                parts = text.split('```')
                for part in parts:
                    if 'json' in part.lower() or '{' in part:
                        text = part
                        if text.lower().startswith('json'):
                            text = text[4:].strip()
                        break
            
            # Clean up text
            text = text.strip()
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            # Try to extract JSON if wrapped in other text
            if '{' in text and '}' in text:
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                text = text[json_start:json_end]
            
            result = json.loads(text)
            data = self._parse_result(result)
            status = self._get_status(data)
            
            if status == ProcessingStatus.PARTIAL:
                warnings.append("Some fields could not be extracted")
            
            return ProcessingResult(
                status=status,
                data=data,
                errors=errors,
                warnings=warnings,
                processing_time_ms=(time.time() - start) * 1000
            )
            
        except json.JSONDecodeError as e:
            errors.append(f"Failed to parse JSON response: {str(e)}")
            if 'text' in locals():
                errors.append(f"Response text: {text[:200]}...")
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                errors=errors,
                processing_time_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            error_msg = str(e)
            errors.append(error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                errors=errors,
                processing_time_ms=(time.time() - start) * 1000
            )
    
    def _parse_result(self, result: Dict[str, Any]) -> ExtractedData:
        # Parse date
        date_obj = None
        if result.get('date'):
            try:
                date_obj = datetime.strptime(result['date'], '%Y-%m-%d')
            except (ValueError, TypeError):
                pass
        
        # Parse line items
        line_items = []
        for item in result.get('line_items', []):
            if isinstance(item, dict):
                line_items.append(LineItem(
                    description=item.get('description', ''),
                    quantity=float(item.get('quantity') or 1),
                    unit_price=float(item.get('unit_price') or 0),
                    total_price=float(item.get('total_price') or 0),
                    discount=float(item.get('discount') or 0)
                ))
        
        # Parse category
        cat_str = result.get('category', 'Other')
        try:
            category = ExpenseCategory(cat_str)
        except ValueError:
            category = ExpenseCategory.OTHER
        
        return ExtractedData(
            merchant_name=result.get('merchant_name'),
            merchant_address=result.get('merchant_address'),
            merchant_phone=result.get('merchant_phone'),
            date=date_obj,
            time=result.get('time'),
            total_amount=float(result['total_amount']) if result.get('total_amount') else None,
            subtotal=float(result['subtotal']) if result.get('subtotal') else None,
            tax_amount=float(result['tax_amount']) if result.get('tax_amount') else None,
            tax_rate=result.get('tax_rate'),
            discount_amount=float(result['discount_amount']) if result.get('discount_amount') else None,
            payment_method=result.get('payment_method'),
            currency=result.get('currency', 'JPY'),
            line_items=line_items,
            raw_text=json.dumps(result, ensure_ascii=False, indent=2),
            confidence_score=float(result.get('confidence', 0.95)),
            category=category,
            detected_language=result.get('detected_language', 'ja'),
            ai_notes=result.get('notes', '')
        )
    
    def _get_status(self, data: ExtractedData) -> ProcessingStatus:
        has_total = data.total_amount is not None and data.total_amount > 0
        has_date = data.date is not None
        has_merchant = data.merchant_name is not None
        
        if data.confidence_score >= 0.9 and has_total:
            return ProcessingStatus.SUCCESS
        elif has_total and has_date:
            return ProcessingStatus.SUCCESS
        elif has_total or has_date or has_merchant:
            return ProcessingStatus.PARTIAL
        return ProcessingStatus.FAILED
