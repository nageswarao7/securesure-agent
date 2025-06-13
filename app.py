import json
import uuid
import re
import sqlite3
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import cv2
import numpy as np
from PIL import Image
import pytesseract
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import openai
from fastapi.middleware.cors import CORSMiddleware

# Configuration
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
    DATABASE_PATH = "securesure.db"
    STATE_FILE = "conversation_states.json"
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update for your system

# Initialize OpenAI client
openai.api_key = Config.OPENAI_API_KEY

# Enums and Data Classes
class FlowType(Enum):
    ACCIDENT_CLAIM = "accident_claim"
    WINDSHIELD_CLAIM = "windshield_claim"
    LEAD_CAPTURE = "lead_capture"
    GENERAL_FAQ = "general_faq"

class DocumentType(Enum):
    CAR_REGISTRATION = "car_registration_copy"
    CIVIL_ID = "civil_id_copy"
    DRIVER_LICENSE = "driver_license_copy"
    POLICE_REPORT = "police_report_copy"
    DAMAGE_WINDSHIELD = "damage_windshield_photo"
    CHASSIS_NUMBER = "vehicle_chassis_number_photo"
    DEALER_STAMP = "windshield_dealer_stamp_photo"

@dataclass
class ConversationState:
    conversation_id: str
    flow_type: Optional[FlowType] = None
    current_step: int = 0
    collected_data: Dict[str, Any] = None
    validated_documents: Dict[str, bool] = None
    created_at: str = None
    last_updated: str = None

    def __post_init__(self):
        if self.collected_data is None:
            self.collected_data = {}
        if self.validated_documents is None:
            self.validated_documents = {}
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()

    def to_dict(self):
        """Convert to a dictionary that is JSON serializable"""
        data = asdict(self)
        if self.flow_type:
            data['flow_type'] = self.flow_type.value
        else:
            data['flow_type'] = None
        return data


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    flow_type: Optional[str] = None
    current_step: int = 0
    requires_file_upload: bool = False
    expected_file_type: Optional[str] = None

# Knowledge Base
KNOWLEDGE_BASE = {
    "business hours": "Our branches are open 9 AM to 5 PM, but I am available to help you 24/7!",
    "classic car": "Yes, we offer specialized insurance plans for classic cars. Would you like me to get you a quote for that?",
    "comprehensive insurance": "Comprehensive insurance typically covers damage to your car from non-collision events like theft, fire, or vandalism. It also includes windshield damage.",
    "contact": "You can reach us at our branches from 9 AM to 5 PM, or chat with me anytime!",
    "coverage": "We offer various insurance coverage options including liability, comprehensive, collision, and specialized coverage for classic cars.",
    "claim process": "To file a claim, I can help you right now! Just tell me if it's for an accident or windshield damage, and I'll guide you through the process.",
}

# Document Validation Keywords
VALIDATION_KEYWORDS = {
    DocumentType.CAR_REGISTRATION: ["Plate", "License", "Owner", "Base Number", "Year of Manufacture"],
    DocumentType.CIVIL_ID: ["Name", "Civil ID No", "Expiry Date", "Nationality", "Gender", "Birth Date"],
    DocumentType.DRIVER_LICENSE: ["License No", "Date of Issue", "Date of Expiry", "Driving License"],
}

class SecureSureAgent:
    def __init__(self):
        self.setup_database()
        self.conversation_states = self.load_states()
        
    def setup_database(self):
        """Initialize SQLite database for leads"""
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                car_value TEXT,
                car_make TEXT,
                car_type TEXT,
                car_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def load_states(self) -> Dict[str, ConversationState]:
        """Load conversation states from file"""
        if os.path.exists(Config.STATE_FILE):
            try:
                with open(Config.STATE_FILE, 'r') as f:
                    data = json.load(f)
                print("Loaded Conversation States:", data)  # Add this line
                loaded_states = {}
                for k, v in data.items():
                    # Convert flow_type back to Enum
                    if v.get('flow_type'):
                        v['flow_type'] = FlowType(v['flow_type'])
                    loaded_states[k] = ConversationState(**v)
                return loaded_states

            except Exception as e:
                print(f"Error loading states: {e}")
                return {}
        return {}
    
    def save_states(self):
        """Save conversation states to file"""
        data = {k: v.to_dict() for k, v in self.conversation_states.items()}
        print("Saving Conversation States:", data)  # Add this line
        with open(Config.STATE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_or_create_conversation(self, conversation_id: Optional[str] = None) -> ConversationState:
        """Get existing conversation or create new one"""
        print(f"Checking for conversation ID: {conversation_id}")
        print(f"Current conversation_states: {self.conversation_states.keys()}")
        if conversation_id and conversation_id in self.conversation_states:
            state = self.conversation_states[conversation_id]
            state.last_updated = datetime.now().isoformat()
            return state
        
        new_id = str(uuid.uuid4())
        state = ConversationState(conversation_id=new_id)
        self.conversation_states[new_id] = state
        self.save_states()
        return state
    
    def classify_intent(self, message: str) -> FlowType:
        """Use OpenAI to classify user intent"""
        prompt = f"""
        Classify the following user message into one of these categories:
        1. accident_claim - User wants to file an accident claim
        2. windshield_claim - User wants to file a windshield damage claim
        3. lead_capture - User wants to get a quote or is interested in insurance
        4. general_faq - User has general questions about insurance

        User message: "{message}"

        Respond with only the category name (accident_claim, windshield_claim, lead_capture, or general_faq).
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0
            )
            
            intent = response.choices[0].message.content.strip().lower()
            
            # Map to enum
            intent_mapping = {
                "accident_claim": FlowType.ACCIDENT_CLAIM,
                "windshield_claim": FlowType.WINDSHIELD_CLAIM,
                "lead_capture": FlowType.LEAD_CAPTURE,
                "general_faq": FlowType.GENERAL_FAQ
            }
            
            return intent_mapping.get(intent, FlowType.GENERAL_FAQ)
        except:
            # Fallback to keyword-based classification
            message_lower = message.lower()
            if any(word in message_lower for word in ["accident", "crash", "collision", "hit"]):
                return FlowType.ACCIDENT_CLAIM
            elif any(word in message_lower for word in ["windshield", "windscreen", "glass", "crack"]):
                return FlowType.WINDSHIELD_CLAIM
            elif any(word in message_lower for word in ["quote", "price", "cost", "insure", "policy"]):
                return FlowType.LEAD_CAPTURE
            else:
                return FlowType.GENERAL_FAQ
    
    def handle_faq(self, message: str) -> str:
        """Handle general FAQ questions"""
        message_lower = message.lower()
        
        for key, answer in KNOWLEDGE_BASE.items():
            if key in message_lower:
                if key == "classic car":
                    return answer  # This will transition to lead capture
                return answer
        
        # Use OpenAI for general responses
        try:
            prompt = f"""
            You are a helpful insurance customer service agent for SecureSure Insurance. 
            Answer the following question professionally and helpfully. If you don't know the specific answer, 
            suggest they contact customer service or ask if you can help them with a claim or quote.

            Question: {message}
            """

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except:
            return "I'm here to help! You can ask me about our services, file a claim, or get a quote. What would you like to do today?"
    
    def validate_document(self, file_type: str, file_content: bytes) -> Dict[str, Any]:
        """Validate uploaded documents"""
        try:
            # Convert bytes to image
            nparr = np.frombuffer(file_content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"is_valid": False, "reason": "Invalid image file. Please upload a valid image."}
            
            # Convert to PIL Image for OCR
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Extract text using OCR
            extracted_text = pytesseract.image_to_string(pil_image)
            
            doc_type = DocumentType(file_type)
            
            if doc_type in VALIDATION_KEYWORDS:
                return self._validate_text_document(doc_type, extracted_text)
            elif doc_type == DocumentType.DAMAGE_WINDSHIELD:
                return self._validate_windshield_damage(image)
            elif doc_type == DocumentType.CHASSIS_NUMBER:
                return self._validate_chassis_number(extracted_text, pil_image)
            else:
                return {"is_valid": True, "reason": "Document received successfully."}
                
        except Exception as e:
            return {"is_valid": False, "reason": f"Error processing document: {str(e)}"}
    
    def _validate_text_document(self, doc_type: DocumentType, extracted_text: str) -> Dict[str, Any]:
        """Validate text-based documents using OCR"""
        required_keywords = VALIDATION_KEYWORDS[doc_type]
        missing_keywords = []
        
        for keyword in required_keywords:
            if keyword.lower() not in extracted_text.lower():
                missing_keywords.append(keyword)
        
        if missing_keywords:
            return {
                "is_valid": False,
                "reason": f"Document is missing the following required information: {', '.join(missing_keywords)}. Please upload a valid copy."
            }
        
        return {"is_valid": True, "reason": "All required information found in the document."}
    
    def _validate_windshield_damage(self, image: np.ndarray) -> Dict[str, Any]:
        """Validate windshield damage using edge detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Count edge pixels
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_density = edge_pixels / total_pixels
            
            # If edge density is high, likely indicates cracks/damage
            if edge_density > 0.1:  # Threshold for damage detection
                return {"is_valid": True, "reason": "Windshield damage detected in the image."}
            else:
                return {"is_valid": False, "reason": "No significant damage visible in the image. Please upload a clear photo of the damaged windshield."}
                
        except Exception as e:
            return {"is_valid": False, "reason": f"Error analyzing image: {str(e)}"}
    
    def _validate_chassis_number(self, extracted_text: str, image: Image.Image) -> Dict[str, Any]:
        """Validate chassis number matches car registration"""
        # Extract chassis number from image
        chassis_patterns = [
            r'[A-Z0-9]{17}',  # Standard VIN format
            r'[A-Z]{3}\d{5}[A-Z]{3}',  # Custom format
        ]
        
        chassis_number = None
        for pattern in chassis_patterns:
            matches = re.findall(pattern, extracted_text)
            if matches:
                chassis_number = matches[0]
                break
        
        if not chassis_number:
            return {"is_valid": False, "reason": "Could not extract chassis number from the image. Please ensure the number is clearly visible."}
        
        # For demo purposes, assume validation passes
        # In real implementation, you would compare with stored car registration data
        return {"is_valid": True, "reason": f"Chassis number {chassis_number} validated successfully."}
    
    def save_lead_to_db(self, lead_data: Dict[str, Any], conversation_id: str) -> Dict[str, Any]:
        """Save lead data to database"""
        try:
            conn = sqlite3.connect(Config.DATABASE_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO leads (conversation_id, car_value, car_make, car_type, car_model)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                conversation_id,
                lead_data.get('car_value'),
                lead_data.get('car_make'),
                lead_data.get('car_type'),
                lead_data.get('car_model')
            ))
            
            conn.commit()
            lead_id = cursor.lastrowid
            conn.close()
            
            return {"success": True, "lead_id": lead_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def process_message(self, message: str, conversation_id: Optional[str] = None) -> ChatResponse:
        """Main message processing logic"""
        state = self.get_or_create_conversation(conversation_id)
        
        # If no flow is set, classify intent
        if state.flow_type is None:
            state.flow_type = self.classify_intent(message)
            state.current_step = 0
            self.save_states()

        # Convert FlowType to string for the ChatResponse
        flow_type_str = state.flow_type.value if state.flow_type else None
        
        # Handle different flows
        if state.flow_type == FlowType.GENERAL_FAQ:
            response_text = self.handle_faq(message)
            
            # Check if FAQ response should transition to lead capture
            if "Would you like me to get you a quote" in response_text:
                state.flow_type = FlowType.LEAD_CAPTURE
                state.current_step = 0
                response_text += " Let's start with some basic information about your vehicle."
                self.save_states()
                return self._get_next_lead_question(state, response_text)
            
            return ChatResponse(
                response=response_text,
                conversation_id=state.conversation_id,
                flow_type=flow_type_str,
                current_step=state.current_step
            )
        
        elif state.flow_type == FlowType.ACCIDENT_CLAIM:
            return self._handle_accident_claim(state, message)
        
        elif state.flow_type == FlowType.WINDSHIELD_CLAIM:
            return self._handle_windshield_claim(state, message)
        
        elif state.flow_type == FlowType.LEAD_CAPTURE:
            return self._handle_lead_capture(state, message)
    
    def _handle_accident_claim(self, state: ConversationState, message: str) -> ChatResponse:
        """Handle accident claim flow"""
        steps = [
            ("car_registration_copy", "I'll help you file your accident claim. First, please upload a copy of your car registration."),
            ("civil_id_copy", "Great! Now please upload a copy of your Civil ID."),
            ("driver_license_copy", "Perfect! Next, please upload a copy of your driver's license."),
            ("police_report_copy", "Excellent! Finally, please upload a copy of the police report.")
        ]
        
        if state.current_step == 0:
            state.current_step = 1
            self.save_states()
            return ChatResponse(
                response=steps[0][1],
                conversation_id=state.conversation_id,
                flow_type=FlowType.ACCIDENT_CLAIM.value,
                current_step=state.current_step,
                requires_file_upload=True,
                expected_file_type=steps[0][0]
            )
        
        if state.current_step <= len(steps):
            current_doc_type = steps[state.current_step - 1][0]
            
            if state.current_step < len(steps):
                next_message = steps[state.current_step][1]
                next_doc_type = steps[state.current_step][0]
                state.current_step += 1
                self.save_states()
                
                return ChatResponse(
                    response=next_message,
                    conversation_id=state.conversation_id,
                    flow_type=FlowType.ACCIDENT_CLAIM.value,
                    current_step=state.current_step,
                    requires_file_upload=True,
                    expected_file_type=next_doc_type
                )
            else:
                # All documents collected
                return ChatResponse(
                    response="Thank you! Your accident claim has been submitted successfully. You will receive a confirmation email shortly with your claim number. Is there anything else I can help you with?",
                    conversation_id=state.conversation_id,
                    flow_type=FlowType.ACCIDENT_CLAIM.value,
                    current_step=state.current_step
                )
    
    def _handle_windshield_claim(self, state: ConversationState, message: str) -> ChatResponse:
        """Handle windshield claim flow"""
        steps = [
            ("car_registration_copy", "I'll help you file your windshield claim. First, please upload a copy of your car registration."),
            ("civil_id_copy", "Great! Now please upload a copy of your Civil ID."),
            ("driver_license_copy", "Perfect! Next, please upload a copy of your driver's license."),
            ("damage_windshield_photo", "Excellent! Now please upload a photo of the damaged windshield."),
            ("vehicle_chassis_number_photo", "Good! Please upload a photo of your vehicle's chassis number."),
            ("windshield_dealer_stamp_photo", "Finally, please upload a photo of the windshield dealer's stamp.")
        ]
        
        if state.current_step == 0:
            state.current_step = 1
            self.save_states()
            return ChatResponse(
                response=steps[0][1],
                conversation_id=state.conversation_id,
                flow_type=FlowType.WINDSHIELD_CLAIM.value,
                current_step=state.current_step,
                requires_file_upload=True,
                expected_file_type=steps[0][0]
            )
        
        if state.current_step <= len(steps):
            if state.current_step < len(steps):
                next_message = steps[state.current_step][1]
                next_doc_type = steps[state.current_step][0]
                state.current_step += 1
                self.save_states()
                
                return ChatResponse(
                    response=next_message,
                    conversation_id=state.conversation_id,
                    flow_type=FlowType.WINDSHIELD_CLAIM.value,
                    current_step=state.current_step,
                    requires_file_upload=True,
                    expected_file_type=next_doc_type
                )
            else:
                return ChatResponse(
                    response="Thank you! Your windshield claim has been submitted successfully. You will receive a confirmation email shortly with your claim number. Is there anything else I can help you with?",
                    conversation_id=state.conversation_id,
                    flow_type=FlowType.WINDSHIELD_CLAIM.value,
                    current_step=state.current_step
                )
    
    def _handle_lead_capture(self, state: ConversationState, message: str) -> ChatResponse:
        """Handle lead capture flow"""
        questions = [
            ("car_value", "What is the estimated value of your car?"),
            ("car_make", "What is the make of the car, like Toyota, Ford?"),
            ("car_type", "Is it an SUV, Sedan, Coupe?"),
            ("car_model", "What year was the car manufactured?")
        ]
        
        if state.current_step == 0:
            return self._get_next_lead_question(state, "I'd be happy to get you a quote! Let me ask you a few questions about your vehicle.")
        
        # Store the answer to the previous question
        if state.current_step > 0:
            question_key = questions[state.current_step - 1][0]
            state.collected_data[question_key] = message
        
        # Check if all questions are answered
        if state.current_step >= len(questions):
            # Save lead to database
            result = self.save_lead_to_db(state.collected_data, state.conversation_id)
            
            if result.get("success"):
                response_text = f"Thank you! I've saved your information. Based on your {state.collected_data['car_make']} {state.collected_data['car_type']} from {state.collected_data['car_model']} valued at {state.collected_data['car_value']}, our team will prepare a customized quote for you. You'll receive it within 24 hours. Is there anything else I can help you with?"
            else:
                response_text = "I apologize, there was an issue saving your information. Please try again or contact our customer service."
            
            return ChatResponse(
                response=response_text,
                conversation_id=state.conversation_id,
                flow_type=FlowType.LEAD_CAPTURE.value,
                current_step=state.current_step
            )
        
        return self._get_next_lead_question(state)
    
    def _get_next_lead_question(self, state: ConversationState, prefix: str = "") -> ChatResponse:
        """Get the next question in lead capture flow"""
        questions = [
            ("car_value", "What is the estimated value of your car?"),
            ("car_make", "What is the make of the car, like Toyota, Ford?"),
            ("car_type", "Is it an SUV, Sedan, Coupe?"),
            ("car_model", "What year was the car manufactured?")
        ]
        
        if state.current_step < len(questions):
            question_text = questions[state.current_step][1]
            if prefix:
                response_text = f"{prefix} {question_text}"
            else:
                response_text = question_text
            
            state.current_step += 1
            self.save_states()
            
            return ChatResponse(
                response=response_text,
                conversation_id=state.conversation_id,
                flow_type=FlowType.LEAD_CAPTURE.value,
                current_step=state.current_step
            )
    
    def process_file_upload(self, file_content: bytes, file_type: str, conversation_id: str) -> Dict[str, Any]:
        """Process uploaded file"""
        print(f"Received upload request for conversation ID: {conversation_id}")
        print(f"Current conversation_states: {self.conversation_states.keys()}")
        if conversation_id not in self.conversation_states:
            return {"success": False, "message": "Invalid conversation ID"}
        
        state = self.conversation_states[conversation_id]
        
        # Validate the document
        validation_result = self.validate_document(file_type, file_content)
        
        if validation_result["is_valid"]:
            state.validated_documents[file_type] = True
            self.save_states()
            return {"success": True, "message": validation_result["reason"]}
        else:
            return {"success": False, "message": validation_result["reason"]}

# FastAPI Application
app = FastAPI(title="SecureSure Insurance Agent API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
agent = SecureSureAgent()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        response = agent.process_message(request.message, request.conversation_id)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    file_type: str = Form(...),
    conversation_id: str = Form(...)
):
    """File upload endpoint"""
    try:
        file_content = await file.read()
        result = agent.process_file_upload(file_content, file_type, conversation_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "SecureSure Insurance Agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)