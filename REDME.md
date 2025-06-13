# SecureSure Insurance AI Agent

A conversational AI agent for SecureSure Insurance that handles insurance claims, lead generation, and general FAQs through natural language processing and document validation.

## Features

- **Intent Recognition**: Automatically classifies user requests into four main flows
- **Document Validation**: OCR-based validation for insurance documents
- **State Management**: Conversation resumption and state persistence
- **Lead Capture**: Automated quote request processing
- **Multi-format Support**: Handles text, images, and structured data

## Supported Flows

1. **Accident Claims**: Collects car registration, civil ID, driver license, and police report
2. **Windshield Claims**: Collects documents plus damage photos and chassis verification
3. **Lead Capture**: Gathers vehicle information for insurance quotes
4. **General FAQ**: Answers common insurance questions

## Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR
- OpenAI API Key

### System Dependencies

#### Windows
```bash
# Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH or update TESSERACT_PATH in config
```

#### macOS
```bash
brew install tesseract
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

### Python Dependencies

1. Clone the repository:
```bash
git clone <repository-url>
cd securesure-agent
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

1. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

2. Update Tesseract path in `Config` class if needed:
```python
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows
# or
# TESSERACT_PATH = "/usr/bin/tesseract"  # Linux/macOS
```

## Usage

### Starting the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Interactive API documentation is available at `http://localhost:8000/docs`

### API Endpoints

#### Chat Endpoint
```bash
POST /chat
Content-Type: application/json

{
    "message": "I need to file an accident claim",
    "conversation_id": "optional-conversation-id"
}
```

#### File Upload Endpoint
```bash
POST /upload
Content-Type: multipart/form-data

file: [file-content]
file_type: "car_registration_copy"
conversation_id: "conversation-id-from-chat"
```

### Example Usage

#### Starting a Conversation
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "I had an accident and need to file a claim"}'
```

#### Uploading Documents
```bash
curl -X POST "http://localhost:8000/upload" \
     -F "file=@car_registration.jpg" \
     -F "file_type=car_registration_copy" \
     -F "conversation_id=your-conversation-id"
```

## Document Validation

### Supported Document Types

1. **Car Registration Copy**
   - Required keywords: "Plate", "License", "Owner", "Base Number", "Year of Manufacture"

2. **Civil ID Copy**
   - Required keywords: "Name", "Civil ID No", "Expiry Date", "Nationality", "Gender", "Birth Date"

3. **Driver License Copy**
   - Required keywords: "License No", "Date of Issue", "Date of Expiry", "Driving License"

4. **Windshield Damage Photo**
   - Uses edge detection to identify cracks and damage
   - Analyzes image for visible windshield damage

5. **Vehicle Chassis Number Photo**
   - Extracts chassis number using OCR
   - Validates against car registration data

## Testing

### Creating Test Documents

Create test images with the following characteristics:

#### Valid Documents
- `valid_car_reg.jpg`: Contains all required keywords
- `valid_civil_id.jpg`: Contains all required keywords  
- `valid_drivers_license.jpg`: Contains all required keywords
- `valid_damaged_windshield.jpg`: Clear photo showing cracks
- `chassis_photo_12345.jpg`: Clear chassis number "ABC12345XYZ"

#### Invalid Documents
- `invalid_car_reg_missing_keywords.jpg`: Missing "Owner" keyword
- `invalid_civil_id_blurry.jpg`: Too blurry for OCR
- `not_a_document.jpg`: Random image (like a cat)
- `invalid_undamaged_windshield.jpg`: Perfect windshield

### Testing Flows

#### 1. Accident Claim Flow
```bash
# Start conversation
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "I had a car accident"}'

# Upload each required document in sequence
curl -X POST "http://localhost:8000/upload" \
     -F "file=@valid_car_reg.jpg" \
     -F "file_type=car_registration_copy" \
     -F "conversation_id=CONVERSATION_ID"
```

#### 2. Lead Capture Flow
```bash
# Start conversation
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "I want to get a quote for my car"}'

# Answer questions in sequence
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "$25,000", "conversation_id": "CONVERSATION_ID"}'
```

## Architecture

### Core Components

1. **SecureSureAgent**: Main agent class handling all conversation logic
2. **ConversationState**: Manages conversation state and persistence
3. **Document Validation**: OCR and image processing for document verification
4. **Database Integration**: SQLite for lead storage
5. **FastAPI Server**: REST API endpoints

### Flow Management

The agent uses a state machine approach:
- Each conversation has a unique ID and persistent state
- Flow types determine the sequence of required steps
- Document validation gates progression through steps
- State is automatically saved and can be resumed

### Document Processing Pipeline

1. **Image Reception**: Accept uploaded files via multipart form
2. **Format Conversion**: Convert to OpenCV/PIL formats
3. **OCR Processing**: Extract text using Tesseract
4. **Keyword Validation**: Check for required document elements
5. **Special Processing**: Edge detection for damage assessment
6. **Response Generation**: Provide validation feedback

## Configuration Options

### Environment Variables
```bash
OPENAI_API_KEY=your-openai-api-key
DATABASE_PATH=securesure.db
STATE_FILE=conversation_states.json
TESSERACT_CMD=/usr/bin/tesseract
```

### Customizable Parameters
- OCR confidence thresholds
- Edge detection sensitivity for damage assessment
- Conversation timeout settings
- Document validation keywords

## Database Schema

### Leads Table
```sql
CREATE TABLE leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    car_value TEXT,
    car_make TEXT,
    car_type TEXT,
    car_model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### State Storage
Conversation states are stored in JSON format:
```json
{
  "conversation_id": {
    "conversation_id": "uuid",
    "flow_type": "accident_claim",
    "current_step": 2,
    "collected_data": {},
    "validated_documents": {
      "car_registration_copy": true
    },
    "created_at": "2024-01-01T00:00:00",
    "last_updated": "2024-01-01T00:00:00"
  }
}
```

## Error Handling

### Common Error Scenarios
1. **Invalid Document Format**: Non-image files or corrupted images
2. **OCR Failures**: Poor image quality or missing text
3. **Missing Keywords**: Documents lacking required information
4. **API Failures**: OpenAI API issues with fallback logic
5. **Database Errors**: Connection or storage issues

### Validation Responses
```json
{
  "is_valid": false,
  "reason": "Document is missing the 'Owner' keyword. Please upload a valid copy."
}
```

## Extending the Agent

### Adding New Document Types
1. Define new `DocumentType` enum value
2. Add validation keywords to `VALIDATION_KEYWORDS`
3. Implement validation logic in `validate_document`
4. Update flow logic to include new document

### Adding New Flows
1. Create new `FlowType` enum value
2. Add intent classification keywords
3. Implement flow handler method
4. Update main processing logic

### Custom Validation Logic
```python
def _validate_custom_document(self, extracted_text: str) -> Dict[str, Any]:
    # Custom validation logic here
    if custom_condition:
        return {"is_valid": True, "reason": "Document validated"}
    else:
        return {"is_valid": False, "reason": "Validation failed"}
```

## Production Considerations

### Security
- Implement proper authentication and authorization
- Sanitize file uploads and validate file types
- Use secure database connections
- Implement rate limiting

### Performance
- Add caching for frequently accessed data
- Optimize image processing for large files
- Use async processing for heavy operations
- Implement connection pooling

### Monitoring
- Add logging for all operations
- Monitor API response times
- Track validation success rates
- Set up health checks and alerts

### Scalability
- Use Redis for state management in production
- Implement horizontal scaling with load balancers
- Use cloud storage for document persistence
- Consider microservice architecture for large scale

## Troubleshooting

### Common Issues

#### Tesseract Not Found
```bash
# Install tesseract and update path
export TESSERACT_CMD=/usr/bin/tesseract
```

#### OpenAI API Errors
- Verify API key is set correctly
- Check API usage limits
- Ensure proper error handling for rate limits

#### Document Validation Failures
- Check image quality and resolution
- Verify required keywords are present
- Test with known good documents

#### State Persistence Issues
- Ensure write permissions for state file
- Check disk space availability
- Verify JSON format validity

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is proprietary software for SecureSure Insurance.

## Support

For technical support or questions about implementation:
- Create an issue in the repository
- Contact the development team
- Check the API documentation at `/docs`

---

**Note**: This implementation is a prototype for demonstration purposes. Production deployment requires additional security, monitoring, and scalability considerations.