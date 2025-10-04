# Viggo Frontend - Intelligent Reading Assistant

A Streamlit-based frontend for the Viggo RAG system that provides an intuitive interface for document upload, reading progress tracking, and AI-powered question answering with spoiler protection.

## 🚀 Features

### 📚 Document Management
- **Upload Support**: PDF and EPUB file upload with validation
- **Processing Status**: Real-time document processing feedback
- **Document Information**: Detailed document metadata and statistics

### 📖 Reading Progress
- **Progress Tracking**: Set and update current reading page
- **Visual Progress**: Interactive progress charts and statistics
- **Reading Status**: Mark books as finished to disable spoiler protection

### 🛡️ Spoiler Protection
- **Automatic Protection**: Queries limited to pages you've read
- **Configurable**: Enable/disable based on reading status
- **Context-Aware**: Smart suggestions based on current progress

### 💬 Intelligent Queries
- **AI-Powered**: Advanced RAG system for accurate answers
- **Multiple Search Methods**: Hybrid, semantic, and keyword search
- **Source References**: Page numbers and relevance scores
- **Query History**: Track all your questions and answers

### 📊 Analytics & Visualization
- **Reading Statistics**: Progress metrics and reading patterns
- **Query Analytics**: Question frequency and success rates
- **System Status**: Real-time backend health monitoring

## 🏗️ Architecture

```
frontend/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── components/               # Reusable UI components
│   ├── document_upload.py   # Document upload interface
│   ├── reading_progress.py  # Progress tracking interface
│   └── query_interface.py   # Question answering interface
├── pages/                   # Streamlit pages
│   ├── home.py             # Home page
│   ├── document_upload.py  # Document upload page
│   ├── reading_progress.py # Reading progress page
│   ├── query_interface.py  # Query interface page
│   └── document_info.py    # Document information page
└── utils/                   # Utility modules
    ├── api_client.py       # Backend API client
    └── session_manager.py  # Session state management
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Viggo backend running on `http://localhost:8000`

### Setup
1. **Install Dependencies**:
   ```bash
   cd frontend
   pip install -r requirements.txt
   ```

2. **Configure Backend URL** (optional):
   ```bash
   export VIGGO_API_URL="http://localhost:8000"
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the Interface**:
   Open your browser to `http://localhost:8501`

## 📱 Usage Workflow

### 1. Document Upload
1. Navigate to "Upload Document"
2. Select a PDF or EPUB file (max 50MB)
3. Click "Upload & Process Document"
4. Wait for processing to complete

### 2. Reading Progress Setup
1. Go to "Reading Progress"
2. Set your current page number
3. Indicate if you've finished the book
4. Save your progress

### 3. Ask Questions
1. Navigate to "Ask Questions"
2. Type your question about the book
3. Get AI-powered answers with source references
4. View query history and statistics

### 4. Update Progress
1. Return to "Reading Progress" as you read
2. Update your current page
3. Mark as finished when complete
4. Disable spoiler protection

## 🔧 Configuration

### Environment Variables
- `VIGGO_API_URL`: Backend API URL (default: `http://localhost:8000`)

### Configuration Options
Edit `config.py` to customize:
- File upload limits
- Supported formats
- Query parameters
- UI settings

## 🎯 Key Components

### Document Upload Component
- File validation and size checking
- Processing status feedback
- Document information display
- Error handling and user feedback

### Reading Progress Component
- Progress setup and updates
- Visual progress charts
- Reading statistics
- Spoiler protection controls

### Query Interface Component
- Question input and validation
- AI-powered answer display
- Source page references
- Query history management

### API Client
- RESTful API communication
- Error handling and retries
- Request/response validation
- Connection status monitoring

### Session Manager
- State persistence across pages
- Progress tracking
- Query history storage
- User preferences

## 🔍 API Integration

The frontend communicates with the Viggo backend through RESTful APIs:

- **Health Check**: `/api/v1/health/`
- **Document Upload**: `/api/v1/documents/upload`
- **RAG Queries**: `/api/v1/rag/query`
- **System Status**: `/api/v1/rag/system`
- **Content Processing**: `/api/v1/content/*`

## 🎨 UI/UX Features

### Responsive Design
- Mobile-friendly interface
- Adaptive layouts
- Touch-friendly controls

### Interactive Elements
- Progress bars and charts
- Real-time status updates
- Smooth page transitions
- Loading indicators

### User Experience
- Intuitive navigation
- Clear error messages
- Helpful tooltips and guides
- Consistent design language

## 🚀 Development

### Running in Development Mode
```bash
streamlit run app.py --server.runOnSave true
```

### Adding New Components
1. Create component in `components/`
2. Import and use in pages
3. Update navigation as needed

### Customizing Styling
- Modify `config.py` for theme settings
- Use Streamlit's theming capabilities
- Add custom CSS if needed

## 📊 Monitoring

### Health Checks
- Backend connectivity monitoring
- API response time tracking
- Error rate monitoring

### User Analytics
- Query frequency tracking
- Reading progress analytics
- Feature usage statistics

## 🔒 Security

### Data Protection
- No persistent data storage
- Session-based state management
- Secure API communication

### Input Validation
- File type and size validation
- Query length limits
- XSS protection

## 🐛 Troubleshooting

### Common Issues

**Backend Connection Failed**:
- Ensure Viggo backend is running
- Check API URL configuration
- Verify network connectivity

**File Upload Fails**:
- Check file size (max 50MB)
- Verify file format (PDF/EPUB)
- Ensure backend has sufficient storage

**Queries Not Working**:
- Verify document is processed
- Check spoiler protection settings
- Ensure backend is healthy

### Debug Mode
```bash
streamlit run app.py --logger.level debug
```

## 📈 Future Enhancements

### Planned Features
- **Multi-document Support**: Handle multiple books simultaneously
- **Reading Groups**: Share progress with friends
- **Advanced Analytics**: Detailed reading insights
- **Mobile App**: Native mobile application
- **Offline Mode**: Work without internet connection

### Technical Improvements
- **Performance Optimization**: Faster loading and queries
- **Caching**: Improved response times
- **Real-time Updates**: Live progress synchronization
- **Advanced Search**: More sophisticated query capabilities

## 📄 License

This project is part of the Viggo system. See the main project license for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation
- Open an issue on GitHub
- Contact the development team
