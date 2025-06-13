import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import PersonalizationSettings from './PersonalizationSettings'
import './ChatBot.css';

// Cấu hình API Base URL
import { API_BASE_URL } from '../config.js';

const ChatBot = () => {
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [sessionId, setSessionId] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isTyping, setIsTyping] = useState(false);
    const [connectionStatus, setConnectionStatus] = useState('checking'); // checking, connected, error
    
    // ✅ NEW: Speech-to-Text states
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessingSpeech, setIsProcessingSpeech] = useState(false);
    const [speechSupported, setSpeechSupported] = useState(false);
    const [mediaRecorder, setMediaRecorder] = useState(null);
    const [recordingTime, setRecordingTime] = useState(0);
    
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);
    const audioChunks = useRef([]);
    const recordingInterval = useRef(null);

    // Thêm vào useState hooks
    const [showPersonalization, setShowPersonalization] = useState(false);
    const [user, setUser] = useState(null);
    const [personalizationEnabled, setPersonalizationEnabled] = useState(false);

    // Cấu hình axios với base URL và timeout
    useEffect(() => {
        axios.defaults.baseURL = API_BASE_URL;
        axios.defaults.timeout = 30000; // 30 seconds timeout
        
        // Test connection khi component mount
        testConnection();
        
        // ✅ NEW: Check speech support
        checkSpeechSupport();
        
        // ✅ Check user authentication status
        checkUserAuth(); 
        
        // ✅ NEW: Cleanup function
        return () => {
            if (recordingInterval.current) {
                clearInterval(recordingInterval.current);
            }
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
        };
    }, []);

    const testConnection = async () => {
        try {
            console.log('Testing backend connection...');
            const response = await axios.get('/api/health/');
            console.log('Backend connection successful:', response.data);
            setConnectionStatus('connected');
        } catch (error) {
            console.error('Backend connection failed:', error);
            setConnectionStatus('error');
            
            // Show connection error in chat
            const errorMessage = {
                type: 'bot',
                content: '⚠️ Không thể kết nối đến server backend. Vui lòng:\n\n1. Kiểm tra server Django đang chạy (python manage.py runserver)\n2. Kiểm tra URL: http://127.0.0.1:8000/api/health/\n3. Kiểm tra CORS settings\n\nServer phải chạy trên port 8000 để frontend hoạt động.',
                timestamp: new Date(),
                isError: true
            };
            setMessages([errorMessage]);
        }
    };

    const checkUserAuth = async () => {
        try {
            const token = localStorage.getItem('auth_token');
            const userData = localStorage.getItem('user_data');
            
            if (token && userData) {
                axios.defaults.headers.common['Authorization'] = `Token ${token}`;
                const parsedUser = JSON.parse(userData);
                setUser(parsedUser);
                setPersonalizationEnabled(true);
                
                // Load personalized context
                const contextResponse = await axios.get('/api/personalized-context/');
                if (contextResponse.data.personalization_enabled) {
                    console.log('Personalization enabled for user:', parsedUser.faculty_code);
                }
            }
        } catch (error) {
            console.log('User not authenticated or personalization not available');
        }
    };

    // ✅ NEW: Check speech support function
    const checkSpeechSupport = async () => {
        try {
            // Check if browser supports MediaRecorder
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                console.log('Browser does not support audio recording');
                setSpeechSupported(false);
                return;
            }

            // Check backend speech service
            const response = await axios.get('/api/speech-status/');
            const speechServiceAvailable = response.data.speech_service?.available || false;
            
            console.log('Speech service status:', response.data);
            setSpeechSupported(speechServiceAvailable);
            
            if (!speechServiceAvailable) {
                console.log('Backend speech service not available');
            }
        } catch (error) {
            console.error('Error checking speech support:', error);
            setSpeechSupported(false);
        }
    };

    // ✅ NEW: Start recording function
    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });
            
            // ✅ FIX: Use WAV format which is supported by Whisper
            let mimeType = 'audio/wav';
            
            // Check supported formats in preferred order
            const preferredFormats = [
                'audio/wav',
                'audio/mp4',
                'audio/webm;codecs=opus',
                'audio/webm'
            ];
            
            for (const format of preferredFormats) {
                if (MediaRecorder.isTypeSupported(format)) {
                    mimeType = format;
                    console.log('✅ Using MIME type:', format);
                    break;
                }
            }
            
            const recorder = new MediaRecorder(stream, {
                mimeType: mimeType,
                audioBitsPerSecond: 128000
            });
            
            audioChunks.current = [];
            
            recorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    console.log('🎵 Audio chunk received:', event.data.size, 'bytes');
                    audioChunks.current.push(event.data);
                }
            };
            
            recorder.onstop = async () => {
                console.log('🎵 Total chunks:', audioChunks.current.length);
                const totalSize = audioChunks.current.reduce((sum, chunk) => sum + chunk.size, 0);
                console.log('🎵 Total audio size:', totalSize, 'bytes');
                
                if (totalSize < 1024) {
                    console.error('❌ Audio too small:', totalSize, 'bytes');
                    showTemporaryMessage('❌ Audio quá ngắn. Vui lòng ghi âm lâu hơn.', 'speech-error');
                    setIsProcessingSpeech(false);
                    return;
                }
                
                const audioBlob = new Blob(audioChunks.current, { type: mimeType });
                console.log('🎵 Final blob:', audioBlob.size, 'bytes, type:', audioBlob.type);
                
                await processAudioBlob(audioBlob, mimeType);
                
                // Stop all audio tracks
                stream.getTracks().forEach(track => track.stop());
            };
            
            recorder.start(1000); // Record 1-second chunks
            setMediaRecorder(recorder);
            setIsRecording(true);
            setRecordingTime(0);
            
            recordingInterval.current = setInterval(() => {
                setRecordingTime(prev => prev + 1);
            }, 1000);
            
            console.log('🎤 Recording started with MIME:', mimeType);
        } catch (error) {
            console.error('❌ Error starting recording:', error);
            showTemporaryMessage('❌ Không thể truy cập microphone. Vui lòng cho phép truy cập và thử lại.', 'speech-error');
        }
    };
    
    // ✅ NEW: Stop recording function
    const stopRecording = () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            setIsRecording(false);
            setIsProcessingSpeech(true);
            
            if (recordingInterval.current) {
                clearInterval(recordingInterval.current);
                recordingInterval.current = null;
            }
            
            console.log('Recording stopped');
        }
    };

    // ✅ NEW: Process audio blob function
    const processAudioBlob = async (audioBlob, originalMimeType) => {
        try {
            const formData = new FormData();
            
            // ✅ Determine file extension and name based on format
            let fileName = 'recording';
            let fileExtension = '.wav';
            
            if (originalMimeType.includes('wav')) {
                fileExtension = '.wav';
            } else if (originalMimeType.includes('mp4')) {
                fileExtension = '.m4a';
            } else if (originalMimeType.includes('webm')) {
                fileExtension = '.webm'; // Will be converted on backend
            }
            
            fileName += fileExtension;
            
            formData.append('audio', audioBlob, fileName);
            formData.append('language', 'vi');
            formData.append('original_format', originalMimeType);
            
            console.log('🎤 Sending audio:', fileName, audioBlob.size, 'bytes');
            console.log('🎤 Original MIME type:', originalMimeType);
            
            const response = await axios.post('/api/speech-to-text/', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                timeout: 60000
            });
            
            console.log('🔍 FULL SPEECH RESPONSE:', JSON.stringify(response.data, null, 2));
            
            if (response.data.success && response.data.text) {
                const transcribedText = response.data.text.trim();
                console.log('✅ Transcribed text:', transcribedText);
                setInputMessage(prev => prev + (prev ? ' ' : '') + transcribedText);
                
                if (inputRef.current) {
                    inputRef.current.focus();
                }
                
                showTemporaryMessage(`🎤 "${transcribedText}"`, 'speech-success');
            } else {
                console.error('❌ Speech failed - Success:', response.data.success);
                console.error('❌ Speech failed - Text:', response.data.text);
                console.error('❌ Speech failed - Error:', response.data.error);
                
                // Show specific error message
                const errorMsg = response.data.error || 'Không nhận diện được giọng nói';
                showTemporaryMessage(`❌ ${errorMsg}`, 'speech-error');
            }
        } catch (error) {
            console.error('❌ Error processing speech:', error);
            console.error('❌ Error response:', error.response?.data);
            
            if (error.response?.status === 413) {
                showTemporaryMessage('❌ File audio quá lớn. Vui lòng ghi âm ngắn hơn.', 'speech-error');
            } else if (error.code === 'ECONNABORTED') {
                showTemporaryMessage('❌ Timeout xử lý giọng nói. Vui lòng thử lại.', 'speech-error');
            } else {
                showTemporaryMessage('❌ Lỗi xử lý giọng nói. Vui lòng thử lại.', 'speech-error');
            }
        } finally {
            setIsProcessingSpeech(false);
        }
    };

    // ✅ NEW: Show temporary message function
    const showTemporaryMessage = (content, type) => {
        const tempMessage = {
            type: 'system',
            content: content,
            timestamp: new Date(),
            temporary: true,
            messageType: type
        };
        
        setMessages(prev => [...prev, tempMessage]);
        
        // Remove temporary message after 3 seconds
        setTimeout(() => {
            setMessages(prev => prev.filter(msg => !msg.temporary));
        }, 3000);
    };

    // ✅ NEW: Format recording time function
    const formatRecordingTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    useEffect(() => {
        if (connectionStatus === 'connected') {
            // Generate unique session ID
            const newSessionId = 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
            setSessionId(newSessionId);
            
            // ✅ UPDATED: Welcome message với speech info
            const welcomeMessage = {
                type: 'bot',
                content: `Xin chào! Tôi là chatbot của Đại học Bình Dương. Tôi có thể giúp bạn tìm hiểu về:

• Thông tin tuyển sinh
• Các ngành đào tạo
• Học phí và chính sách hỗ trợ
• Đời sống sinh viên
• Cơ sở vật chất

${speechSupported ? '🎤 Bạn có thể gõ hoặc nói để hỏi câu hỏi!' : 'Bạn có câu hỏi gì muốn hỏi không?'}`,
                timestamp: new Date(),
                confidence: 1.0
            };
            
            setMessages([welcomeMessage]);
        }
    }, [connectionStatus, speechSupported]);

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    const sendMessage = async () => {
        if (!inputMessage.trim() || isLoading || connectionStatus !== 'connected') return;

        const userMessage = {
            type: 'user',
            content: inputMessage.trim(),
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setIsLoading(true);
        setIsTyping(true);
        
        const messageToSend = inputMessage.trim();
        setInputMessage('');

        try {
            console.log('Sending message:', messageToSend);
            console.log('Session ID:', sessionId);
            console.log('API URL:', `${API_BASE_URL}/api/chat/`);
            
            const response = await axios.post('/api/chat/', {
                message: messageToSend,
                session_id: sessionId
            }, {
                headers: {
                    'Content-Type': 'application/json',
                },
                timeout: 30000
            });

            console.log('Response received:', response.data);

            // Simulate typing delay for better UX
            setTimeout(() => {
                const botMessage = {
                    type: 'bot',
                    content: response.data.response,
                    confidence: response.data.confidence,
                    sources: response.data.sources || [],
                    method: response.data.method,
                    response_time: response.data.response_time,
                    timestamp: new Date(),
                    chat_id: Date.now() // For feedback purposes
                };

                setMessages(prev => [...prev, botMessage]);
                setIsTyping(false);
            }, 1000);

        } catch (error) {
            console.error('Error sending message:', error);
            
            let errorContent = 'Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn.';
            
            if (error.code === 'ECONNABORTED') {
                errorContent = 'Timeout: Server mất quá nhiều thời gian để phản hồi. Vui lòng thử lại.';
            } else if (error.response) {
                // Server responded with error status
                errorContent = `Lỗi server (${error.response.status}): ${error.response.data?.error || error.response.statusText}`;
            } else if (error.request) {
                // Request was made but no response received
                errorContent = 'Không thể kết nối đến server. Vui lòng kiểm tra:\n\n1. Server Django đang chạy\n2. URL backend: http://127.0.0.1:8000\n3. CORS settings';
            }
            
            setTimeout(() => {
                const errorMessage = {
                    type: 'bot',
                    content: errorContent,
                    timestamp: new Date(),
                    isError: true
                };
                setMessages(prev => [...prev, errorMessage]);
                setIsTyping(false);
            }, 1000);
            
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const sendFeedback = async (chatId, feedbackType, comment = '') => {
        try {
            await axios.post('/api/feedback/', {
                chat_id: chatId,
                feedback_type: feedbackType,
                comment: comment
            });
            
            // Update message to show feedback was sent
            setMessages(prev => prev.map(msg => 
                msg.chat_id === chatId 
                    ? { ...msg, feedbackSent: feedbackType }
                    : msg
            ));
            
        } catch (error) {
            console.error('Error sending feedback:', error);
        }
    };

    const formatMessage = (content) => {
        // ✅ INTELLIGENT TEXT FORMATTER - VERSION 2.0 
        if (!content) return null;
        
        // Normalize line breaks first
        let formattedContent = content.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
        
        // 🔥 ENHANCED: Detect numbered sections pattern (1. 2. 3. etc.)
        formattedContent = formattedContent.replace(/(\d+)\.\s*([^0-9\n][^\n]*)/g, (match, num, text) => {
            return `\n\n<div class="format-section-number">
                <div class="section-number-header">
                    <span class="section-number">${num}.</span>
                    <strong class="section-number-title">${text.trim()}</strong>
                </div>
            </div>\n`;
        });
        
        // 🔥 NEW: Detect emoji-prefixed sections (📚, 💼, ✅, etc.)
        formattedContent = formattedContent.replace(/([\u{1F300}-\u{1F9FF}])\s*([^:\n]+):/gu, (match, emoji, title) => {
            return `\n\n<div class="format-emoji-section">
                <div class="emoji-section-header">
                    <span class="emoji-icon">${emoji}</span>
                    <strong class="emoji-section-title">${title.trim()}:</strong>
                </div>
                <div class="emoji-section-content">`;
        });
        
        // 🔥 NEW: Close emoji sections before next section or end
        formattedContent = formattedContent.replace(/(<div class="emoji-section-content">[\s\S]*?)(?=\n\n<div class="format-|$)/g, '$1</div></div>');
        
        // 🎯 FORMAT BOLD TEXT (**text**)
        formattedContent = formattedContent.replace(/\*\*([^*\n]+)\*\*/g, '<strong class="format-bold">$1</strong>');
        
        // 🎯 FORMAT BULLET POINTS (•, -, *, với multiline support)
        formattedContent = formattedContent.replace(/^[\s]*[•\-\*]\s+(.+)$/gm, '<div class="format-bullet"><span class="bullet-icon">•</span><span class="bullet-text">$1</span></div>');
        
        // 🎯 FORMAT SUB-BULLETS (indented bullets)
        formattedContent = formattedContent.replace(/^[\s]{2,}[•\-\*]\s+(.+)$/gm, '<div class="format-sub-bullet"><span class="sub-bullet-icon">▸</span><span class="sub-bullet-text">$1</span></div>');
        
        // 🔥 NEW: Format questions and answers
        formattedContent = formattedContent.replace(/([?？])\s*([A-ZÁÊÔƠƯĐ][^?]*)/g, '$1</div><div class="format-answer"><strong>Trả lời:</strong> $2');
        
        // 🎯 FORMAT INLINE CODE (`code`)
        formattedContent = formattedContent.replace(/`([^`\n]+)`/g, '<code class="format-code">$1</code>');
        
        // 🔥 NEW: Format important keywords
        formattedContent = formattedContent.replace(/\b(thầy\/cô|giảng viên|học phí|tuyển sinh|đăng ký|liên hệ|quan trọng|lưu ý|chú ý|hạn chót|deadline)\b/gi, '<span class="format-keyword">$1</span>');
        
        // 🎯 FORMAT LINKS
        formattedContent = formattedContent.replace(/(https?:\/\/[^\s<>]+)/g, '<a href="$1" target="_blank" class="format-link">$1</a>');
        
        // 🎯 FORMAT PHONE NUMBERS  
        formattedContent = formattedContent.replace(/(\b0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b)/g, '<a href="tel:$1" class="format-phone">📞 $1</a>');
        
        // 🎯 FORMAT EMAIL
        formattedContent = formattedContent.replace(/(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)/g, '<a href="mailto:$1" class="format-email">✉️ $1</a>');
        
        // 🔥 NEW: Format common academic terms
        formattedContent = formattedContent.replace(/\b(BDU|Đại học Bình Dương)\b/g, '<span class="format-university">$1</span>');
        
        // 🎯 CLEAN UP: Remove excessive whitespace but preserve structure
        formattedContent = formattedContent.replace(/\n{3,}/g, '\n\n');
        
        // 🎯 CONVERT LINE BREAKS to <br> (except in formatted blocks)
        formattedContent = formattedContent.replace(/\n(?![^<]*>)/g, '<br/>');
        
        // 🔥 NEW: Add spacing after sections
        formattedContent = formattedContent.replace(/(<\/div>)(<div class="format-)/g, '$1<br/>$2');
        
        return <div 
            className="formatted-content" 
            dangerouslySetInnerHTML={{ __html: formattedContent }}
        />;
    };

    const getConfidenceColor = (confidence) => {
        if (confidence >= 0.8) return '#4CAF50'; // Green
        if (confidence >= 0.6) return '#FF9800'; // Orange
        return '#F44336'; // Red
    };

    const clearChat = () => {
        const newSessionId = 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
        setSessionId(newSessionId);
        
        const welcomeMessage = {
            type: 'bot',
            content: 'Cuộc trò chuyện mới đã bắt đầu. Bạn có câu hỏi gì muốn hỏi không?',
            timestamp: new Date(),
            confidence: 1.0
        };
        
        setMessages([welcomeMessage]);
    };

    const retryConnection = () => {
        setConnectionStatus('checking');
        setMessages([]);
        testConnection();
    };

    return (
        <div className="chatbot-container">
            <div className="chatbot-header">
                <div className="header-content">
                    <h3>🤖 AI Assistant</h3>
                    <div className="header-actions">
                        <div className="connection-status">
                            {connectionStatus === 'checking' && <span className="status checking">🔄 Đang kết nối...</span>}
                            {connectionStatus === 'connected' && <span className="status connected">🟢 Đã kết nối</span>}
                            {connectionStatus === 'error' && (
                                <span className="status error" onClick={retryConnection} style={{cursor: 'pointer'}}>
                                    🔴 Lỗi kết nối (Click để thử lại)
                                </span>
                            )}
                        </div>
                        
                        {/* ✅ THÊM: Personalization button */}
                        {personalizationEnabled && user && (
                            <button 
                                className="settings-btn"
                                onClick={() => setShowPersonalization(true)}
                                title="Cài đặt cá nhân hóa"
                            >
                                🎯 Settings
                            </button>
                        )}

                        {/* ✅ NEW: Speech status indicator */}
                        {speechSupported && (
                            <div className="speech-status">
                                🎤 Voice enabled
                            </div>
                        )}

                        <button 
                            className="clear-btn"
                            onClick={clearChat}
                            title="Bắt đầu cuộc trò chuyện mới"
                            disabled={connectionStatus !== 'connected'}
                        >
                            🗑️
                        </button>
                    </div>
                </div>
                <div className="session-info">
                    Session: {sessionId.split('_')[1] || 'N/A'}
                </div>
            </div>
            
            <div className="messages-container">
                {messages.map((message, index) => (
                    <div 
                        key={index} 
                        className={`message ${message.type} ${message.isError ? 'error' : ''} ${message.temporary ? 'temporary' : ''} ${message.messageType || ''}`}
                    >
                        <div className="message-content">
                            <div className="message-text">
                                {formatMessage(message.content)}
                            </div>
                            
                            {message.type === 'bot' && !message.isError && !message.temporary && (
                                <div className="message-metadata">
                                    {/* Existing metadata */}
                                    
                                    {/* New hybrid metadata */}
                                    {message.intent && (
                                        <div className="intent-info">
                                            🎯 Ý định: {message.intent.description || message.intent.intent}
                                            <span className="intent-confidence">
                                                ({(message.intent.confidence * 100).toFixed(1)}%)
                                            </span>
                                        </div>
                                    )}
                                    
                                    {message.entities && Object.keys(message.entities).length > 0 && (
                                        <div className="entities-info">
                                            🏷️ Nhận diện: {Object.entries(message.entities).map(([key, value]) => 
                                                `${key}: ${value}`
                                            ).join(', ')}
                                        </div>
                                    )}
                                    
                                    {message.strategy && (
                                        <div className="strategy-info">
                                            ⚙️ Chiến lược: {
                                                message.strategy === 'generation' ? 'Sinh văn bản' :
                                                message.strategy === 'hybrid' ? 'Kết hợp' :
                                                message.strategy === 'retrieval' ? 'Tìm kiếm' : message.strategy
                                            }
                                        </div>
                                    )}
                                </div>
                            )}
                            
                            {message.sources && message.sources.length > 0 && (
                                <div className="sources">
                                    <div className="sources-title">📚 Nguồn tham khảo:</div>
                                    {message.sources.slice(0, 2).map((source, idx) => (
                                        <div key={idx} className="source-item">
                                            <div className="source-question">
                                                Q: {source.question}
                                            </div>
                                            <div className="source-similarity">
                                                Độ tương đồng: {(source.similarity * 100).toFixed(1)}%
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                            
                            {message.type === 'bot' && !message.isError && !message.temporary && message.chat_id && (
                                <div className="feedback-buttons">
                                    {!message.feedbackSent ? (
                                        <>
                                            <button
                                                className="feedback-btn like"
                                                onClick={() => sendFeedback(message.chat_id, 'like')}
                                                title="Hữu ích"
                                            >
                                                👍
                                            </button>
                                            <button
                                                className="feedback-btn dislike"
                                                onClick={() => sendFeedback(message.chat_id, 'dislike')}
                                                title="Không hữu ích"
                                            >
                                                👎
                                            </button>
                                        </>
                                    ) : (
                                        <div className="feedback-sent">
                                            {message.feedbackSent === 'like' ? '👍 Đã đánh giá' : '👎 Đã đánh giá'}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                        
                        <div className="message-time">
                            {message.timestamp.toLocaleTimeString('vi-VN')}
                        </div>
                    </div>
                ))}
                
                {isTyping && (
                    <div className="message bot typing">
                        <div className="message-content">
                            <div className="typing-indicator">
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                            <div className="typing-text">AI đang suy nghĩ...</div>
                        </div>
                    </div>
                )}
                
                <div ref={messagesEndRef} />
            </div>
            
            <div className="input-container">
                {/* ✅ NEW: Recording Status Display */}
                {(isRecording || isProcessingSpeech) && (
                    <div className="recording-status">
                        {isRecording && (
                            <div className="recording-indicator">
                                🔴 Đang ghi âm... {formatRecordingTime(recordingTime)}
                                <div className="recording-animation">
                                    <div className="pulse"></div>
                                </div>
                            </div>
                        )}
                        {isProcessingSpeech && (
                            <div className="processing-indicator">
                                🔄 Đang xử lý giọng nói...
                            </div>
                        )}
                    </div>
                )}

                <div className="input-wrapper">
                    <textarea
                        ref={inputRef}
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder={connectionStatus === 'connected' 
                            ? (speechSupported 
                                ? "Nhập hoặc nói câu hỏi... (Enter để gửi, Shift+Enter để xuống dòng)"
                                : "Nhập câu hỏi của bạn... (Enter để gửi, Shift+Enter để xuống dòng)")
                            : "Đang kết nối đến server..."}
                        rows="2"
                        disabled={isLoading || connectionStatus !== 'connected' || isRecording}
                        maxLength={1000}
                    />
                    <div className="input-actions">
                        <div className="char-count">
                            {inputMessage.length}/1000
                        </div>
                        
                        {/* ✅ NEW: Speech-to-Text Button */}
                        {speechSupported && (
                            <button 
                                className={`speech-button ${isRecording ? 'recording' : ''} ${isProcessingSpeech ? 'processing' : ''}`}
                                onClick={isRecording ? stopRecording : startRecording}
                                disabled={isLoading || connectionStatus !== 'connected' || isProcessingSpeech}
                                title={isRecording ? 'Dừng ghi âm' : 'Bắt đầu ghi âm'}
                            >
                                {isRecording ? '⏹️' : (isProcessingSpeech ? '⏳' : '🎤')}
                            </button>
                        )}
                        
                        <button 
                            onClick={sendMessage}
                            disabled={isLoading || !inputMessage.trim() || connectionStatus !== 'connected' || isRecording}
                            className="send-button"
                            title="Gửi tin nhắn"
                        >
                            {isLoading ? '⏳' : '📤'}
                        </button>
                    </div>
                </div>
            </div>
            
            <div className="chatbot-footer">
                <div className="footer-info">
                    <span>🤖 Powered by AI • PhoBERT + Whisper + FAISS</span>
                    <span>Made with ❤️ for Đại học Bình Dương</span>
                </div>
                <div className="debug-info" style={{fontSize: '10px', color: '#666', marginTop: '5px'}}>
                    Backend: {API_BASE_URL} | Status: {connectionStatus} | Speech: {speechSupported ? 'Yes' : 'No'}
                </div>
            </div>

            {/* ✅ THÊM: PersonalizationSettings Modal */}
            {showPersonalization && (
                <PersonalizationSettings
                    user={user}
                    onClose={() => setShowPersonalization(false)}
                    onUpdateSuccess={(newData) => {
                        console.log('Personalization updated:', newData);
                        // Có thể refresh welcome message hoặc update UI
                    }}
                />
            )}
        </div>
    );
};

export default ChatBot;