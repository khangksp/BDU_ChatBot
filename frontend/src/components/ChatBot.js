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
    const [connectionStatus, setConnectionStatus] = useState('checking');
    
    // Speech-to-Text states
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessingSpeech, setIsProcessingSpeech] = useState(false);
    const [speechSupported, setSpeechSupported] = useState(false);
    const [mediaRecorder, setMediaRecorder] = useState(null);
    const [recordingTime, setRecordingTime] = useState(0);
    
    // Reference sources state
    const [expandedSources, setExpandedSources] = useState(new Set());
    
    // UI States
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [chatSessions, setChatSessions] = useState([
        { id: 1, title: 'Đoạn chat 1', active: true },
        { id: 2, title: 'Đoạn chat 2', active: false }
    ]);
    
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);
    const audioChunks = useRef([]);
    const recordingInterval = useRef(null);

    // Personalization states
    const [showPersonalization, setShowPersonalization] = useState(false);
    const [user, setUser] = useState(null);
    const [personalizationEnabled, setPersonalizationEnabled] = useState(false);

    // Cấu hình axios với base URL và timeout
    useEffect(() => {
        axios.defaults.baseURL = API_BASE_URL;
        axios.defaults.timeout = 30000;
        
        testConnection();
        checkSpeechSupport();
        checkUserAuth(); 
        
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
            
            const errorMessage = {
                type: 'bot',
                content: '⚠️ Không thể kết nối đến server backend. Vui lòng kiểm tra kết nối.',
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
                
                const contextResponse = await axios.get('/api/personalized-context/');
                if (contextResponse.data.personalization_enabled) {
                    console.log('Personalization enabled for user:', parsedUser.faculty_code);
                }
            }
        } catch (error) {
            console.log('User not authenticated or personalization not available');
        }
    };

    const checkSpeechSupport = async () => {
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                console.log('Browser does not support audio recording');
                setSpeechSupported(false);
                return;
            }

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
            
            let mimeType = 'audio/wav';
            
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
                
                stream.getTracks().forEach(track => track.stop());
            };
            
            recorder.start(1000);
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

    const processAudioBlob = async (audioBlob, originalMimeType) => {
        try {
            const formData = new FormData();
            
            let fileName = 'recording';
            let fileExtension = '.wav';
            
            if (originalMimeType.includes('wav')) {
                fileExtension = '.wav';
            } else if (originalMimeType.includes('mp4')) {
                fileExtension = '.m4a';
            } else if (originalMimeType.includes('webm')) {
                fileExtension = '.webm';
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
    
    const showTemporaryMessage = (content, type) => {
        const tempMessage = {
            type: 'system',
            content: content,
            timestamp: new Date(),
            temporary: true,
            messageType: type
        };
        
        setMessages(prev => [...prev, tempMessage]);
        
        setTimeout(() => {
            setMessages(prev => prev.filter(msg => !msg.temporary));
        }, 3000);
    };

    const formatRecordingTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const toggleSources = (messageIndex) => {
        const newExpanded = new Set(expandedSources);
        if (newExpanded.has(messageIndex)) {
            newExpanded.delete(messageIndex);
        } else {
            newExpanded.add(messageIndex);
        }
        setExpandedSources(newExpanded);
    };

    useEffect(() => {
        if (connectionStatus === 'connected') {
            const newSessionId = 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
            setSessionId(newSessionId);
            
            const welcomeMessage = {
                type: 'bot',
                content: `Xin chào! Tôi là trợ lý AI của Đại học Bình Dương. Tôi có thể giúp bạn:

• 📚 Thông tin tuyển sinh và ngành học
• 💰 Học phí và chính sách hỗ trợ  
• 🎓 Đời sống sinh viên
• 🏢 Cơ sở vật chất và tiện ích
• 📞 Thông tin liên hệ

${speechSupported ? '🎤 Bạn có thể gõ hoặc nói để đặt câu hỏi!' : 'Hãy đặt câu hỏi để bắt đầu!'}`,
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
            const response = await axios.post('/api/chat/', {
                message: messageToSend,
                session_id: sessionId
            }, {
                headers: { 'Content-Type': 'application/json' },
                timeout: 30000
            });

            const finalReferenceLinks = response.data.reference_links || [];

            setTimeout(() => {
                const botMessage = {
                    type: 'bot',
                    content: response.data.response,
                    confidence: response.data.confidence,
                    sources: response.data.sources || [],
                    method: response.data.method,
                    response_time: response.data.response_time,
                    timestamp: new Date(),
                    chat_id: Date.now(),
                    reference_links: finalReferenceLinks
                };

                console.log('🎯 FINAL BOTMESSAGE:', botMessage);
                setMessages(prev => [...prev, botMessage]);
                setIsTyping(false);
            }, 1000);

        } catch (error) {
            console.error('Error:', error);
            setTimeout(() => {
                const errorMessage = {
                    type: 'bot',
                    content: 'Xin lỗi, đã có lỗi xảy ra.',
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
        if (!content) return null;
        
        let formattedContent = content.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
        
        // Enhanced formatting patterns
        formattedContent = formattedContent.replace(/(\d+)\.\s*([^0-9\n][^\n]*)/g, (match, num, text) => {
            return `\n\n<div class="format-section-number">
                <div class="section-number-header">
                    <span class="section-number">${num}.</span>
                    <strong class="section-number-title">${text.trim()}</strong>
                </div>
            </div>\n`;
        });
        
        formattedContent = formattedContent.replace(/([\u{1F300}-\u{1F9FF}])\s*([^:\n]+):/gu, (match, emoji, title) => {
            return `\n\n<div class="format-emoji-section">
                <div class="emoji-section-header">
                    <span class="emoji-icon">${emoji}</span>
                    <strong class="emoji-section-title">${title.trim()}:</strong>
                </div>
                <div class="emoji-section-content">`;
        });
        
        formattedContent = formattedContent.replace(/(<div class="emoji-section-content">[\s\S]*?)(?=\n\n<div class="format-|$)/g, '$1</div></div>');
        
        formattedContent = formattedContent.replace(/\*\*([^*\n]+)\*\*/g, '<strong class="format-bold">$1</strong>');
        formattedContent = formattedContent.replace(/^[\s]*[•\-\*]\s+(.+)$/gm, '<div class="format-bullet"><span class="bullet-icon">•</span><span class="bullet-text">$1</span></div>');
        formattedContent = formattedContent.replace(/^[\s]{2,}[•\-\*]\s+(.+)$/gm, '<div class="format-sub-bullet"><span class="sub-bullet-icon">▸</span><span class="sub-bullet-text">$1</span></div>');
        formattedContent = formattedContent.replace(/([?？])\s*([A-ZÁÊÔƠƯĐ][^?]*)/g, '$1</div><div class="format-answer"><strong>Trả lời:</strong> $2');
        formattedContent = formattedContent.replace(/`([^`\n]+)`/g, '<code class="format-code">$1</code>');
        formattedContent = formattedContent.replace(/\b(thầy\/cô|giảng viên|học phí|tuyển sinh|đăng ký|liên hệ|quan trọng|lưu ý|chú ý|hạn chót|deadline)\b/gi, '<span class="format-keyword">$1</span>');
        formattedContent = formattedContent.replace(/(https?:\/\/[^\s<>]+)/g, '<a href="$1" target="_blank" class="format-link">$1</a>');
        formattedContent = formattedContent.replace(/(\b0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b)/g, '<a href="tel:$1" class="format-phone">📞 $1</a>');
        formattedContent = formattedContent.replace(/(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)/g, '<a href="mailto:$1" class="format-email">✉️ $1</a>');
        formattedContent = formattedContent.replace(/\b(BDU|Đại học Bình Dương)\b/g, '<span class="format-university">$1</span>');
        formattedContent = formattedContent.replace(/\n{3,}/g, '\n\n');
        formattedContent = formattedContent.replace(/\n(?![^<]*>)/g, '<br/>');
        formattedContent = formattedContent.replace(/(<\/div>)(<div class="format-)/g, '$1<br/>$2');
        
        return <div 
            className="formatted-content" 
            dangerouslySetInnerHTML={{ __html: formattedContent }}
        />;
    };

    const getConfidenceColor = (confidence) => {
        if (confidence >= 0.8) return '#4CAF50';
        if (confidence >= 0.6) return '#FF9800';
        return '#F44336';
    };

    const clearChat = () => {
        const newSessionId = 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
        setSessionId(newSessionId);
        
        const welcomeMessage = {
            type: 'bot',
            content: `Cuộc trò chuyện mới đã bắt đầu! 🎉

Tôi sẵn sàng hỗ trợ bạn với:
• 📚 Thông tin tuyển sinh 
• 🎓 Các ngành đào tạo
• 💰 Học phí và hỗ trợ
• 🏢 Cơ sở vật chất

Hãy đặt câu hỏi để bắt đầu!`,
            timestamp: new Date(),
            confidence: 1.0
        };
        
        setMessages([welcomeMessage]);
        setExpandedSources(new Set());
    };

    const retryConnection = () => {
        setConnectionStatus('checking');
        setMessages([]);
        testConnection();
    };

    const createNewChat = () => {
        const newId = Math.max(...chatSessions.map(s => s.id)) + 1;
        const newSession = {
            id: newId,
            title: `Đoạn chat ${newId}`,
            active: false
        };
        setChatSessions(prev => [...prev, newSession]);
    };

    const switchChatSession = (id) => {
        setChatSessions(prev => 
            prev.map(session => ({
                ...session,
                active: session.id === id
            }))
        );
        clearChat();
    };

    return (
        <div className="modern-chatbot-container">
            {/* Sidebar */}
            <div className={`modern-sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
                <div className="sidebar-header">
                    <div className="logo-section">
                        <div className="logo-icon">
                            <img 
                                src="../assets/logo.png" 
                                alt="BDU Logo" 
                                className="logo-image"
                            />
                        </div>
                        <div className="logo-text">
                            <h3>BDU ChatBot</h3>
                            <span>Trợ lý AI thông minh</span>
                        </div>
                    </div>
                    <button 
                        className="toggle-sidebar-btn"
                        onClick={() => setSidebarOpen(!sidebarOpen)}
                    >
                        {sidebarOpen ? '‹' : '›'}
                    </button>
                </div>

                <div className="sidebar-actions">
                    <button className="new-chat-btn" onClick={createNewChat}>
                        <span className="btn-icon">✏️</span>
                        <span className="btn-text">Đoạn chat mới</span>
                    </button>
                    
                    <div className="search-container">
                        <span className="search-icon">🔍</span>
                        <input 
                            type="text" 
                            placeholder="Tìm kiếm đoạn chat..."
                            className="search-input"
                        />
                    </div>
                    
                    <button className="create-project-btn">
                        <span className="btn-icon">📂</span>
                        <span className="btn-text">Tạo dự án mới</span>
                    </button>
                </div>

                <div className="sidebar-content">
                    <div className="chat-list-header">
                        <h4>Đoạn chat</h4>
                    </div>
                    
                    <div className="chat-list">
                        {chatSessions.map(session => (
                            <div 
                                key={session.id}
                                className={`chat-item ${session.active ? 'active' : ''}`}
                                onClick={() => switchChatSession(session.id)}
                            >
                                <span className="chat-icon">💬</span>
                                <span className="chat-title">{session.title}</span>
                                <button className="chat-menu-btn">⋯</button>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="sidebar-footer">
                    <div className="connection-indicator">
                        <div className={`status-dot ${connectionStatus}`}></div>
                        <span className="status-text">
                            {connectionStatus === 'connected' ? 'Đã kết nối' : 
                             connectionStatus === 'checking' ? 'Đang kết nối...' : 'Lỗi kết nối'}
                        </span>
                    </div>
                    
                    {personalizationEnabled && user && (
                        <button 
                            className="personalization-btn"
                            onClick={() => setShowPersonalization(true)}
                        >
                            <span className="btn-icon">⚙️</span>
                            <span className="btn-text">Cài đặt</span>
                        </button>
                    )}
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="modern-main-area">
                {/* Header */}
                <div className="modern-header">
                    {!sidebarOpen && (
                        <button 
                            className="open-sidebar-btn"
                            onClick={() => setSidebarOpen(true)}
                        >
                            ☰
                        </button>
                    )}
                    
                    <div className="header-title">
                        <h2>Tôi có thể giúp gì cho bạn?</h2>
                        <p>Đặt câu hỏi về Đại học Bình Dương hoặc bất kỳ điều gì bạn muốn biết</p>
                    </div>

                    <div className="header-controls">
                        {speechSupported && (
                            <div className="voice-indicator">
                                <span className="voice-icon">🎤</span>
                                <span className="voice-text">Voice enabled</span>
                            </div>
                        )}
                        
                        <button className="clear-chat-btn" onClick={clearChat}>
                            <span className="btn-icon">🔄</span>
                            <span className="btn-text">Làm mới</span>
                        </button>
                    </div>
                </div>

                {/* Messages Area */}
                <div className="modern-messages-area">
                    {messages.length === 0 || (messages.length === 1 && messages[0].type === 'bot') ? (
                        <div className="welcome-section">
                            <div className="welcome-animation">
                                <div className="floating-icon">
                                    <img
                                        src="../assets/logo.png"
                                        alt="BDU Assistant"
                                        className="floating-logo"
                                    />
                                </div>
                                <div className="pulse-ring"></div>
                            </div>
                            
                            {messages.length > 0 && (
                                <div className="welcome-message">
                                    {formatMessage(messages[0].content)}
                                </div>
                            )}
                            
                            <div className="quick-actions">
                                <h3>Gợi ý câu hỏi:</h3>
                                <div className="quick-buttons">
                                    <button 
                                        className="quick-btn"
                                        onClick={() => setInputMessage('Thông tin tuyển sinh năm 2024')}
                                    >
                                        📚 Tuyển sinh 2024
                                    </button>
                                    <button 
                                        className="quick-btn"
                                        onClick={() => setInputMessage('Học phí các ngành năm 2024')}
                                    >
                                        💰 Học phí
                                    </button>
                                    <button 
                                        className="quick-btn"
                                        onClick={() => setInputMessage('Các ngành đào tạo tại BDU')}
                                    >
                                        🎓 Ngành học
                                    </button>
                                    <button 
                                        className="quick-btn"
                                        onClick={() => setInputMessage('Cơ sở vật chất và ký túc xá')}
                                    >
                                        🏢 Cơ sở vật chất
                                    </button>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="messages-list">
                            {messages.map((message, index) => (
                                <div
                                    key={index}
                                    className={`modern-message ${message.type} ${message.isError ? 'error' : ''} ${message.temporary ? 'temporary' : ''} ${message.messageType || ''}`}
                                >
                                    <div className="message-bubble">
                                        <div className="message-content">
                                            {formatMessage(message.content)}
                                        </div>

                                        {/* Sources and Reference Links */}
                                        {(message.sources && message.sources.length > 0) || (message.reference_links && message.reference_links.length > 0) ? (
                                            <div className="message-attachments">
                                                {message.sources && message.sources.length > 0 && (
                                                    <div className="sources-section">
                                                        <h4>📚 Nguồn tham khảo:</h4>
                                                        {message.sources.map((source, idx) => (
                                                            <div key={idx} className="source-item">
                                                                <div className="source-question">Q: {source.question}</div>
                                                                <div className="source-similarity">
                                                                    Độ tương đồng: {(source.similarity * 100).toFixed(1)}%
                                                                </div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                )}

                                                {message.reference_links && message.reference_links.length > 0 && (
                                                    <div className="reference-links-section">
                                                        <h4>🔗 Tài liệu liên quan:</h4>
                                                        <div className="reference-links">
                                                            {message.reference_links.map((link, linkIdx) => (
                                                                <a
                                                                    key={linkIdx}
                                                                    href={link.url}
                                                                    target="_blank"
                                                                    rel="noopener noreferrer"
                                                                    className="reference-link"
                                                                >
                                                                    📄 {link.stt || link.title || 'Tài liệu'}
                                                                </a>
                                                            ))}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        ) : null}

                                        {/* Message metadata */}
                                        {message.type === 'bot' && !message.isError && !message.temporary && (
                                            <div className="message-metadata">
                                                {message.confidence && (
                                                    <div className="confidence-badge">
                                                        Độ tin cậy: {(message.confidence * 100).toFixed(1)}%
                                                    </div>
                                                )}
                                                
                                                {message.response_time && (
                                                    <div className="response-time">
                                                        ⏱️ {message.response_time}ms
                                                    </div>
                                                )}
                                            </div>
                                        )}

                                        {/* Feedback buttons */}
                                        {message.type === 'bot' && !message.isError && !message.temporary && message.chat_id && (
                                            <div className="message-feedback">
                                                {!message.feedbackSent ? (
                                                    <div className="feedback-buttons">
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
                                                    </div>
                                                ) : (
                                                    <div className="feedback-sent">
                                                        {message.feedbackSent === 'like' ? '👍 Đã đánh giá' : '👎 Đã đánh giá'}
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>

                                    <div className="message-timestamp">
                                        {message.timestamp.toLocaleTimeString('vi-VN')}
                                    </div>
                                </div>
                            ))}
                            
                            {isTyping && (
                                <div className="modern-message bot typing">
                                    <div className="message-bubble">
                                        <div className="typing-animation">
                                            <div className="typing-dots">
                                                <span></span>
                                                <span></span>
                                                <span></span>
                                            </div>
                                            <span className="typing-text">AI đang suy nghĩ...</span>
                                        </div>
                                    </div>
                                </div>
                            )}
                            
                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                {/* Input Section */}
                <div className="modern-input-section">
                    {/* Recording Status */}
                    {(isRecording || isProcessingSpeech) && (
                        <div className="recording-status">
                            {isRecording && (
                                <div className="recording-indicator">
                                    <div className="recording-animation">
                                        <div className="pulse-dot"></div>
                                    </div>
                                    <span className="recording-text">
                                        🔴 Đang ghi âm... {formatRecordingTime(recordingTime)}
                                    </span>
                                </div>
                            )}
                            {isProcessingSpeech && (
                                <div className="processing-indicator">
                                    <div className="processing-spinner"></div>
                                    <span className="processing-text">🔄 Đang xử lý giọng nói...</span>
                                </div>
                            )}
                        </div>
                    )}

                    <div className="modern-input-container">
                        <div className="input-wrapper">
                            <textarea
                                ref={inputRef}
                                value={inputMessage}
                                onChange={(e) => setInputMessage(e.target.value)}
                                onKeyPress={handleKeyPress}
                                placeholder={connectionStatus === 'connected' 
                                    ? "Hỏi bất cứ điều gì về Đại học Bình Dương..."
                                    : "Đang kết nối đến server..."}
                                rows="1"
                                disabled={isLoading || connectionStatus !== 'connected' || isRecording}
                                maxLength={1000}
                                className="modern-input"
                            />
                            
                            <div className="input-actions">
                                {speechSupported && (
                                    <button 
                                        className={`voice-btn ${isRecording ? 'recording' : ''} ${isProcessingSpeech ? 'processing' : ''}`}
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
                                    className="send-btn"
                                    title="Gửi tin nhắn"
                                >
                                    {isLoading ? '⏳' : '↗️'}
                                </button>
                            </div>
                        </div>
                        
                        <div className="input-footer">
                            <div className="char-counter">
                                {inputMessage.length}/1000
                            </div>
                            
                            <div className="input-tools">
                                <button className="tool-btn" title="Công cụ khác">
                                    <span className="tool-icon">⚡</span>
                                    <span className="tool-text">Công cụ khác</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Personalization Modal */}
            {showPersonalization && (
                <PersonalizationSettings
                    user={user}
                    onClose={() => setShowPersonalization(false)}
                    onUpdateSuccess={(newData) => {
                        console.log('Personalization updated:', newData);
                    }}
                />
            )}
        </div>
    );
};

export default ChatBot;