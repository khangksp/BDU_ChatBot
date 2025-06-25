import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import PersonalizationSettings from './PersonalizationSettings'
import './ChatBot.css';

// Cấu hình API Base URL
import { API_BASE_URL } from '../config.js';

const ChatBot = () => {
    // ✅ 1. BASIC STATES FIRST
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
    
    // ✅ 2. PERSONALIZATION STATES (MUST BE BEFORE useEffect)
    const [showPersonalization, setShowPersonalization] = useState(false);
    const [user, setUser] = useState(null);
    const [personalizationEnabled, setPersonalizationEnabled] = useState(false);
    
    // ✅ 3. CHAT HISTORY STATES
    const [currentSessionId, setCurrentSessionId] = useState('');
    const [loadingSessions, setLoadingSessions] = useState(false);
    const [loadingMessages, setLoadingMessages] = useState(false);
    
    // ✅ STATE DECLARATIONS
    const [contextMenu, setContextMenu] = useState(null);
    const [showRenameModal, setShowRenameModal] = useState(false);
    const [renameSessionId, setRenameSessionId] = useState('');
    const [newSessionTitle, setNewSessionTitle] = useState('');

    // ✅ 4. UI STATES
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [chatSessions, setChatSessions] = useState([]);

    // ✅ 5. REFS
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);
    const audioChunks = useRef([]);
    const recordingInterval = useRef(null);

    // ✅ 6. MAIN SETUP useEffect (FIRST)
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

    // ✅ 7. CHAT SESSIONS useEffect (AFTER user is declared)
    useEffect(() => {
        if (user && personalizationEnabled) {
            console.log('Loading chat sessions for user:', user.faculty_code);
            loadChatSessions();
        }
    }, [user, personalizationEnabled]);

    useEffect(() => {
        // Close context menu when clicking outside
        const handleClickOutside = (event) => {
            if (contextMenu && !event.target.closest('.context-menu') && !event.target.closest('.chat-menu-btn')) {
                setContextMenu(null);
            }
        };

        document.addEventListener('click', handleClickOutside);
        return () => {
            document.removeEventListener('click', handleClickOutside);
        };
    }, [contextMenu]);

    // ✅ 8. CONNECTION STATUS useEffect
    useEffect(() => {
        if (connectionStatus === 'connected') {
            // Only set welcome message if no user or no sessions loaded
            if (!user || chatSessions.length === 0) {
                const newSessionId = 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
                setSessionId(newSessionId);
                
                const welcomeMessage = {
                    type: 'bot',
                    content: getPersonalizedWelcomeMessage(),
                    timestamp: new Date(),
                    confidence: 1.0
                };
                
                setMessages([welcomeMessage]);
            }
        }
    }, [connectionStatus, speechSupported, user]);

    // ✅ 9. SCROLL useEffect
    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // ✅ 10. HELPER FUNCTIONS

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

    // ✅ 11. SIMPLIFIED CHAT SESSION FUNCTIONS

    const showContextMenu = (e, sessionId) => {
        e.stopPropagation();
        e.preventDefault();
        
        const rect = e.target.getBoundingClientRect();
        setContextMenu({
            sessionId: sessionId,
            x: rect.left,
            y: rect.bottom + 5,
            show: true
        });
    };

    const hideContextMenu = () => {
        setContextMenu(null);
    };

    const handleRenameSession = (sessionId) => {
        const session = chatSessions.find(s => s.session_id === sessionId || s.id === sessionId);
        setRenameSessionId(sessionId);
        setNewSessionTitle(session?.title || '');
        setShowRenameModal(true);
        hideContextMenu();
    };

    const confirmRenameSession = async () => {
        if (!newSessionTitle.trim()) return;
        
        try {
            const response = await axios.patch(`/api/chat-sessions/${renameSessionId}/`, {
                title: newSessionTitle.trim()
            });
            
            if (response.data.success) {
                // Update local state
                setChatSessions(prev => 
                    prev.map(session => 
                        (session.session_id === renameSessionId || session.id === renameSessionId)
                            ? { ...session, title: newSessionTitle.trim() }
                            : session
                    )
                );
                
                console.log('Session renamed successfully');
            }
        } catch (error) {
            console.error('Error renaming session:', error);
            alert('Lỗi khi đổi tên đoạn chat. Vui lòng thử lại.');
        } finally {
            setShowRenameModal(false);
            setRenameSessionId('');
            setNewSessionTitle('');
        }
    };

    const handleDeleteSession = async (sessionId) => {
        const confirmed = window.confirm('Bạn có chắc muốn xóa đoạn chat này không?');
        if (!confirmed) return;
        
        try {
            const response = await axios.delete(`/api/chat-sessions/${sessionId}/`);
            
            if (response.data.success) {
                // Remove from local state
                setChatSessions(prev => 
                    prev.filter(session => 
                        session.session_id !== sessionId && session.id !== sessionId
                    )
                );
                
                // If deleted session was active, create new one
                const deletedSession = chatSessions.find(s => 
                    (s.session_id === sessionId || s.id === sessionId) && s.active
                );
                
                if (deletedSession) {
                    await createNewChatSession();
                }
                
                console.log('Session deleted successfully');
            }
        } catch (error) {
            console.error('Error deleting session:', error);
            alert('Lỗi khi xóa đoạn chat. Vui lòng thử lại.');
        } finally {
            hideContextMenu();
        }
    };

    const loadChatSessions = async () => {
        if (!user) {
            console.log('No user found, skipping load sessions');
            return;
        }
        
        setLoadingSessions(true);
        try {
            console.log('Calling /api/chat-sessions/ for user:', user.faculty_code);
            const response = await axios.get('/api/chat-sessions/');
            
            if (response.data.success) {
                const sessions = response.data.sessions;
                console.log('Loaded sessions:', sessions);
                
                if (sessions.length > 0) {
                    // Set session đầu tiên là active
                    const updatedSessions = sessions.map((session, index) => ({
                        ...session,
                        active: index === 0,
                        id: session.session_id // Để tương thích với UI hiện tại
                    }));
                    
                    setChatSessions(updatedSessions);
                    
                    // Load messages của session đầu tiên
                    if (updatedSessions[0]) {
                        setCurrentSessionId(updatedSessions[0].session_id);
                        loadSessionMessages(updatedSessions[0].session_id);
                    }
                } else {
                    // Tạo session mới nếu chưa có
                    console.log('No sessions found, creating new one');
                    createNewChatSession();
                }
            }
        } catch (error) {
            console.error('Error loading chat sessions:', error);
            // Fallback: load welcome message without sessions
            loadWelcomeMessage();
        } finally {
            setLoadingSessions(false);
        }
    };

    const loadSessionMessages = async (sessionId) => {
        setLoadingMessages(true);
        try {
            console.log('Loading messages for session:', sessionId);
            const response = await axios.get(`/api/chat-sessions/${sessionId}/`);
            
            if (response.data.success) {
                const loadedMessages = response.data.messages.map(msg => ({
                    ...msg,
                    timestamp: new Date(msg.timestamp),
                    sources: msg.sources || [],
                    reference_links: msg.reference_links || []
                }));
                
                console.log('Loaded messages:', loadedMessages.length);
                setMessages(loadedMessages);
                setSessionId(sessionId);
                setCurrentSessionId(sessionId);
            }
        } catch (error) {
            console.error('Error loading session messages:', error);
            // Fallback: tạo session mới
            createNewChatSession();
        } finally {
            setLoadingMessages(false);
        }
    };

    const createNewChatSession = async () => {
        try {
            if (!user) {
                // Fallback cho user chưa login
                const fallbackSessionId = 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
                setSessionId(fallbackSessionId);
                setCurrentSessionId(fallbackSessionId);
                loadWelcomeMessage();
                return fallbackSessionId;
            }

            console.log('Creating new session for user:', user.faculty_code);
            const response = await axios.post('/api/chat-sessions/', {
                title: `Chat mới - ${new Date().toLocaleTimeString('vi-VN')}`
            });
            
            if (response.data.success) {
                const newSession = {
                    id: response.data.session_id,
                    session_id: response.data.session_id,
                    title: response.data.title,
                    active: true,
                    message_count: 0,
                    preview: '',
                    last_message_time: new Date().toISOString()
                };
                
                // Update sessions list
                setChatSessions(prev => [
                    newSession,
                    ...prev.map(s => ({ ...s, active: false }))
                ]);
                
                // Set current session
                setSessionId(newSession.session_id);
                setCurrentSessionId(newSession.session_id);
                
                // Clear messages và hiển thị welcome message
                loadWelcomeMessage();
                
                return newSession.session_id;
            }
        } catch (error) {
            console.error('Error creating new session:', error);
            // Fallback: tạo session ID local
            const fallbackSessionId = 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
            setSessionId(fallbackSessionId);
            setCurrentSessionId(fallbackSessionId);
            loadWelcomeMessage();
            return fallbackSessionId;
        }
    };

    const loadWelcomeMessage = () => {
        const welcomeMessage = {
            type: 'bot',
            content: getPersonalizedWelcomeMessage(),
            timestamp: new Date(),
            confidence: 1.0
        };
        setMessages([welcomeMessage]);
    };

    const getPersonalizedWelcomeMessage = () => {
        if (user) {
            const name = user.full_name?.split(' ').pop() || user.faculty_code;
            return `Xin chào ${user.position_name || 'giảng viên'} ${name}! 

Tôi là ChatBDU, trợ lý AI của Đại học Bình Dương. Tôi có thể hỗ trợ ${user.position_name?.toLowerCase() || 'bạn'} về:

• 📚 Thông tin đào tạo và ngành học
• 🎓 Quy định và thủ tục  
• 💰 Học phí và chính sách
• 🏢 Cơ sở vật chất
• 📞 Thông tin liên hệ

${speechSupported ? '🎤 Bạn có thể gõ hoặc nói để đặt câu hỏi!' : 'Hãy đặt câu hỏi để bắt đầu!'}`;
        }
        
        // Fallback cho user chưa login
        return `Xin chào! Tôi là trợ lý AI của Đại học Bình Dương. Tôi có thể giúp bạn:

• 📚 Thông tin tuyển sinh và ngành học
• 💰 Học phí và chính sách hỗ trợ  
• 🎓 Đời sống sinh viên
• 🏢 Cơ sở vật chất và tiện ích
• 📞 Thông tin liên hệ

${speechSupported ? '🎤 Bạn có thể gõ hoặc nói để đặt câu hỏi!' : 'Hãy đặt câu hỏi để bắt đầu!'}`;
    };

    const switchChatSession = async (sessionId) => {
        // Update active state
        setChatSessions(prev => 
            prev.map(session => ({
                ...session,
                active: session.session_id === sessionId || session.id === sessionId
            }))
        );
        
        // Load messages của session đó
        await loadSessionMessages(sessionId);
        
        // Scroll to bottom
        setTimeout(() => scrollToBottom(), 100);
    };

    const createNewChat = async () => {
        await createNewChatSession();
    };

    const clearChat = async () => {
        await createNewChatSession();
    };

    const renderChatSessions = () => {
        if (loadingSessions) {
            return (
                <div className="loading-sessions">
                    <div className="spinner-small"></div>
                    <span>Đang tải...</span>
                </div>
            );
        }

        if (chatSessions.length === 0) {
            return (
                <div className="empty-sessions">
                    <p>Chưa có đoạn chat nào</p>
                    <button onClick={createNewChat}>Tạo chat đầu tiên</button>
                </div>
            );
        }

        return (
            <>
                {chatSessions.map(session => (
                    <div 
                        key={session.session_id || session.id}
                        className={`chat-item ${session.active ? 'active' : ''}`}
                        onClick={() => switchChatSession(session.session_id || session.id)}
                    >
                        <span className="chat-icon">💬</span>
                        <div className="chat-info">
                            <span className="chat-title">{session.title}</span>
                            {session.preview && (
                                <span className="chat-preview">{session.preview}</span>
                            )}
                            <span className="chat-time">
                                {session.last_message_time ? 
                                    new Date(session.last_message_time).toLocaleDateString('vi-VN') :
                                    'Mới tạo'
                                }
                            </span>
                        </div>
                        <div className="chat-menu">
                            <button 
                                className="chat-menu-btn"
                                onClick={(e) => showContextMenu(e, session.session_id || session.id)}
                                title="Tùy chọn"
                            >
                                ⋯
                            </button>
                        </div>
                    </div>
                ))}
                
                {/* ✅ SIMPLIFIED Context Menu - Only Rename and Delete */}
                {contextMenu && contextMenu.show && (
                    <div 
                        className="context-menu"
                        style={{
                            position: 'fixed',
                            left: contextMenu.x,
                            top: contextMenu.y,
                            zIndex: 1000
                        }}
                    >
                        <div className="context-menu-item" onClick={() => handleRenameSession(contextMenu.sessionId)}>
                            <span className="menu-icon">✏️</span>
                            <span className="menu-text">Đổi tên</span>
                        </div>
                        <div className="context-menu-item danger" onClick={() => handleDeleteSession(contextMenu.sessionId)}>
                            <span className="menu-icon">🗑️</span>
                            <span className="menu-text">Xóa</span>
                        </div>
                    </div>
                )}
            </>
        );
    };

    // ✅ 12. SPEECH FUNCTIONS (unchanged)

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

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    // ✅ 13. CHAT FUNCTIONS

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

    const retryConnection = () => {
        setConnectionStatus('checking');
        setMessages([]);
        testConnection();
    };

    // ✅ 14. RENDER JSX

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
                        <h4>Đoạn chat ({chatSessions.length})</h4>
                        {loadingMessages && (
                            <span className="loading-indicator">⏳</span>
                        )}
                    </div>
                    
                    <div className="chat-list">
                        {renderChatSessions()}
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
                <div className={`modern-messages-area ${loadingMessages ? 'loading' : ''}`}>
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

            {/* ✅ SIMPLIFIED Rename Modal */}
            {showRenameModal && (
                <div className="modal-overlay" onClick={() => setShowRenameModal(false)}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3>✏️ Đổi tên đoạn chat</h3>
                            <button 
                                className="modal-close"
                                onClick={() => setShowRenameModal(false)}
                            >
                                ✕
                            </button>
                        </div>
                        <div className="modal-body">
                            <input
                                type="text"
                                value={newSessionTitle}
                                onChange={(e) => setNewSessionTitle(e.target.value)}
                                placeholder="Nhập tên mới cho đoạn chat..."
                                className="rename-input"
                                maxLength={100}
                                autoFocus
                                onKeyPress={(e) => {
                                    if (e.key === 'Enter') {
                                        confirmRenameSession();
                                    }
                                    if (e.key === 'Escape') {
                                        setShowRenameModal(false);
                                    }
                                }}
                            />
                            <div className="input-hint">
                                💡 Tên mới sẽ giúp bạn dễ dàng tìm lại đoạn chat này
                            </div>
                        </div>
                        <div className="modal-footer">
                            <button 
                                className="btn-cancel"
                                onClick={() => setShowRenameModal(false)}
                            >
                                Hủy
                            </button>
                            <button 
                                className="btn-confirm"
                                onClick={confirmRenameSession}
                                disabled={!newSessionTitle.trim()}
                            >
                                💾 Lưu
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ChatBot;