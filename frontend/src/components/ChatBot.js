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
    
    // ✅ TTS states
    const [ttsSupported, setTtsSupported] = useState(false);
    const [isPlayingAudio, setIsPlayingAudio] = useState(false);
    const [currentAudio, setCurrentAudio] = useState(null);
    const [autoPlayEnabled, setAutoPlayEnabled] = useState(true);
    const [voiceModeEnabled, setVoiceModeEnabled] = useState(false);
    const [lastUserInputMethod, setLastUserInputMethod] = useState('text');
    const [forceVoiceMode, setForceVoiceMode] = useState(false);
    
    // 🚀 NEW: File Upload và Document Context states
    const [selectedFile, setSelectedFile] = useState(null);
    const [isUploadingFile, setIsUploadingFile] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [currentDocument, setCurrentDocument] = useState(null);
    const [documentContext, setDocumentContext] = useState(null);
    const [isProcessingDocument, setIsProcessingDocument] = useState(false);
    const [documentPreview, setDocumentPreview] = useState(null);
    const [showDocumentModal, setShowDocumentModal] = useState(false);
    const [documentProcessingStatus, setDocumentProcessingStatus] = useState('');
    const [autoDownloadEnabled, setAutoDownloadEnabled] = useState(true);
    const [pendingDocumentUrls, setPendingDocumentUrls] = useState([]);
    
    // Reference sources state
    const [expandedSources, setExpandedSources] = useState(new Set());
    
    // ✅ 2. PERSONALIZATION STATES
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
    const audioRef = useRef(null);
    const fileInputRef = useRef(null); // 🚀 NEW: File input ref

    // ✅ 6. MAIN SETUP useEffect
    useEffect(() => {
        axios.defaults.baseURL = API_BASE_URL;
        axios.defaults.timeout = 30000;
        
        testConnection();
        checkSpeechSupport();
        checkTtsSupport();
        checkUserAuth(); 
        checkDocumentSupportStatus(); // 🚀 NEW: Check document support
        
        const handleResize = () => {
            if (window.innerWidth <= 768) {
                setSidebarOpen(false);
            } else {
                setSidebarOpen(true);
            }
        };
        
        handleResize();
        window.addEventListener('resize', handleResize);
        
        return () => {
            if (recordingInterval.current) {
                clearInterval(recordingInterval.current);
            }
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
            if (currentAudio) {
                currentAudio.pause();
                currentAudio.src = '';
            }
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    // ✅ 7. CHAT SESSIONS useEffect
    useEffect(() => {
        if (user && personalizationEnabled) {
            console.log('Loading chat sessions for user:', user.faculty_code);
            loadChatSessions();
        }
    }, [user, personalizationEnabled]);

    useEffect(() => {
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
    }, [connectionStatus, speechSupported, ttsSupported, user]);

    // ✅ 9. SCROLL useEffect
    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // 🚀 NEW: Document context useEffect
    useEffect(() => {
        if (currentDocument && !isProcessingDocument) {
            showTemporaryMessage(
                `📄 Đang sử dụng tài liệu: ${currentDocument.filename}`,
                'document-context'
            );
        }
    }, [currentDocument]);

    // 🚀 NEW: Auto-download documents from reference links
    useEffect(() => {
        if (autoDownloadEnabled && pendingDocumentUrls.length > 0) {
            processPendingDocuments();
        }
    }, [autoDownloadEnabled, pendingDocumentUrls]);

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

    const checkTtsSupport = async () => {
        try {
            const response = await axios.get('/api/speech-status/');
            const ttsServiceAvailable = response.data.tts_service?.available || false;
            
            console.log('TTS service status:', response.data.tts_service);
            setTtsSupported(ttsServiceAvailable);
            
            if (ttsServiceAvailable) {
                console.log('✅ TTS service available');
                if (speechSupported) {
                    setVoiceModeEnabled(true);
                    console.log('🎤🔊 Auto-enabled voice mode (STT + TTS available)');
                }
            } else {
                console.log('⚠️ Backend TTS service not available');
                setVoiceModeEnabled(false);
            }
        } catch (error) {
            console.error('Error checking TTS support:', error);
            setTtsSupported(false);
            setVoiceModeEnabled(false);
        }
    };

    // 🚀 NEW: Check document support status
    const checkDocumentSupportStatus = async () => {
        try {
            const response = await axios.get('/api/document-support-status/');
            const documentSupported = response.data.document_context_support || false;
            
            console.log('📄 Document support status:', response.data);
            
            if (documentSupported) {
                console.log('✅ Document context support available');
                showTemporaryMessage('📄 Hỗ trợ tài liệu PDF/DOCX có sẵn', 'document-support');
            } else {
                console.log('⚠️ Document context support not available');
            }
        } catch (error) {
            console.error('Error checking document support:', error);
        }
    };

    // 🚀 NEW: File Upload Functions
    const handleFileSelect = (event) => {
        const file = event.target.files[0];
        if (!file) return;

        // Giữ lại logic kiểm tra file type và size
        const allowedTypes = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword'
        ];
        if (!allowedTypes.includes(file.type)) {
            alert('❌ Chỉ hỗ trợ file PDF và DOCX. Vui lòng chọn file khác.');
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            alert('❌ File quá lớn (tối đa 10MB). Vui lòng chọn file nhỏ hơn.');
            return;
        }

        // ✅ THAY ĐỔI CHÍNH: Chỉ cập nhật state, không upload ngay
        setSelectedFile(file);
        showTemporaryMessage(`📄 Đã chọn file: ${file.name}. Gửi tin nhắn để bắt đầu phân tích.`, 'document-selected');
    };

    // 🚀 NEW: Auto-download document from URL
    const downloadAndProcessDocument = async (url, filename = null) => {
        setIsProcessingDocument(true);
        setDocumentProcessingStatus('Đang tải từ link...');

        try {
            const response = await axios.post('/api/download-and-process-document/', {
                url: url,
                filename: filename,
                session_id: sessionId
            }, {
                timeout: 120000 // 2 minutes timeout
            });

            if (response.data.success) {
                const documentData = {
                    filename: response.data.filename,
                    file_size: response.data.file_size,
                    file_type: response.data.file_type,
                    upload_time: new Date(),
                    document_id: response.data.document_id,
                    text_content: response.data.text_content,
                    page_count: response.data.page_count,
                    processing_time: response.data.processing_time,
                    source_url: url
                };

                setCurrentDocument(documentData);
                setDocumentContext(response.data.text_content);
                setDocumentPreview(response.data.preview || response.data.text_content?.substring(0, 500));

                // Show success message
                showTemporaryMessage(
                    `✅ Đã tải và xử lý "${response.data.filename}" từ link`,
                    'document-download-success'
                );

                // Add system message about document
                const documentMessage = {
                    type: 'system',
                    content: `📄 Đã tải tài liệu từ link: **${response.data.filename}**\n\n` +
                            `📊 Thông tin: ${response.data.page_count} trang\n\n` +
                            `💡 Bạn có thể hỏi về nội dung tài liệu này ngay bây giờ!`,
                    timestamp: new Date(),
                    temporary: false,
                    document_info: documentData
                };

                setMessages(prev => [...prev, documentMessage]);
                setDocumentProcessingStatus('Hoàn thành');

                return true;

            } else {
                throw new Error(response.data.error || 'Không thể tải tài liệu từ link');
            }

        } catch (error) {
            console.error('❌ Error downloading document:', error);
            showTemporaryMessage(`❌ Không thể tải tài liệu từ link: ${error.message}`, 'document-download-error');
            setDocumentProcessingStatus('Lỗi');
            return false;
        } finally {
            setIsProcessingDocument(false);
        }
    };

    // 🚀 NEW: Process pending documents from reference links
    const processPendingDocuments = async () => {
        if (pendingDocumentUrls.length === 0) return;

        for (const urlData of pendingDocumentUrls) {
            const success = await downloadAndProcessDocument(urlData.url, urlData.title);
            if (success) {
                // Remove processed URL from pending list
                setPendingDocumentUrls(prev => prev.filter(item => item.url !== urlData.url));
                break; // Only process one document at a time
            }
        }
    };

    // 🚀 NEW: Clear document context
    const clearDocumentContext = () => {
        setCurrentDocument(null);
        setDocumentContext(null);
        setDocumentPreview(null);
        setDocumentProcessingStatus('');
        showTemporaryMessage('🗑️ Đã xóa tài liệu khỏi ngữ cảnh', 'document-cleared');
    };

    // 🚀 NEW: Toggle document modal
    const toggleDocumentModal = () => {
        setShowDocumentModal(!showDocumentModal);
    };

    // 🚀 NEW: Toggle auto-download
    const toggleAutoDownload = () => {
        setAutoDownloadEnabled(!autoDownloadEnabled);
        if (!autoDownloadEnabled) {
            showTemporaryMessage('⚡ Tự động tải tài liệu: BẬT', 'auto-download-on');
        } else {
            showTemporaryMessage('⏸️ Tự động tải tài liệu: TẮT', 'auto-download-off');
        }
    };

    // ✅ EXISTING FUNCTIONS (keeping all existing functions)

    const playAudioFromBase64 = async (base64Audio) => {
        if (!base64Audio || !autoPlayEnabled) {
            return;
        }

        try {
            setIsPlayingAudio(true);
            
            if (currentAudio) {
                currentAudio.pause();
                currentAudio.src = '';
            }

            const audioBlob = base64ToBlob(base64Audio, 'audio/mp3');
            const audioUrl = URL.createObjectURL(audioBlob);
            
            const audio = new Audio(audioUrl);
            setCurrentAudio(audio);
            
            audio.onended = () => {
                setIsPlayingAudio(false);
                URL.revokeObjectURL(audioUrl);
                setCurrentAudio(null);
            };
            
            audio.onerror = (error) => {
                console.error('Audio playback error:', error);
                setIsPlayingAudio(false);
                showTemporaryMessage('❌ Lỗi phát âm thanh', 'audio-error');
                URL.revokeObjectURL(audioUrl);
                setCurrentAudio(null);
            };
            
            await audio.play();
            console.log('🔊 Playing TTS audio');
            
        } catch (error) {
            console.error('Error playing audio:', error);
            setIsPlayingAudio(false);
            showTemporaryMessage('❌ Không thể phát âm thanh', 'audio-error');
        }
    };

    const base64ToBlob = (base64, mimeType) => {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mimeType });
    };

    const stopCurrentAudio = () => {
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.src = '';
            setCurrentAudio(null);
            setIsPlayingAudio(false);
        }
    };

    const toggleAutoPlay = () => {
        setAutoPlayEnabled(!autoPlayEnabled);
        if (!autoPlayEnabled) {
            showTemporaryMessage('🔊 Tự động phát âm thanh: BẬT', 'audio-info');
        } else {
            showTemporaryMessage('🔇 Tự động phát âm thanh: TẮT', 'audio-info');
            stopCurrentAudio();
        }
    };

    const toggleVoiceMode = () => {
        if (!ttsSupported) {
            showTemporaryMessage('❌ Chế độ giọng nói không khả dụng', 'audio-error');
            return;
        }
        
        setVoiceModeEnabled(!voiceModeEnabled);
        if (!voiceModeEnabled) {
            showTemporaryMessage('🎤🔊 Chế độ giọng nói: BẬT', 'voice-mode-on');
        } else {
            showTemporaryMessage('📝 Chế độ văn bản: BẬT', 'voice-mode-off');
            stopCurrentAudio();
        }
    };

    const toggleSidebar = () => {
        setSidebarOpen(!sidebarOpen);
    };

    const handleLogoClick = () => {
        setSidebarOpen(!sidebarOpen);
    };

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
                setChatSessions(prev => 
                    prev.filter(session => 
                        session.session_id !== sessionId && session.id !== sessionId
                    )
                );
                
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
                    const updatedSessions = sessions.map((session, index) => ({
                        ...session,
                        active: index === 0,
                        id: session.session_id
                    }));
                    
                    setChatSessions(updatedSessions);
                    
                    if (updatedSessions[0]) {
                        setCurrentSessionId(updatedSessions[0].session_id);
                        loadSessionMessages(updatedSessions[0].session_id);
                    }
                } else {
                    console.log('No sessions found, creating new one');
                    createNewChatSession();
                }
            }
        } catch (error) {
            console.error('Error loading chat sessions:', error);
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
            createNewChatSession();
        } finally {
            setLoadingMessages(false);
        }
    };

    const createNewChatSession = async () => {
        try {
            if (!user) {
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
                
                setChatSessions(prev => [
                    newSession,
                    ...prev.map(s => ({ ...s, active: false }))
                ]);
                
                setSessionId(newSession.session_id);
                setCurrentSessionId(newSession.session_id);
                
                // Clear document context when creating new session
                clearDocumentContext();
                
                loadWelcomeMessage();
                
                return newSession.session_id;
            }
        } catch (error) {
            console.error('Error creating new session:', error);
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
            
            const hasUserMemoryPrompt = user.chatbot_preferences?.user_memory_prompt?.trim();
            const memoryStatus = hasUserMemoryPrompt ? 
                "🧠 Tôi đã ghi nhớ những chỉ dẫn riêng mà bạn đã thiết lập!" : 
                "💡 Bạn có thể thiết lập 'Ghi nhớ và chỉ dẫn' trong cài đặt để tôi phục vụ bạn tốt hơn!";
            
            const voiceStatus = ttsSupported ? 
                (speechSupported ? 
                    "🎤🔊 Bạn có thể gõ, nói hoặc nghe phản hồi bằng giọng nói!" :
                    "🔊 Bạn có thể nghe phản hồi bằng giọng nói!") :
                (speechSupported ? 
                    "🎤 Bạn có thể gõ hoặc nói để đặt câu hỏi!" :
                    "Hãy đặt câu hỏi để bắt đầu!");
            
            return `Xin chào ${user.position_name || 'giảng viên'} ${name}! 

Tôi là ChatBDU, trợ lý AI của Đại học Bình Dương. ${memoryStatus}

Tôi có thể hỗ trợ ${user.position_name?.toLowerCase() || 'bạn'} về:

• 📚 Thông tin đào tạo và ngành học
• 🎓 Quy định và thủ tục  
• 💰 Học phí và chính sách
• 🏢 Cơ sở vật chất
• 📞 Thông tin liên hệ
• 📄 Phân tích tài liệu PDF/DOCX

${voiceStatus}

💡 **Mới**: Bạn có thể tải lên tài liệu PDF/DOCX để tôi phân tích và trả lời câu hỏi về nội dung!`;
        }
        
        const voiceStatus = ttsSupported ? 
            (speechSupported ? 
                "🎤🔊 Bạn có thể gõ, nói hoặc nghe phản hồi bằng giọng nói!" :
                "🔊 Bạn có thể nghe phản hồi bằng giọng nói!") :
            (speechSupported ? 
                "🎤 Bạn có thể gõ hoặc nói để đặt câu hỏi!" :
                "Hãy đặt câu hỏi để bắt đầu!");
        
        return `Xin chào! Tôi là trợ lý AI của Đại học Bình Dương. Tôi có thể giúp bạn:

• 📚 Thông tin tuyển sinh và ngành học
• 💰 Học phí và chính sách hỗ trợ  
• 🎓 Đời sống sinh viên
• 🏢 Cơ sở vật chất và tiện ích
• 📞 Thông tin liên hệ
• 📄 Phân tích tài liệu PDF/DOCX

${voiceStatus}

💡 **Mới**: Bạn có thể tải lên tài liệu PDF/DOCX để tôi phân tích và trả lời câu hỏi về nội dung!`;
    };

    const switchChatSession = async (sessionId) => {
        setChatSessions(prev => 
            prev.map(session => ({
                ...session,
                active: session.session_id === sessionId || session.id === sessionId
            }))
        );
        
        await loadSessionMessages(sessionId);
        
        // Clear document context when switching sessions
        clearDocumentContext();
        
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

    // ✅ SPEECH FUNCTIONS (keeping existing)

    const startRecording = async () => {
        try {
            setLastUserInputMethod('voice');
            setForceVoiceMode(true);
            
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
                    setForceVoiceMode(false);
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
            setForceVoiceMode(false);
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
                
                if (ttsSupported && voiceModeEnabled) {
                    showTemporaryMessage('🎤🔊 Sẵn sàng trả lời bằng giọng nói', 'voice-mode-ready');
                }
            } else {
                console.error('❌ Speech failed - Success:', response.data.success);
                console.error('❌ Speech failed - Text:', response.data.text);
                console.error('❌ Speech failed - Error:', response.data.error);
                
                const errorMsg = response.data.error || 'Không nhận diện được giọng nói';
                showTemporaryMessage(`❌ ${errorMsg}`, 'speech-error');
                setForceVoiceMode(false);
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
            setForceVoiceMode(false);
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

    // ✅ CHAT FUNCTIONS (Enhanced with document context)

    const sendMessage = async () => {
        // ✅ Kiểm tra cơ bản
        const messageContent = inputMessage.trim();
        if ((!messageContent && !selectedFile) || isLoading || connectionStatus !== 'connected') {
            if (!selectedFile) return; // Chỉ cho phép gửi file mà không cần tin nhắn
        }

        // ✅ Tạo một tin nhắn user tạm thời để hiển thị ngay lập tức
        const userMessage = {
            type: 'user',
            // Hiển thị tên file nếu không có tin nhắn text
            content: messageContent || `(Đã gửi file: ${selectedFile.name})`,
            timestamp: new Date()
        };
        setMessages(prev => [...prev, userMessage]);

        setIsLoading(true);
        setIsTyping(true);

        const messageToSend = messageContent;
        setInputMessage('');
        const fileToSend = selectedFile;
        setSelectedFile(null); // Xóa file đã chọn sau khi chuẩn bị gửi

        // ... (logic xác định requestMode giữ nguyên) ...
        let requestMode = 'text';
        if (ttsSupported && (forceVoiceMode || voiceModeEnabled)) {
            requestMode = 'voice';
        }

        try {
            let response;
            // 🚀 LOGIC QUAN TRỌNG NHẤT
            if (fileToSend) {
                // Trường hợp 1: Có file đính kèm -> Sử dụng FormData
                console.log(`📤 Sending message WITH DOCUMENT to /api/chat/`);
                const formData = new FormData();
                formData.append('message', messageToSend);
                formData.append('document', fileToSend); // Key là 'document'
                formData.append('session_id', sessionId);
                formData.append('mode', requestMode);

                response = await axios.post('/api/chat/', formData, {
                    headers: {
                        // Trình duyệt sẽ tự set 'Content-Type' là 'multipart/form-data'
                    },
                    timeout: 120000 // Tăng timeout cho việc xử lý file
                });
            } else {
                // Trường hợp 2: Chỉ có text -> Gửi JSON như cũ
                console.log(`📤 Sending TEXT-ONLY message to /api/chat/`);
                const requestData = {
                    message: messageToSend,
                    session_id: sessionId,
                    mode: requestMode
                };
                response = await axios.post('/api/chat/', requestData, {
                    headers: { 'Content-Type': 'application/json' },
                    timeout: 30000
                });
            }

            // ... (Toàn bộ phần xử lý response phía dưới giữ nguyên y hệt) ...
            const finalReferenceLinks = response.data.reference_links || [];

            if (autoDownloadEnabled && finalReferenceLinks.length > 0) {
                const documentLinks = finalReferenceLinks.filter(link => 
                    link.url && (link.url.includes('.pdf') || link.url.includes('.docx'))
                );
                
                if (documentLinks.length > 0) {
                    setPendingDocumentUrls(documentLinks);
                }
            }

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
                    reference_links: finalReferenceLinks,
                    user_memory_applied: response.data.personalization?.user_memory_info?.memory_applied || false,
                    external_api_used: response.data.external_api?.external_api_used || false,
                    audio_content: response.data.audio_content,
                    mode: response.data.mode,
                    tts_info: response.data.tts_info,
                    document_context_used: response.data.document_context_used || false,
                    document_enhanced: response.data.document_enhanced || false
                };
                
                setMessages(prev => [...prev, botMessage]);
                setIsTyping(false);
                
                if (botMessage.audio_content && autoPlayEnabled) {
                    playAudioFromBase64(botMessage.audio_content);
                }

            }, 1000);

        } catch (error) {
            console.error('Error sending message:', error);
            setTimeout(() => {
                const errorMessage = {
                    type: 'bot',
                    content: 'Xin lỗi, đã có lỗi xảy ra khi gửi tin nhắn.',
                    timestamp: new Date(),
                    isError: true
                };
                setMessages(prev => [...prev, errorMessage]);
                setIsTyping(false);
            }, 1000);
        } finally {
            setIsLoading(false);
            setLastUserInputMethod('text');
            setForceVoiceMode(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            setLastUserInputMethod('text');
            sendMessage();
        }
    };

    const handleInputChange = (e) => {
        setInputMessage(e.target.value);
        if (!forceVoiceMode) {
            setLastUserInputMethod('text');
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

    // ✅ RENDER JSX

    return (
        <div className="modern-chatbot-container">
            {/* Sidebar */}
            <div className={`modern-sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
                <div className="sidebar-header">
                    <div className="logo-section" onClick={handleLogoClick}>
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
                    
                    {(speechSupported || ttsSupported) && (
                        <div className="voice-features-status">
                            {speechSupported && (
                                <span className="feature-badge stt" title="Speech-to-Text khả dụng">
                                    🎤
                                </span>
                            )}
                            {ttsSupported && (
                                <span className="feature-badge tts" title="Text-to-Speech khả dụng">
                                    🔊
                                </span>
                            )}
                            {speechSupported && ttsSupported && (
                                <span className="feature-badge voice-chat" title="Trò chuyện bằng giọng nói hoàn chỉnh">
                                    💬
                                </span>
                            )}
                        </div>
                    )}
                    
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

                {/* 🚀 NEW: Document Context Panel */}
                {currentDocument && (
                    <div className="document-context-panel">
                        <div className="document-info">
                            <div className="document-header">
                                <span className="document-icon">📄</span>
                                <div className="document-details">
                                    <span className="document-name">{currentDocument.filename}</span>
                                    <span className="document-meta">
                                        {currentDocument.page_count} trang • {(currentDocument.file_size / 1024).toFixed(1)}KB
                                    </span>
                                </div>
                                <div className="document-actions">
                                    <button 
                                        className="doc-action-btn"
                                        onClick={toggleDocumentModal}
                                        title="Xem chi tiết"
                                    >
                                        👁️
                                    </button>
                                    <button 
                                        className="doc-action-btn danger"
                                        onClick={clearDocumentContext}
                                        title="Xóa khỏi ngữ cảnh"
                                    >
                                        🗑️
                                    </button>
                                </div>
                            </div>
                            <div className="document-status">
                                <span className="status-indicator active">
                                    ✅ Đang sử dụng trong ngữ cảnh
                                </span>
                            </div>
                        </div>
                    </div>
                )}

                {/* 🚀 NEW: File Upload and Document Controls */}
                <div className="document-control-panel">
                    <div className="upload-section">
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileSelect}
                            accept=".pdf,.docx,.doc"
                            style={{ display: 'none' }}
                        />
                        
                        <button 
                            className="upload-btn"
                            onClick={() => fileInputRef.current?.click()}
                            disabled={isUploadingFile || isProcessingDocument}
                            title="Tải lên tài liệu PDF/DOCX"
                        >
                            {isUploadingFile ? (
                                <>
                                    <span className="btn-icon">⏳</span>
                                    <span className="btn-text">Đang tải... {uploadProgress}%</span>
                                </>
                            ) : (
                                <>
                                    <span className="btn-icon">📁</span>
                                    <span className="btn-text">Tải lên tài liệu</span>
                                </>
                            )}
                        </button>

                        {(speechSupported || ttsSupported) && (
                            <div className="document-voice-controls">
                                <button 
                                    className={`doc-control-btn ${autoDownloadEnabled ? 'active' : ''}`}
                                    onClick={toggleAutoDownload}
                                    title={autoDownloadEnabled ? 'Tắt tự động tải' : 'Bật tự động tải'}
                                >
                                    {autoDownloadEnabled ? '⚡' : '⏸️'} 
                                    {autoDownloadEnabled ? 'Tự động tải' : 'Thủ công'}
                                </button>
                            </div>
                        )}
                    </div>

                    {/* Processing Status */}
                    {(isUploadingFile || isProcessingDocument) && (
                        <div className="processing-status">
                            <div className="processing-animation">
                                <div className="processing-spinner"></div>
                                <span className="processing-text">
                                    {documentProcessingStatus}
                                </span>
                            </div>
                            {uploadProgress > 0 && (
                                <div className="progress-bar">
                                    <div 
                                        className="progress-fill" 
                                        style={{ width: `${uploadProgress}%` }}
                                    ></div>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Voice Control Panel */}
                {(speechSupported || ttsSupported) && (
                    <div className="voice-control-panel">
                        {ttsSupported && (
                            <>
                                <button 
                                    className={`voice-control-btn ${voiceModeEnabled ? 'active' : ''}`}
                                    onClick={toggleVoiceMode}
                                    title={voiceModeEnabled ? 'Tắt chế độ giọng nói' : 'Bật chế độ giọng nói'}
                                >
                                    {voiceModeEnabled ? '🎤🔊' : '📝'} 
                                    {voiceModeEnabled ? 'Giọng nói' : 'Văn bản'}
                                </button>
                                
                                <button 
                                    className={`voice-control-btn ${autoPlayEnabled ? 'active' : ''}`}
                                    onClick={toggleAutoPlay}
                                    title={autoPlayEnabled ? 'Tắt tự động phát' : 'Bật tự động phát'}
                                >
                                    {autoPlayEnabled ? '🔊' : '🔇'} 
                                    {autoPlayEnabled ? 'Tự động' : 'Thủ công'}
                                </button>
                                
                                {isPlayingAudio && (
                                    <button 
                                        className="voice-control-btn stop-audio"
                                        onClick={stopCurrentAudio}
                                        title="Dừng phát âm thanh"
                                    >
                                        ⏹️ Dừng
                                    </button>
                                )}
                            </>
                        )}
                        
                        <div className="voice-status-indicator">
                            {voiceModeEnabled && ttsSupported && (
                                <span className="status-badge voice-mode">🎤🔊 Chế độ giọng nói</span>
                            )}
                            {forceVoiceMode && (
                                <span className="status-badge force-voice">🎤 Sẵn sàng nói</span>
                            )}
                            {isPlayingAudio && (
                                <span className="status-badge playing">🔊 Đang phát...</span>
                            )}
                            {currentDocument && (
                                <span className="status-badge document-active">📄 Có tài liệu</span>
                            )}
                        </div>
                    </div>
                )}

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
                            
                            <div className="quick-actions">
                                <h3>Gợi ý câu hỏi:</h3>
                                <div className="quick-buttons">
                                    <button 
                                        className="quick-btn"
                                        onClick={() => {
                                            setInputMessage('Thông tin tuyển sinh năm 2024');
                                            setLastUserInputMethod('text');
                                            setForceVoiceMode(false);
                                        }}
                                    >
                                        📚 Tuyển sinh 2024
                                    </button>
                                    <button 
                                        className="quick-btn"
                                        onClick={() => {
                                            setInputMessage('Học phí các ngành năm 2024');
                                            setLastUserInputMethod('text');
                                            setForceVoiceMode(false);
                                        }}
                                    >
                                        💰 Học phí
                                    </button>
                                    <button 
                                        className="quick-btn"
                                        onClick={() => {
                                            setInputMessage('Các ngành đào tạo tại BDU');
                                            setLastUserInputMethod('text');
                                            setForceVoiceMode(false);
                                        }}
                                    >
                                        🎓 Ngành học
                                    </button>
                                    <button 
                                        className="quick-btn"
                                        onClick={() => {
                                            setInputMessage('Cơ sở vật chất và ký túc xá');
                                            setLastUserInputMethod('text');
                                            setForceVoiceMode(false);
                                        }}
                                    >
                                        🏢 Cơ sở vật chất
                                    </button>
                                    {/* 🚀 NEW: Document-related quick actions */}
                                    <button 
                                        className="quick-btn document-btn"
                                        onClick={() => fileInputRef.current?.click()}
                                    >
                                        📄 Tải lên tài liệu
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

                                        {/* 🚀 NEW: Document info display */}
                                        {message.document_info && (
                                            <div className="document-info-display">
                                                <div className="document-preview">
                                                    <span className="document-icon">📄</span>
                                                    <div className="document-details">
                                                        <span className="document-name">{message.document_info.filename}</span>
                                                        <span className="document-meta">
                                                            {message.document_info.page_count} trang • 
                                                            {(message.document_info.file_size / 1024).toFixed(1)}KB
                                                        </span>
                                                    </div>
                                                </div>
                                            </div>
                                        )}

                                        {/* Audio playback controls */}
                                        {message.type === 'bot' && message.audio_content && !message.temporary && (
                                            <div className="audio-controls">
                                                <button 
                                                    className={`audio-play-btn ${isPlayingAudio && currentAudio ? 'playing' : ''}`}
                                                    onClick={() => playAudioFromBase64(message.audio_content)}
                                                    disabled={isPlayingAudio}
                                                    title="Phát âm thanh"
                                                >
                                                    {isPlayingAudio ? '🔊' : '▶️'} 
                                                    {isPlayingAudio ? 'Đang phát...' : 'Nghe phản hồi'}
                                                </button>
                                                {message.tts_info && (
                                                    <span className="audio-info">
                                                        🎵 Âm thanh có sẵn ({message.tts_info.processing_time?.toFixed(2) || 0}s)
                                                    </span>
                                                )}
                                            </div>
                                        )}

                                        {/* Enhanced metadata with document context */}
                                        {message.type === 'bot' && !message.isError && !message.temporary && (
                                            <div className="message-metadata-enhanced">
                                                {message.user_memory_applied && (
                                                    <div className="memory-applied-badge">
                                                        🧠 Đã áp dụng ghi nhớ cá nhân
                                                    </div>
                                                )}
                                                
                                                {message.external_api_used && (
                                                    <div className="external-api-badge">
                                                        🌐 Thông tin cá nhân từ hệ thống
                                                    </div>
                                                )}
                                                
                                                {message.mode === 'voice' && (
                                                    <div className="voice-mode-badge">
                                                        🎤🔊 Chế độ giọng nói
                                                    </div>
                                                )}

                                                {/* 🚀 NEW: Document context badges */}
                                                {message.document_context_used && (
                                                    <div className="document-context-badge">
                                                        📄 Dựa trên tài liệu đã tải
                                                    </div>
                                                )}

                                                {message.document_enhanced && (
                                                    <div className="document-enhanced-badge">
                                                        🔍 Phân tích từ tài liệu
                                                    </div>
                                                )}
                                            </div>
                                        )}

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
                                                                <div key={linkIdx} className="reference-link-item">
                                                                    <a
                                                                        href={link.url}
                                                                        target="_blank"
                                                                        rel="noopener noreferrer"
                                                                        className="reference-link"
                                                                    >
                                                                        📄 {link.stt || link.title || 'Tài liệu'}
                                                                    </a>
                                                                    {/* 🚀 NEW: Auto-download button */}
                                                                    {(link.url.includes('.pdf') || link.url.includes('.docx')) && (
                                                                        <button
                                                                            className="auto-download-btn"
                                                                            onClick={() => downloadAndProcessDocument(link.url, link.title)}
                                                                            disabled={isProcessingDocument}
                                                                            title="Tải và phân tích tài liệu này"
                                                                        >
                                                                            {isProcessingDocument ? '⏳' : '⚡'} 
                                                                            {isProcessingDocument ? 'Đang tải...' : 'Tải & phân tích'}
                                                                        </button>
                                                                    )}
                                                                </div>
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
                                            <span className="typing-text">
                                                {currentDocument ? 'AI đang phân tích tài liệu...' : 'AI đang suy nghĩ...'}
                                            </span>
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

                    {/* ✅ NEW: Hiển thị file đã chọn */}
                    {selectedFile && (
                        <div className="selected-file-indicator">
                            <span>📄 Đã chọn: <strong>{selectedFile.name}</strong></span>
                            <button 
                                onClick={() => setSelectedFile(null)} 
                                className="remove-file-btn"
                                title="Bỏ chọn file này"
                            >
                                ✕
                            </button>
                        </div>
                    )}

                    <div className="modern-input-container">
                        <div className="input-wrapper">
                            <textarea
                                ref={inputRef}
                                value={inputMessage}
                                onChange={handleInputChange}
                                onKeyPress={handleKeyPress}
                                placeholder={connectionStatus === 'connected' 
                                    ? (currentDocument 
                                        ? `Hỏi về tài liệu "${currentDocument.filename}" hoặc bất cứ điều gì khác...`
                                        : "Hỏi bất cứ điều gì về Đại học Bình Dương...")
                                    : "Đang kết nối đến server..."}
                                rows="1"
                                disabled={isLoading || connectionStatus !== 'connected' || isRecording}
                                maxLength={1000}
                                className="modern-input"
                            />
                            
                            <div className="input-actions">
                                {speechSupported && (
                                    <button 
                                        className={`voice-btn ${isRecording ? 'recording' : ''} ${isProcessingSpeech ? 'processing' : ''} ${forceVoiceMode ? 'voice-ready' : ''}`}
                                        onClick={isRecording ? stopRecording : startRecording}
                                        disabled={isLoading || connectionStatus !== 'connected' || isProcessingSpeech}
                                        title={isRecording ? 'Dừng ghi âm' : 'Bắt đầu ghi âm'}
                                    >
                                        {isRecording ? '⏹️' : (isProcessingSpeech ? '⏳' : '🎤')}
                                    </button>
                                )}
                                
                                {/* 🚀 NEW: File upload button in input */}
                                <button 
                                    className="file-btn"
                                    onClick={() => fileInputRef.current?.click()}
                                    disabled={isLoading || connectionStatus !== 'connected' || isUploadingFile}
                                    title="Tải lên tài liệu"
                                >
                                    {isUploadingFile ? '⏳' : '📎'}
                                </button>
                            </div>
                        </div>
                        
                        <div className="input-footer">
                            <div className="input-mode-indicator">
                                {currentDocument && (
                                    <span className="mode-badge document">
                                        📄 {currentDocument.filename}
                                    </span>
                                )}
                                {forceVoiceMode ? (
                                    <span className="mode-badge voice-ready">🎤🔊 Sẵn sàng nói</span>
                                ) : voiceModeEnabled && ttsSupported ? (
                                    <span className="mode-badge voice">🎤🔊 Chế độ giọng nói</span>
                                ) : (
                                    <span className="mode-badge text">📝 Chế độ văn bản</span>
                                )}
                            </div>
                            <div className="char-counter">
                                {inputMessage.length}/1000
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* 🚀 NEW: Document Preview Modal */}
            {showDocumentModal && currentDocument && (
                <div className="modal-overlay" onClick={toggleDocumentModal}>
                    <div className="modal-content document-modal" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3>📄 Chi tiết tài liệu</h3>
                            <button 
                                className="modal-close"
                                onClick={toggleDocumentModal}
                            >
                                ✕
                            </button>
                        </div>
                        <div className="modal-body">
                            <div className="document-details-full">
                                <div className="detail-row">
                                    <span className="detail-label">Tên file:</span>
                                    <span className="detail-value">{currentDocument.filename}</span>
                                </div>
                                <div className="detail-row">
                                    <span className="detail-label">Kích thước:</span>
                                    <span className="detail-value">{(currentDocument.file_size / 1024).toFixed(1)}KB</span>
                                </div>
                                <div className="detail-row">
                                    <span className="detail-label">Số trang:</span>
                                    <span className="detail-value">{currentDocument.page_count}</span>
                                </div>
                                <div className="detail-row">
                                    <span className="detail-label">Thời gian tải:</span>
                                    <span className="detail-value">{currentDocument.upload_time?.toLocaleString('vi-VN')}</span>
                                </div>
                                {currentDocument.processing_time && (
                                    <div className="detail-row">
                                        <span className="detail-label">Thời gian xử lý:</span>
                                        <span className="detail-value">{currentDocument.processing_time}ms</span>
                                    </div>
                                )}
                                {currentDocument.source_url && (
                                    <div className="detail-row">
                                        <span className="detail-label">Nguồn:</span>
                                        <span className="detail-value">
                                            <a href={currentDocument.source_url} target="_blank" rel="noopener noreferrer">
                                                Link gốc
                                            </a>
                                        </span>
                                    </div>
                                )}
                            </div>
                            
                            {documentPreview && (
                                <div className="document-preview-section">
                                    <h4>📖 Xem trước nội dung:</h4>
                                    <div className="document-preview-text">
                                        {documentPreview}
                                        {documentPreview.length >= 500 && '...'}
                                    </div>
                                </div>
                            )}
                        </div>
                        <div className="modal-footer">
                            <button 
                                className="btn-danger"
                                onClick={() => {
                                    clearDocumentContext();
                                    toggleDocumentModal();
                                }}
                            >
                                🗑️ Xóa khỏi ngữ cảnh
                            </button>
                            <button 
                                className="btn-confirm"
                                onClick={toggleDocumentModal}
                            >
                                Đóng
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Personalization Modal */}
            {showPersonalization && (
                <PersonalizationSettings
                    user={user}
                    onClose={() => setShowPersonalization(false)}
                    onUpdateSuccess={(newData) => {
                        console.log('Personalization updated:', newData);
                        checkUserAuth();
                    }}
                />
            )}

            {/* Rename Modal */}
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