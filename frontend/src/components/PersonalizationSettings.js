import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './PersonalizationSettings.css';

const PersonalizationSettings = ({ user, onClose, onUpdateSuccess }) => {
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [testing, setTesting] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    
    // ✅ ENHANCED: Form data với comprehensive support
    const [formData, setFormData] = useState({
        user_memory_prompt: '',
        response_style: 'professional',
        department_priority: true
    });
    
    // Enhanced user context và style info
    const [userContext, setUserContext] = useState(null);
    const [styleInfo, setStyleInfo] = useState(null);
    const [testResults, setTestResults] = useState(null);

    // ✅ NEW: Style descriptions với emoji và detailed info
    const getEnhancedStyleInfo = () => ({
        professional: {
            emoji: '🏢',
            name: 'Chuyên nghiệp',
            description: 'Trang trọng, lịch sự',
            details: 'Phù hợp cho công việc chính thức, họp hành, báo cáo',
            example: 'Dạ thầy/cô, theo quy định của nhà trường...',
            recommended_for: ['Báo cáo', 'Họp hành', 'Công việc chính thức'],
            temperature: 'Thấp - Ổn định',
            color: '#2563eb'
        },
        friendly: {
            emoji: '😊',
            name: 'Thân thiện',
            description: 'Gần gũi, dễ gần',
            details: 'Tạo không khí thoải mái, vui vẻ, gần gũi',
            example: 'Dạ thầy/cô, em rất vui được hỗ trợ! 😊',
            recommended_for: ['Chat thường ngày', 'Tương tác sinh viên', 'Hướng dẫn'],
            temperature: 'Trung bình - Linh hoạt',
            color: '#10b981'
        },
        technical: {
            emoji: '🔧',
            name: 'Kỹ thuật',
            description: 'Chi tiết, thuật ngữ chuyên môn',
            details: 'Sử dụng thuật ngữ chính xác, giải thích kỹ thuật sâu',
            example: 'Dạ thầy/cô, theo specification kỹ thuật...',
            recommended_for: ['Giải thích kỹ thuật', 'Nghiên cứu', 'Chuyên môn sâu'],
            temperature: 'Thấp - Chính xác',
            color: '#7c3aed'
        },
        brief: {
            emoji: '⚡',
            name: 'Ngắn gọn',
            description: 'Trả lời súc tích',
            details: 'Đi thẳng vào vấn đề, tiết kiệm thời gian',
            example: 'Dạ thầy/cô, thời hạn là 15/12. 🎓',
            recommended_for: ['Câu hỏi nhanh', 'Thông tin cơ bản', 'Lịch bận'],
            temperature: 'Trung bình - Tập trung',
            color: '#f59e0b'
        },
        detailed: {
            emoji: '📚',
            name: 'Chi tiết',
            description: 'Giải thích đầy đủ',
            details: 'Cung cấp nhiều thông tin, ví dụ, ngữ cảnh',
            example: 'Dạ thầy/cô, về vấn đề này, em xin giải thích từng bước...',
            recommended_for: ['Hướng dẫn phức tạp', 'Giải thích quy trình', 'Đào tạo'],
            temperature: 'Cao - Mở rộng',
            color: '#dc2626'
        }
    });

    useEffect(() => {
        if (user) {
            loadPersonalizationData();
        }
    }, [user]);

    const loadPersonalizationData = async () => {
        setLoading(true);
        setError('');
        
        try {
            // Load current preferences with enhanced info
            const preferencesResponse = await axios.get('/api/auth/chatbot/preferences/');
            if (preferencesResponse.data.success) {
                const data = preferencesResponse.data.data;
                const currentPrefs = data.preferences || {};
                
                setFormData({
                    user_memory_prompt: currentPrefs.user_memory_prompt || '',
                    response_style: currentPrefs.response_style || 'professional',
                    department_priority: currentPrefs.department_priority !== false
                });
                
                setUserContext(data.user_context);
                setStyleInfo(data.style_info);
            }
            
        } catch (error) {
            console.error('Error loading personalization data:', error);
            setError('Không thể tải dữ liệu cá nhân hóa. Vui lòng thử lại.');
        } finally {
            setLoading(false);
        }
    };

    const handleInputChange = (e) => {
        const { name, value, type, checked } = e.target;
        
        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }));
        
        // Clear messages when user makes changes
        if (error) setError('');
        if (success) setSuccess('');
        if (testResults) setTestResults(null);
    };

    // ✅ NEW: Test response style functionality
    const handleTestStyle = async () => {
        if (!formData.response_style) return;
        
        setTesting(true);
        setTestResults(null);
        
        try {
            const response = await axios.post('/api/auth/chatbot/test-response-style/', {
                test_query: 'Hỏi về ngân hàng đề thi của khoa',
                current_style: formData.response_style
            });
            
            if (response.data.success) {
                setTestResults(response.data.data);
            } else {
                setError('Không thể test response style');
            }
        } catch (error) {
            console.error('Error testing style:', error);
            setError('Lỗi khi test phong cách trả lời');
        } finally {
            setTesting(false);
        }
    };

    const handleSave = async () => {
        setSaving(true);
        setError('');
        setSuccess('');
        
        try {
            const response = await axios.post('/api/auth/chatbot/preferences/update/', {
                preferences: formData
            });
            
            if (response.data.success) {
                const changesSummary = response.data.data.changes_summary || [];
                setSuccess(`✅ Cài đặt đã được lưu thành công! 
${changesSummary.length > 0 ? '📝 Thay đổi: ' + changesSummary.join(', ') : ''}`);
                
                if (onUpdateSuccess) {
                    onUpdateSuccess(response.data.data);
                }
                
                // Auto close after 3 seconds
                setTimeout(() => {
                    onClose();
                }, 3000);
            } else {
                setError(response.data.message || 'Có lỗi xảy ra khi lưu cài đặt');
            }
        } catch (error) {
            console.error('Error saving preferences:', error);
            if (error.response?.data?.message) {
                setError(error.response.data.message);
            } else if (error.response?.data?.validation_errors) {
                setError('Lỗi validation: ' + error.response.data.validation_errors.join(', '));
            } else {
                setError('Không thể lưu cài đặt. Vui lòng thử lại.');
            }
        } finally {
            setSaving(false);
        }
    };

    // ✅ NEW: Quick setup presets
    const handleQuickSetup = async (presetType) => {
        const presets = {
            teaching: {
                response_style: 'friendly',
                department_priority: true,
                user_memory_prompt: `Tôi tập trung vào hoạt động giảng dạy. Tôi thích câu trả lời thân thiện, dễ hiểu cho sinh viên.`
            },
            research: {
                response_style: 'technical',
                department_priority: true,
                user_memory_prompt: `Tôi chuyên về nghiên cứu khoa học. Tôi cần thông tin chi tiết, chính xác với thuật ngữ chuyên môn.`
            },
            administration: {
                response_style: 'professional',
                department_priority: false,
                user_memory_prompt: `Tôi làm công tác quản lý. Tôi cần thông tin chính thức, quy trình và thủ tục rõ ràng.`
            }
        };
        
        const preset = presets[presetType];
        if (preset) {
            setFormData(prev => ({
                ...prev,
                ...preset
            }));
            setSuccess(`✨ Đã áp dụng preset "${presetType}". Bạn có thể chỉnh sửa thêm trước khi lưu.`);
        }
    };

    if (loading) {
        return (
            <div className="personalization-overlay">
                <div className="personalization-modal">
                    <div className="loading-container">
                        <div className="spinner"></div>
                        <p>Đang tải cài đặt cá nhân hóa nâng cao...</p>
                    </div>
                </div>
            </div>
        );
    }

    const enhancedStyleInfo = getEnhancedStyleInfo();
    const currentStyleInfo = enhancedStyleInfo[formData.response_style];

    return (
        <div className="personalization-overlay">
            <div className="personalization-modal enhanced">
                <div className="modal-header">
                    <h2>🎯 Cài đặt Chatbot cá nhân hóa nâng cao</h2>
                    <button className="close-btn" onClick={onClose}>✕</button>
                </div>

                {/* Enhanced User Info Banner */}
                {userContext && (
                    <div className="user-info-banner enhanced">
                        <div className="user-avatar">
                            {userContext.department === 'cntt' ? '💻' :
                             userContext.department === 'duoc' ? '💊' :
                             userContext.department === 'dien_tu' ? '🔌' :
                             userContext.department === 'co_khi' ? '⚙️' :
                             userContext.department === 'y_khoa' ? '🏥' :
                             userContext.department === 'kinh_te' ? '💰' :
                             userContext.department === 'luat' ? '⚖️' : '👤'}
                        </div>
                        <div className="user-details">
                            <h3>{userContext.full_name}</h3>
                            <p>{userContext.position_name} - {userContext.department_name}</p>
                            <span className="faculty-code">#{userContext.faculty_code}</span>
                            {/* ✅ NEW: Current style indicator */}
                            <div className="current-style-indicator">
                                <span className="style-badge" style={{ backgroundColor: currentStyleInfo.color }}>
                                    {currentStyleInfo.emoji} {currentStyleInfo.name}
                                </span>
                            </div>
                        </div>
                    </div>
                )}

                {/* ✅ NEW: Quick Setup Buttons */}
                <div className="quick-setup-section">
                    <h4>⚡ Thiết lập nhanh</h4>
                    <div className="quick-setup-buttons">
                        <button 
                            type="button" 
                            className="quick-setup-btn teaching"
                            onClick={() => handleQuickSetup('teaching')}
                        >
                            👨‍🏫 Giảng dạy
                        </button>
                        <button 
                            type="button" 
                            className="quick-setup-btn research"
                            onClick={() => handleQuickSetup('research')}
                        >
                            🔬 Nghiên cứu
                        </button>
                        <button 
                            type="button" 
                            className="quick-setup-btn admin"
                            onClick={() => handleQuickSetup('administration')}
                        >
                            📊 Quản lý
                        </button>
                    </div>
                </div>

                {/* Messages */}
                {error && (
                    <div className="error-message enhanced">
                        ⚠️ {error}
                    </div>
                )}
                
                {success && (
                    <div className="success-message enhanced">
                        {success}
                    </div>
                )}

                {/* Enhanced Settings Content */}
                <div className="modal-content enhanced">
                    <div className="settings-section">
                        
                        {/* ✅ ENHANCED: Response Style Selection */}
                        <div className="setting-group enhanced-styles">
                            <h3>🎨 Phong cách trả lời ChatBDU</h3>
                            <p className="setting-description">
                                Chọn cách ChatBDU sẽ trả lời phù hợp với công việc và sở thích của bạn
                            </p>
                            
                            <div className="response-styles-grid">
                                {Object.entries(enhancedStyleInfo).map(([styleCode, styleData]) => (
                                    <label 
                                        key={styleCode} 
                                        className={`style-card ${formData.response_style === styleCode ? 'selected' : ''}`}
                                        style={{ borderColor: formData.response_style === styleCode ? styleData.color : '#e5e7eb' }}
                                    >
                                        <input
                                            type="radio"
                                            name="response_style"
                                            value={styleCode}
                                            checked={formData.response_style === styleCode}
                                            onChange={handleInputChange}
                                        />
                                        <div className="style-card-header">
                                            <span className="style-emoji">{styleData.emoji}</span>
                                            <div className="style-info">
                                                <h4 className="style-name">{styleData.name}</h4>
                                                <p className="style-description">{styleData.description}</p>
                                            </div>
                                        </div>
                                        <div className="style-card-body">
                                            <p className="style-details">{styleData.details}</p>
                                            <div className="style-example">
                                                <strong>Ví dụ:</strong> <em>{styleData.example}</em>
                                            </div>
                                            <div className="style-metadata">
                                                <small>🎯 Phù hợp: {styleData.recommended_for.join(', ')}</small>
                                                <small>🌡️ {styleData.temperature}</small>
                                            </div>
                                        </div>
                                    </label>
                                ))}
                            </div>

                            {/* ✅ NEW: Test Style Button */}
                            <div className="style-test-section">
                                <button 
                                    type="button" 
                                    className="test-style-btn"
                                    onClick={handleTestStyle}
                                    disabled={testing}
                                >
                                    {testing ? '⏳ Đang test...' : '🧪 Test phong cách này'}
                                </button>
                                
                                {testResults && (
                                    <div className="test-results">
                                        <h5>📊 Kết quả test:</h5>
                                        <p><strong>Query test:</strong> {testResults.test_query}</p>
                                        <p><strong>Phong cách hiện tại:</strong> {testResults.current_style_name}</p>
                                        <p><strong>Khuyến nghị:</strong> {testResults.recommendation}</p>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* ✅ ENHANCED: User Memory Prompt */}
                        <div className="setting-group">
                            <h3>🧠 ChatBDU sẽ nhớ gì về bạn?</h3>
                            <p className="setting-description">
                                Viết những thông tin bạn muốn ChatBDU luôn nhớ khi trò chuyện. 
                                Càng chi tiết, ChatBDU sẽ hỗ trợ bạn càng tốt!
                            </p>
                            
                            {/* ✅ NEW: Memory prompt templates */}
                            <div className="memory-templates">
                                <h5>💡 Gợi ý nội dung:</h5>
                                <div className="template-chips">
                                    <span className="template-chip" onClick={() => {
                                        const addition = "Tôi thích câu trả lời có ví dụ cụ thể. ";
                                        setFormData(prev => ({
                                            ...prev,
                                            user_memory_prompt: prev.user_memory_prompt + addition
                                        }));
                                    }}>+ Thích ví dụ cụ thể</span>
                                    
                                    <span className="template-chip" onClick={() => {
                                        const addition = "Tôi quan tâm đến nghiên cứu khoa học. ";
                                        setFormData(prev => ({
                                            ...prev,
                                            user_memory_prompt: prev.user_memory_prompt + addition
                                        }));
                                    }}>+ Nghiên cứu khoa học</span>
                                    
                                    <span className="template-chip" onClick={() => {
                                        const addition = "Tôi thường làm việc với sinh viên. ";
                                        setFormData(prev => ({
                                            ...prev,
                                            user_memory_prompt: prev.user_memory_prompt + addition
                                        }));
                                    }}>+ Làm việc với SV</span>
                                </div>
                            </div>
                            
                            <textarea
                                name="user_memory_prompt"
                                value={formData.user_memory_prompt}
                                onChange={handleInputChange}
                                placeholder={`Ví dụ: Tôi là ${userContext?.position_name || 'giảng viên'} khoa ${userContext?.department_name || 'CNTT'}, quan tâm đến AI và machine learning. Tôi thích câu trả lời chi tiết với ví dụ cụ thể. Tôi thường làm việc với sinh viên năm cuối...`}
                                rows={6}
                                className="memory-prompt-textarea enhanced"
                                maxLength={1000}
                            />
                            <div className="char-count">
                                <span className={formData.user_memory_prompt.length > 900 ? 'warning' : ''}>
                                    {formData.user_memory_prompt.length}/1000 ký tự
                                </span>
                                {formData.user_memory_prompt.length > 900 && (
                                    <span className="char-warning">⚠️ Gần đạt giới hạn</span>
                                )}
                            </div>
                        </div>

                        {/* ✅ Enhanced Department Priority */}
                        <div className="setting-group">
                            <h3>🎯 Tùy chọn chuyên ngành</h3>
                            <div className="department-priority-section enhanced">
                                <label className="toggle-option enhanced">
                                    <input
                                        type="checkbox"
                                        name="department_priority"
                                        checked={formData.department_priority}
                                        onChange={handleInputChange}
                                    />
                                    <span className="toggle-slider enhanced"></span>
                                    <div className="toggle-content">
                                        <strong>Ưu tiên thông tin chuyên ngành</strong>
                                        <p>ChatBDU sẽ tập trung vào thông tin liên quan đến ngành {userContext?.department_name}</p>
                                        {formData.department_priority ? (
                                            <small className="toggle-status enabled">✅ Đang bật - ChatBDU sẽ ưu tiên kiến thức chuyên ngành</small>
                                        ) : (
                                            <small className="toggle-status disabled">⭕ Đang tắt - ChatBDU sẽ trả lời thông tin chung</small>
                                        )}
                                    </div>
                                </label>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Enhanced Footer */}
                <div className="modal-footer enhanced">
                    <div className="footer-info">
                        <small>
                            💡 Tip: Sau khi lưu, hãy thử chat để xem sự thay đổi!
                        </small>
                    </div>
                    <div className="footer-buttons">
                        <button 
                            className="cancel-btn" 
                            onClick={onClose}
                            disabled={saving}
                        >
                            Hủy
                        </button>
                        <button 
                            className="save-btn enhanced" 
                            onClick={handleSave}
                            disabled={saving}
                        >
                            {saving ? '⏳ Đang lưu...' : '💾 Lưu cài đặt'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PersonalizationSettings;