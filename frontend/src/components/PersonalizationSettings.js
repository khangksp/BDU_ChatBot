import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './PersonalizationSettings.css';

const PersonalizationSettings = ({ user, onClose, onUpdateSuccess }) => {
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    
    // Simplified form data
    const [formData, setFormData] = useState({
        user_memory_prompt: '', // NEW: User's personal prompt
        response_style: 'professional',
        department_priority: true
    });
    
    // User context info
    const [userContext, setUserContext] = useState(null);

    useEffect(() => {
        if (user) {
            loadPersonalizationData();
        }
    }, [user]);

    const loadPersonalizationData = async () => {
        setLoading(true);
        setError('');
        
        try {
            // Load current preferences
            const preferencesResponse = await axios.get('/api/auth/chatbot/preferences/');
            if (preferencesResponse.data.success) {
                const currentPrefs = preferencesResponse.data.data.preferences || {};
                setFormData({
                    user_memory_prompt: currentPrefs.user_memory_prompt || '',
                    response_style: currentPrefs.response_style || 'professional',
                    department_priority: currentPrefs.department_priority !== false
                });
                setUserContext(preferencesResponse.data.data.user_context);
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
                setSuccess('✅ Cài đặt đã được lưu thành công!');
                if (onUpdateSuccess) {
                    onUpdateSuccess(response.data.data);
                }
                
                // Auto close after 2 seconds
                setTimeout(() => {
                    onClose();
                }, 2000);
            } else {
                setError(response.data.message || 'Có lỗi xảy ra khi lưu cài đặt');
            }
        } catch (error) {
            console.error('Error saving preferences:', error);
            if (error.response?.data?.message) {
                setError(error.response.data.message);
            } else {
                setError('Không thể lưu cài đặt. Vui lòng thử lại.');
            }
        } finally {
            setSaving(false);
        }
    };

    const getResponseStyleDescription = (style) => {
        const descriptions = {
            professional: '🏢 Chuyên nghiệp - Trang trọng, lịch sự',
            friendly: '😊 Thân thiện - Gần gũi, dễ gần',
            technical: '🔧 Kỹ thuật - Chi tiết, thuật ngữ chuyên môn',
            brief: '⚡ Ngắn gọn - Trả lời súc tích',
            detailed: '📚 Chi tiết - Giải thích đầy đủ'
        };
        return descriptions[style] || style;
    };

    if (loading) {
        return (
            <div className="personalization-overlay">
                <div className="personalization-modal">
                    <div className="loading-container">
                        <div className="spinner"></div>
                        <p>Đang tải cài đặt cá nhân hóa...</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="personalization-overlay">
            <div className="personalization-modal">
                <div className="modal-header">
                    <h2>🎯 Cài đặt Chatbot cá nhân hóa</h2>
                    <button className="close-btn" onClick={onClose}>✕</button>
                </div>

                {/* User Info Banner */}
                {userContext && (
                    <div className="user-info-banner">
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
                        </div>
                    </div>
                )}

                {/* Messages */}
                {error && (
                    <div className="error-message">
                        ⚠️ {error}
                    </div>
                )}
                
                {success && (
                    <div className="success-message">
                        {success}
                    </div>
                )}

                {/* Single Tab Content */}
                <div className="modal-content">
                    <div className="settings-section">
                        {/* 1. User Memory Prompt */}
                        <div className="setting-group">
                            <h3>🧠 Bạn muốn ChatBDU ghi nhớ điều gì?</h3>
                            <p className="setting-description">
                                Viết những thông tin bạn muốn ChatBDU luôn nhớ về bạn khi trò chuyện
                            </p>
                            <textarea
                                name="user_memory_prompt"
                                value={formData.user_memory_prompt}
                                onChange={handleInputChange}
                                placeholder="Ví dụ: Tôi là giảng viên khoa CNTT, quan tâm đến AI và machine learning. Tôi thích câu trả lời chi tiết với ví dụ cụ thể..."
                                rows={6}
                                className="memory-prompt-textarea"
                                maxLength={1000}
                            />
                            <div className="char-count">
                                {formData.user_memory_prompt.length}/1000 ký tự
                            </div>
                        </div>

                        {/* 2. Response Style */}
                        <div className="setting-group">
                            <h3>🎨 Phong cách trả lời</h3>
                            <p className="setting-description">
                                Chọn cách ChatBDU trả lời phù hợp với sở thích của bạn
                            </p>
                            <div className="response-styles">
                                {['professional', 'friendly', 'technical', 'brief', 'detailed'].map(style => (
                                    <label key={style} className={`style-option ${formData.response_style === style ? 'selected' : ''}`}>
                                        <input
                                            type="radio"
                                            name="response_style"
                                            value={style}
                                            checked={formData.response_style === style}
                                            onChange={handleInputChange}
                                        />
                                        <div className="style-content">
                                            <span className="style-label">
                                                {getResponseStyleDescription(style)}
                                            </span>
                                        </div>
                                    </label>
                                ))}
                            </div>
                        </div>

                        {/* 3. Department Priority */}
                        <div className="setting-group">
                            <h3>🎯 Tùy chọn chuyên ngành</h3>
                            <div className="department-priority-section">
                                <label className="toggle-option">
                                    <input
                                        type="checkbox"
                                        name="department_priority"
                                        checked={formData.department_priority}
                                        onChange={handleInputChange}
                                    />
                                    <span className="toggle-slider"></span>
                                    <div className="toggle-content">
                                        <strong>Ưu tiên thông tin chuyên ngành</strong>
                                        <p>ChatBDU sẽ tập trung vào thông tin liên quan đến ngành {userContext?.department_name}</p>
                                    </div>
                                </label>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="modal-footer">
                    <button 
                        className="cancel-btn" 
                        onClick={onClose}
                        disabled={saving}
                    >
                        Hủy
                    </button>
                    <button 
                        className="save-btn" 
                        onClick={handleSave}
                        disabled={saving}
                    >
                        {saving ? '⏳ Đang lưu...' : '💾 Lưu cài đặt'}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default PersonalizationSettings;