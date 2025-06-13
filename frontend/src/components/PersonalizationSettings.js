import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './PersonalizationSettings.css';

const PersonalizationSettings = ({ user, onClose, onUpdateSuccess }) => {
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [activeTab, setActiveTab] = useState('preferences');
    
    // Form data
    const [formData, setFormData] = useState({
        response_style: 'professional',
        department_priority: true,
        focus_areas: [],
        notification_preferences: {
            email_updates: true,
            system_notifications: true
        }
    });
    
    // Available options
    const [availableOptions, setAvailableOptions] = useState({
        response_styles: ['professional', 'friendly', 'technical', 'brief', 'detailed'],
        focus_areas: [],
        suggested_topics: [],
        quick_actions: []
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
                    response_style: currentPrefs.response_style || 'professional',
                    department_priority: currentPrefs.department_priority !== false,
                    focus_areas: currentPrefs.focus_areas || [],
                    notification_preferences: {
                        email_updates: currentPrefs.notification_preferences?.email_updates !== false,
                        system_notifications: currentPrefs.notification_preferences?.system_notifications !== false
                    }
                });
                setUserContext(preferencesResponse.data.data.user_context);
            }
            
            // Load available options
            const suggestionsResponse = await axios.get('/api/auth/chatbot/suggestions/');
            if (suggestionsResponse.data.success) {
                setAvailableOptions({
                    response_styles: ['professional', 'friendly', 'technical', 'brief', 'detailed'],
                    focus_areas: suggestionsResponse.data.data.valid_focus_areas || [],
                    suggested_topics: suggestionsResponse.data.data.suggested_topics || [],
                    quick_actions: suggestionsResponse.data.data.quick_actions || []
                });
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
        
        if (name.includes('.')) {
            // Handle nested properties (e.g., notification_preferences.email_updates)
            const [parent, child] = name.split('.');
            setFormData(prev => ({
                ...prev,
                [parent]: {
                    ...prev[parent],
                    [child]: type === 'checkbox' ? checked : value
                }
            }));
        } else {
            setFormData(prev => ({
                ...prev,
                [name]: type === 'checkbox' ? checked : value
            }));
        }
        
        // Clear messages when user makes changes
        if (error) setError('');
        if (success) setSuccess('');
    };

    const handleFocusAreaToggle = (area) => {
        setFormData(prev => ({
            ...prev,
            focus_areas: prev.focus_areas.includes(area)
                ? prev.focus_areas.filter(a => a !== area)
                : [...prev.focus_areas, area]
        }));
        
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

                {/* Tabs */}
                <div className="tabs">
                    <button 
                        className={`tab ${activeTab === 'preferences' ? 'active' : ''}`}
                        onClick={() => setActiveTab('preferences')}
                    >
                        🎛️ Tùy chọn
                    </button>
                    <button 
                        className={`tab ${activeTab === 'focus' ? 'active' : ''}`}
                        onClick={() => setActiveTab('focus')}
                    >
                        🎯 Chuyên môn
                    </button>
                    <button 
                        className={`tab ${activeTab === 'info' ? 'active' : ''}`}
                        onClick={() => setActiveTab('info')}
                    >
                        ℹ️ Thông tin
                    </button>
                </div>

                <div className="modal-content">
                    {activeTab === 'preferences' && (
                        <div className="tab-content">
                            <h3>🎨 Phong cách trả lời</h3>
                            <div className="response-styles">
                                {availableOptions.response_styles.map(style => (
                                    <label key={style} className="style-option">
                                        <input
                                            type="radio"
                                            name="response_style"
                                            value={style}
                                            checked={formData.response_style === style}
                                            onChange={handleInputChange}
                                        />
                                        <span className="style-label">
                                            {getResponseStyleDescription(style)}
                                        </span>
                                    </label>
                                ))}
                            </div>

                            <h3>🔧 Tùy chọn khác</h3>
                            <div className="other-options">
                                <label className="checkbox-option">
                                    <input
                                        type="checkbox"
                                        name="department_priority"
                                        checked={formData.department_priority}
                                        onChange={handleInputChange}
                                    />
                                    <span>Ưu tiên thông tin chuyên ngành</span>
                                </label>
                            </div>

                            <h3>📢 Thông báo</h3>
                            <div className="notification-options">
                                <label className="checkbox-option">
                                    <input
                                        type="checkbox"
                                        name="notification_preferences.email_updates"
                                        checked={formData.notification_preferences.email_updates}
                                        onChange={handleInputChange}
                                    />
                                    <span>Nhận thông báo qua email</span>
                                </label>
                                <label className="checkbox-option">
                                    <input
                                        type="checkbox"
                                        name="notification_preferences.system_notifications"
                                        checked={formData.notification_preferences.system_notifications}
                                        onChange={handleInputChange}
                                    />
                                    <span>Thông báo hệ thống</span>
                                </label>
                            </div>
                        </div>
                    )}

                    {activeTab === 'focus' && (
                        <div className="tab-content">
                            <h3>🎯 Lĩnh vực quan tâm</h3>
                            <p>Chọn các lĩnh vực bạn muốn chatbot tập trung hỗ trợ:</p>
                            
                            <div className="focus-areas">
                                {availableOptions.focus_areas.map(area => (
                                    <button
                                        key={area}
                                        className={`focus-area-btn ${formData.focus_areas.includes(area) ? 'selected' : ''}`}
                                        onClick={() => handleFocusAreaToggle(area)}
                                    >
                                        {area}
                                    </button>
                                ))}
                            </div>

                            <div className="selected-count">
                                Đã chọn: {formData.focus_areas.length} lĩnh vực
                            </div>
                        </div>
                    )}

                    {activeTab === 'info' && (
                        <div className="tab-content">
                            <h3>📊 Thống kê cá nhân hóa</h3>
                            
                            {availableOptions.suggested_topics.length > 0 && (
                                <div className="info-section">
                                    <h4>💡 Chủ đề gợi ý cho bạn:</h4>
                                    <ul className="suggested-list">
                                        {availableOptions.suggested_topics.map((topic, index) => (
                                            <li key={index}>{topic}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {availableOptions.quick_actions.length > 0 && (
                                <div className="info-section">
                                    <h4>⚡ Thao tác nhanh:</h4>
                                    <ul className="suggested-list">
                                        {availableOptions.quick_actions.map((action, index) => (
                                            <li key={index}>{action}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            <div className="info-section">
                                <h4>🤖 Về tính năng cá nhân hóa:</h4>
                                <ul className="info-list">
                                    <li>Chatbot sẽ xưng hô phù hợp với chức vụ của bạn</li>
                                    <li>Ưu tiên thông tin liên quan đến chuyên ngành</li>
                                    <li>Sử dụng phong cách trả lời phù hợp</li>
                                    <li>Tăng độ chính xác cho câu hỏi chuyên môn</li>
                                </ul>
                            </div>
                        </div>
                    )}
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