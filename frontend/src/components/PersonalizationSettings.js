import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './PersonalizationSettings.css';

const PersonalizationSettings = ({ user, onClose, onUpdateSuccess }) => {
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    
    // ✅ SIMPLIFIED: Form data với only user_memory_prompt và department_priority
    const [formData, setFormData] = useState({
        user_memory_prompt: '',
        department_priority: true
    });
    
    // Enhanced user context
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
                const data = preferencesResponse.data.data;
                const currentPrefs = data.preferences || {};
                
                setFormData({
                    user_memory_prompt: currentPrefs.user_memory_prompt || '',
                    department_priority: currentPrefs.department_priority !== false
                });
                
                setUserContext(data.user_context);
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
            <div className="personalization-modal enhanced">
                <div className="modal-header">
                    <h2>🧠 Tùy chỉnh "Bộ não" cho Chatbot</h2>
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
                            {/* ✅ NEW: Current memory status indicator */}
                            <div className="current-memory-indicator">
                                <span className={`memory-badge ${formData.user_memory_prompt.trim() ? 'active' : 'inactive'}`}>
                                    {formData.user_memory_prompt.trim() ? 
                                        `🧠 Đã có ghi nhớ (${formData.user_memory_prompt.length} ký tự)` : 
                                        '💭 Chưa có ghi nhớ cá nhân'
                                    }
                                </span>
                            </div>
                        </div>
                    </div>
                )}

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
                        
                        {/* ✅ MAIN SECTION: User Memory Prompt (The Star of the Show) */}
                        <div className="setting-group user-memory-main">
                            <h3>🧠 Ghi nhớ và chỉ dẫn riêng cho ChatBDU</h3>
                            <div className="memory-description">
                                <p className="setting-description main">
                                    Đây là "bộ não" cá nhân của ChatBDU dành riêng cho bạn! Viết những quy tắc, 
                                    sở thích, cách làm việc mà bạn muốn ChatBDU luôn ghi nhớ và tuân thủ.
                                </p>
                                <div className="memory-benefits">
                                    <h5>💡 ChatBDU sẽ nhớ và áp dụng:</h5>
                                    <div className="benefit-tags">
                                        <span className="benefit-tag">🎯 Phong cách trả lời của bạn</span>
                                        <span className="benefit-tag">📋 Quy tắc riêng của bạn</span>
                                        <span className="benefit-tag">🎨 Sở thích cá nhân</span>
                                        <span className="benefit-tag">🏢 Thông tin công việc</span>
                                        <span className="benefit-tag">📚 Lĩnh vực quan tâm</span>
                                        <span className="benefit-tag">⚡ Cách làm việc hiệu quả</span>
                                    </div>
                                </div>
                            </div>
                            
                            {/* ✅ ENHANCED: Memory prompt examples và templates */}
                            <div className="memory-examples">
                                <h5>📝 Ví dụ nội dung bạn có thể viết:</h5>
                                <div className="example-cards">
                                    <div className="example-card" onClick={() => {
                                        const example = "Phong cách: Luôn trả lời chi tiết, thân thiện với emoji. Tôi thích có ví dụ cụ thể trong mỗi câu trả lời.";
                                        setFormData(prev => ({...prev, user_memory_prompt: example}));
                                    }}>
                                        <div className="example-header">🎨 Phong cách trả lời</div>
                                        <div className="example-content">"Luôn trả lời chi tiết, thân thiện với emoji..."</div>
                                    </div>
                                    
                                    <div className="example-card" onClick={() => {
                                        const example = "Tôi là giảng viên khoa CNTT, chuyên về AI và Machine Learning. Hãy ưu tiên thông tin về công nghệ, nghiên cứu khoa học.";
                                        setFormData(prev => ({...prev, user_memory_prompt: example}));
                                    }}>
                                        <div className="example-header">🎓 Chuyên môn</div>
                                        <div className="example-content">"Tôi chuyên về AI, ưu tiên thông tin công nghệ..."</div>
                                    </div>
                                    
                                    <div className="example-card" onClick={() => {
                                        const example = "Quy tắc: Không dùng từ kỹ thuật phức tạp. Tôi thích câu trả lời ngắn gọn, dễ hiểu. Luôn kết thúc bằng câu hỏi để tôi có thể hỏi thêm.";
                                        setFormData(prev => ({...prev, user_memory_prompt: example}));
                                    }}>
                                        <div className="example-header">📏 Quy tắc riêng</div>
                                        <div className="example-content">"Không dùng từ kỹ thuật, câu trả lời ngắn gọn..."</div>
                                    </div>
                                </div>
                            </div>
                            
                            {/* ✅ MAIN TEXTAREA: The Central Feature */}
                            <div className="memory-input-section">
                                <label htmlFor="user_memory_prompt" className="memory-label">
                                    ✍️ Viết ghi nhớ và chỉ dẫn của bạn:
                                </label>
                                <textarea
                                    id="user_memory_prompt"
                                    name="user_memory_prompt"
                                    value={formData.user_memory_prompt}
                                    onChange={handleInputChange}
                                    placeholder={`Ví dụ:

Phong cách: Tôi thích câu trả lời chi tiết có ví dụ cụ thể. Luôn dùng emoji phù hợp để tạo không khí vui vẻ.

Về tôi: Tôi là ${userContext?.position_name || 'giảng viên'} khoa ${userContext?.department_name || 'CNTT'}, quan tâm đến AI và công nghệ giáo dục. Tôi thường làm việc với sinh viên năm cuối.

Quy tắc riêng:
- Luôn hỏi lại nếu thông tin chưa rõ
- Đề xuất các bước cụ thể khi hướng dẫn
- Nhắc nhở deadline và thời hạn quan trọng
- Ưu tiên thông tin mới nhất về quy định

Sở thích cá nhân: Tôi thích học hỏi công nghệ mới và chia sẻ kinh nghiệm với đồng nghiệp.`}
                                    rows={10}
                                    className="memory-prompt-textarea main-feature"
                                    maxLength={1500}
                                />
                                <div className="char-count enhanced">
                                    <span className={formData.user_memory_prompt.length > 1400 ? 'warning' : 
                                                   formData.user_memory_prompt.length > 1000 ? 'good' : 
                                                   formData.user_memory_prompt.length > 100 ? 'excellent' : ''}>
                                        {formData.user_memory_prompt.length}/1500 ký tự
                                    </span>
                                    {formData.user_memory_prompt.length > 1400 && (
                                        <span className="char-warning">⚠️ Gần đạt giới hạn</span>
                                    )}
                                    {formData.user_memory_prompt.length > 100 && formData.user_memory_prompt.length <= 1000 && (
                                        <span className="char-good">✨ Độ dài tốt</span>
                                    )}
                                    {formData.user_memory_prompt.length > 1000 && formData.user_memory_prompt.length <= 1400 && (
                                        <span className="char-excellent">🎯 Rất chi tiết</span>
                                    )}
                                </div>
                                
                                {/* Memory Effectiveness Indicator */}
                                <div className="memory-effectiveness">
                                    <div className="effectiveness-bar">
                                        <div className="effectiveness-label">Hiệu quả cá nhân hóa:</div>
                                        <div className="effectiveness-progress">
                                            <div 
                                                className={`effectiveness-fill ${
                                                    formData.user_memory_prompt.length > 500 ? 'high' :
                                                    formData.user_memory_prompt.length > 200 ? 'medium' : 'low'
                                                }`}
                                                style={{width: `${Math.min(100, (formData.user_memory_prompt.length / 800) * 100)}%`}}
                                            ></div>
                                        </div>
                                        <span className="effectiveness-text">
                                            {formData.user_memory_prompt.length > 500 ? 'Cao 🚀' :
                                             formData.user_memory_prompt.length > 200 ? 'Trung bình 📈' :
                                             formData.user_memory_prompt.length > 50 ? 'Thấp 📊' : 'Chưa có 💭'}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* ✅ SECONDARY SECTION: Department Priority */}
                        <div className="setting-group secondary">
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
                            💡 Tip: Viết càng chi tiết, ChatBDU sẽ hiểu và phục vụ bạn càng tốt!
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