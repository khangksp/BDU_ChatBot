import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Login.css';

import { API_BASE_URL } from '../config.js';

const Login = ({ onLoginSuccess }) => {
    const [formData, setFormData] = useState({
        faculty_code: '',
        password: '',
        remember_me: false
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [connectionStatus, setConnectionStatus] = useState('checking');
    const [showForgotPassword, setShowForgotPassword] = useState(false);
    
    // Forgot password form
    const [forgotPasswordData, setForgotPasswordData] = useState({
        faculty_code: '',
        email: ''
    });
    const [forgotPasswordLoading, setForgotPasswordLoading] = useState(false);
    const [forgotPasswordMessage, setForgotPasswordMessage] = useState('');

    useEffect(() => {
        // Test connection khi component mount
        testConnection();
        // Check if already logged in
        checkAuthStatus();
    }, []);

    const testConnection = async () => {
        try {
            await axios.get(`${API_BASE_URL}/api/health/`);
            setConnectionStatus('connected');
        } catch (error) {
            console.error('Connection test failed:', error);
            setConnectionStatus('error');
            setError('Không thể kết nối đến server. Vui lòng kiểm tra server có đang chạy không.');
        }
    };

    const checkAuthStatus = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/api/auth/status/`);
            if (response.data.authenticated) {
                onLoginSuccess(response.data.user);
            }
        } catch (error) {
            console.log('Not authenticated');
        }
    };

    const handleInputChange = (e) => {
        const { name, value, type, checked } = e.target;

        let processedValue = value;
        if (name === 'faculty_code') {
            processedValue = value.toUpperCase();
        }

        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : processedValue
        }));
        // Clear error when user starts typing
        if (error) setError('');
    };

    const handleForgotPasswordChange = (e) => {
        const { name, value } = e.target;
        
        // Chỉ uppercase faculty_code
        let processedValue = value;
        if (name === 'faculty_code') {
            processedValue = value.toUpperCase();
        }
        
        setForgotPasswordData(prev => ({
            ...prev,
            [name]: processedValue
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        
        if (!formData.faculty_code.trim() || !formData.password.trim()) {
            setError('Vui lòng nhập đầy đủ mã giảng viên và mật khẩu');
            return;
        }

        setLoading(true);
        setError('');

        try {
            const response = await axios.post(`${API_BASE_URL}/api/auth/login/`, {
                faculty_code: formData.faculty_code.trim(),
                password: formData.password,
                remember_me: formData.remember_me
            });

            if (response.data.success) {
                // Store token
                localStorage.setItem('auth_token', response.data.data.token);
                localStorage.setItem('user_data', JSON.stringify(response.data.data.user));
                
                // Set axios default header
                axios.defaults.headers.common['Authorization'] = `Token ${response.data.data.token}`;
                
                // Call success callback
                onLoginSuccess(response.data.data.user);
            } else {
                setError(response.data.message || 'Đăng nhập thất bại');
            }
        } catch (error) {
            console.error('Login error:', error);
            
            if (error.response?.data?.message) {
                setError(error.response.data.message);
            } else if (error.response?.status === 401) {
                setError('Mã giảng viên hoặc mật khẩu không chính xác');
            } else if (error.code === 'ECONNABORTED') {
                setError('Timeout: Server mất quá nhiều thời gian để phản hồi');
            } else {
                setError('Lỗi kết nối. Vui lòng thử lại sau.');
            }
        } finally {
            setLoading(false);
        }
    };

    const handleForgotPassword = async (e) => {
        e.preventDefault();
        
        if (!forgotPasswordData.faculty_code.trim() || !forgotPasswordData.email.trim()) {
            setForgotPasswordMessage('Vui lòng nhập đầy đủ mã giảng viên và email');
            return;
        }

        setForgotPasswordLoading(true);
        setForgotPasswordMessage('');

        try {
            const response = await axios.post(`${API_BASE_URL}/api/auth/password/reset/request/`, {
                faculty_code: forgotPasswordData.faculty_code.trim(),
                email: forgotPasswordData.email.trim()
            });

            if (response.data.success) {
                setForgotPasswordMessage(
                    `✅ ${response.data.message}${response.data.debug_info ? 
                    `\n\n🔧 Debug info:\nToken: ${response.data.debug_info.token}\nHết hạn: ${new Date(response.data.debug_info.expires_at).toLocaleString('vi-VN')}` : ''}`
                );
                
                // Reset form
                setForgotPasswordData({ faculty_code: '', email: '' });
            } else {
                setForgotPasswordMessage(`❌ ${response.data.message}`);
            }
        } catch (error) {
            console.error('Forgot password error:', error);
            setForgotPasswordMessage('❌ Lỗi khi gửi yêu cầu reset password. Vui lòng thử lại.');
        } finally {
            setForgotPasswordLoading(false);
        }
    };

    const retryConnection = () => {
        setConnectionStatus('checking');
        setError('');
        testConnection();
    };

    if (connectionStatus === 'checking') {
        return (
            <div className="login-container">
                <div className="login-box">
                    <div className="connection-checking">
                        <div className="spinner"></div>
                        <h3>🔄 Đang kết nối đến server...</h3>
                        <p>Vui lòng chờ trong giây lát</p>
                    </div>
                </div>
            </div>
        );
    }

    if (connectionStatus === 'error') {
        return (
            <div className="login-container">
                <div className="login-box">
                    <div className="connection-error">
                        <h3>🔴 Lỗi kết nối server</h3>
                        <p>{error}</p>
                        <div className="error-help">
                            <h4>Hướng dẫn khắc phục:</h4>
                            <ol>
                                <li>Kiểm tra server Django đang chạy: <code>python manage.py runserver</code></li>
                                <li>Kiểm tra URL: <code>http://127.0.0.1:8000/api/health/</code></li>
                                <li>Kiểm tra CORS settings trong Django</li>
                            </ol>
                        </div>
                        <button onClick={retryConnection} className="retry-btn">
                            🔄 Thử lại
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="login-container">
            <div className="login-header">
                <h1>🎓 Chatbot Đại học Bình Dương</h1>
                <p>Hệ thống hỗ trợ thông tin tuyển sinh và đào tạo</p>
            </div>

            <div className="login-box">
                {!showForgotPassword ? (
                    <>
                        <div className="login-form-header">
                            <h2>🔐 Đăng nhập</h2>
                            <p>Dành cho cán bộ, giảng viên</p>
                        </div>

                        <form onSubmit={handleSubmit} className="login-form">
                            {error && (
                                <div className="error-message">
                                    ⚠️ {error}
                                </div>
                            )}

                            <div className="form-group">
                                <label htmlFor="faculty_code">👤 Mã giảng viên</label>
                                <input
                                    type="text"
                                    id="faculty_code"
                                    name="faculty_code"
                                    value={formData.faculty_code}
                                    onChange={handleInputChange}
                                    placeholder="VD: GV001, ADMIN001"
                                    maxLength={20}
                                    autoComplete="username"
                                    disabled={loading}
                                />
                            </div>

                            <div className="form-group">
                                <label htmlFor="password">🔑 Mật khẩu</label>
                                <div className="password-input-wrapper">
                                    <input
                                        type={showPassword ? "text" : "password"}
                                        id="password"
                                        name="password"
                                        value={formData.password}
                                        onChange={handleInputChange}
                                        placeholder="Nhập mật khẩu"
                                        autoComplete="current-password"
                                        disabled={loading}
                                    />
                                    <button
                                        type="button"
                                        className="password-toggle"
                                        onClick={() => setShowPassword(!showPassword)}
                                        disabled={loading}
                                    >
                                        {showPassword ? '🙈' : '👁️'}
                                    </button>
                                </div>
                            </div>

                            <div className="form-options">
                                <label className="checkbox-label">
                                    <input
                                        type="checkbox"
                                        name="remember_me"
                                        checked={formData.remember_me}
                                        onChange={handleInputChange}
                                        disabled={loading}
                                    />
                                    <span>🔄 Ghi nhớ đăng nhập (2 tuần)</span>
                                </label>

                                <button
                                    type="button"
                                    className="forgot-password-link"
                                    onClick={() => setShowForgotPassword(true)}
                                    disabled={loading}
                                >
                                    🔑 Quên mật khẩu?
                                </button>
                            </div>

                            <button
                                type="submit"
                                className="login-btn"
                                disabled={loading || !formData.faculty_code.trim() || !formData.password.trim()}
                            >
                                {loading ? '⏳ Đang đăng nhập...' : '🚪 Đăng nhập'}
                            </button>
                        </form>

                        <div className="login-help">
                            <h4>💡 Tài khoản mẫu để test:</h4>
                            <div className="sample-accounts">
                                <div className="sample-account">
                                    <strong>ADMIN001</strong> / admin123456 (Quản trị)
                                </div>
                                <div className="sample-account">
                                    <strong>GV001</strong> / gv001@2024 (Giảng viên)
                                </div>
                                <div className="sample-account">
                                    <strong>TEST</strong> / 123456 (Test)
                                </div>
                            </div>
                        </div>
                    </>
                ) : (
                    <>
                        <div className="login-form-header">
                            <h2>🔑 Quên mật khẩu</h2>
                            <p>Nhập thông tin để reset mật khẩu</p>
                        </div>

                        <form onSubmit={handleForgotPassword} className="login-form">
                            {forgotPasswordMessage && (
                                <div className={`message ${forgotPasswordMessage.includes('✅') ? 'success' : 'error'}`}>
                                    <pre>{forgotPasswordMessage}</pre>
                                </div>
                            )}

                            <div className="form-group">
                                <label htmlFor="forgot_faculty_code">👤 Mã giảng viên</label>
                                <input
                                    type="text"
                                    id="forgot_faculty_code"
                                    name="faculty_code"
                                    value={forgotPasswordData.faculty_code}
                                    onChange={handleForgotPasswordChange}
                                    placeholder="VD: GV001"
                                    maxLength={20}
                                    disabled={forgotPasswordLoading}
                                />
                            </div>

                            <div className="form-group">
                                <label htmlFor="forgot_email">📧 Email đăng ký</label>
                                <input
                                    type="email"
                                    id="forgot_email"
                                    name="email"
                                    value={forgotPasswordData.email}
                                    onChange={handleForgotPasswordChange}
                                    placeholder="email@bdu.edu.vn"
                                    disabled={forgotPasswordLoading}
                                />
                            </div>

                            <div className="form-actions">
                                <button
                                    type="button"
                                    className="back-btn"
                                    onClick={() => {
                                        setShowForgotPassword(false);
                                        setForgotPasswordMessage('');
                                        setForgotPasswordData({ faculty_code: '', email: '' });
                                    }}
                                    disabled={forgotPasswordLoading}
                                >
                                    ← Quay lại đăng nhập
                                </button>

                                <button
                                    type="submit"
                                    className="submit-btn"
                                    disabled={forgotPasswordLoading || !forgotPasswordData.faculty_code.trim() || !forgotPasswordData.email.trim()}
                                >
                                    {forgotPasswordLoading ? '⏳ Đang gửi...' : '📨 Gửi yêu cầu'}
                                </button>
                            </div>
                        </form>
                    </>
                )}
            </div>

            <div className="login-footer">
                <div className="footer-info">
                    <span>🤖 Powered by AI • PhoBERT + Whisper + FAISS</span>
                    <span>Made with ❤️ for Đại học Bình Dương</span>
                </div>
                <div className="debug-info">
                    Backend: {API_BASE_URL} | Status: {connectionStatus}
                </div>
            </div>
        </div>
    );
};

export default Login;