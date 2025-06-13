import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ChatBot from './components/ChatBot';
import Login from './components/Login';
import './App.css';
import { API_BASE_URL } from './config.js';

function App() {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Setup axios defaults
        axios.defaults.baseURL = API_BASE_URL;
        axios.defaults.timeout = 30000;
        
        // Check if user is already authenticated
        checkAuthStatus();
    }, []);

    const checkAuthStatus = async () => {
        try {
            // Check for stored token
            const token = localStorage.getItem('auth_token');
            const userData = localStorage.getItem('user_data');
            
            if (token && userData) {
                // Set axios header
                axios.defaults.headers.common['Authorization'] = `Token ${token}`;
                
                // Verify token with backend
                const response = await axios.get('/api/auth/status/');
                
                if (response.data.authenticated) {
                    setUser(response.data.user);
                    setIsAuthenticated(true);
                } else {
                    // Token invalid, clear storage
                    handleLogout();
                }
            }
        } catch (error) {
            console.error('Auth check failed:', error);
            // Clear invalid auth data
            handleLogout();
        } finally {
            setLoading(false);
        }
    };

    const handleLoginSuccess = (userData) => {
        setUser(userData);
        setIsAuthenticated(true);
    };

    const handleLogout = async () => {
        try {
            // Call logout API if authenticated
            if (isAuthenticated) {
                await axios.post('/api/auth/logout/');
            }
        } catch (error) {
            console.error('Logout API error:', error);
        } finally {
            // Clear local storage and state regardless of API call result
            localStorage.removeItem('auth_token');
            localStorage.removeItem('user_data');
            delete axios.defaults.headers.common['Authorization'];
            setUser(null);
            setIsAuthenticated(false);
        }
    };

    // Loading screen
    if (loading) {
        return (
            <div className="app-loading">
                <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <h3>🤖 Đang khởi tạo hệ thống...</h3>
                    <p>Vui lòng chờ trong giây lát</p>
                </div>
            </div>
        );
    }

    return (
        <div className="App">
            {isAuthenticated ? (
                <div className="app-authenticated">
                    {/* Header with user info and logout */}
                    <header className="app-header">
                        <div className="header-left">
                            <h1>🎓 Chatbot Đại học Bình Dương</h1>
                            <p>Hệ thống hỗ trợ thông tin</p>
                        </div>
                        <div className="header-right">
                            <div className="user-info">
                                <div className="user-details">
                                    <span className="user-name">👤 {user?.full_name}</span>
                                    <span className="user-code">{user?.faculty_code}</span>
                                    <span className="user-department">{user?.department}</span>
                                </div>
                                <button 
                                    className="logout-btn"
                                    onClick={handleLogout}
                                    title="Đăng xuất"
                                >
                                    🚪 Đăng xuất
                                </button>
                            </div>
                        </div>
                    </header>

                    {/* Main Chat Interface */}
                    <main className="app-main">
                        <ChatBot user={user} />
                    </main>
                </div>
            ) : (
                <Login onLoginSuccess={handleLoginSuccess} />
            )}
        </div>
    );
}

export default App;