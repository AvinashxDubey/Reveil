// src/components/Signup.jsx
import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { FcGoogle } from "react-icons/fc";
import { axiosInstance, API } from "../api";
import "../styles/Signup.css";

function Signup() {
  const navigate = useNavigate();
  const [name, setName] = useState(""); // backend expects "name"
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const googleAuth = () => {
    alert("Google OAuth is not set up on the backend. Implement /auth/google in FastAPI first.");
  };

  async function handleSubmit(e) {
    e.preventDefault();
    if (!name || !email || !password) {
      alert("All fields are required.");
      return;
    }
    setLoading(true);
    try {
      const res = await axiosInstance.post(API.REGISTER, { name, email, password });
      // backend returns message and user info: { message, name, email }
      alert(res.data?.message || "Registered");
      // after signup, optionally log in automatically by calling /auth/login
      // we'll try to login automatically for convenience:
      try {
        const loginRes = await axiosInstance.post(API.LOGIN, { email, password });
        const accessToken = loginRes?.data?.access_token;
        if (accessToken) {
          localStorage.setItem("token", accessToken);
        }
      } catch (loginErr) {
        console.warn("Auto-login failed after register:", loginErr);
      }
      navigate("/dashboard");
    } catch (err) {
      console.error("Signup error:", err);
      const message = err?.response?.data?.detail || err?.response?.data?.message || err.message || "Signup failed";
      alert(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="signup-wrapper">
      <div className="signup-card">
        <h3 className="signup-subtitle">Create your account</h3>
        <h1 className="signup-title">Get started</h1>

        <form className="signup-form" onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Name"
            className="input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />
          <input
            type="email"
            placeholder="Email address"
            className="input"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            type="password"
            placeholder="Password (min 8 chars)"
            className="input"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            minLength={8}
            required
          />

          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? "Signing up..." : "Sign up"}
          </button>

          <button type="button" className="btn-google" onClick={googleAuth}>
            <FcGoogle size={20} />
            <span>Sign up with Google</span>
          </button>

          <p className="login-text">
            Already have an account? <Link to="/login">Log in</Link>
          </p>
        </form>
      </div>
    </div>
  );
}

export default Signup;
