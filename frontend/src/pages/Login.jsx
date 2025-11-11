import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { axiosInstance, API } from "../api";
import "../styles/Login.css";

function Login() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!email || !password) {
      alert("Please enter email and password.");
      return;
    }
    setLoading(true);
    try {
      const res = await axiosInstance.post(API.LOGIN, { email, password });
      const accessToken = res?.data?.access_token;
      if (!accessToken) throw new Error("No access token returned by server.");

      localStorage.setItem("token", accessToken);

      window.dispatchEvent(new Event("authChange"));

      navigate("/dashboard");
    } catch (err) {
      console.error("Login error:", err);
      const message =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        err.message ||
        "Login failed";
      alert(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-wrapper">
      <div className="login-card">
        <h3 className="login-subtitle">Please enter your details</h3>
        <h1 className="login-title">Welcome back</h1>

        <form className="login-form" onSubmit={handleSubmit}>
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
            placeholder="Password"
            className="input"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? "Signing in..." : "Sign in"}
          </button>

          <p className="signup-text">
            Don’t have an account? <Link to="/signup">Sign up</Link>
          </p>
        </form>
      </div>
    </div>
  );
}

export default Login;
