// src/api.js
import axios from "axios";

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const axiosInstance = axios.create({
  baseURL: BASE,
  withCredentials: false,
  headers: {
    Accept: "application/json",
    "Content-Type": "application/json",
  },
});

axiosInstance.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Endpoints
export const API = {
  LOGIN: "/auth/login",
  REGISTER: "/auth/register",
  PREDICT: "/predict/", // note the trailing slash to match your FastAPI router prefix
};

// Helper functions for endpoints
export async function login(credentials) {
  const res = await axiosInstance.post(API.LOGIN, credentials);
  return res.data;
}

export async function register(payload) {
  const res = await axiosInstance.post(API.REGISTER, payload);
  return res.data;
}

/**
 * Send features/tweet payload to /predict/ and return the PredictionResponse (res.data).
 * payload should match your Pydantic PredictionRequest shape.
 *
 * Example payload:
 * {
 *   created_at: "2023-01-15T10:30:00Z",
 *   username: "john",
 *   tweet: "hello",
 *   hashtags: "#x #y",
 *   retweet_count: 0,
 *   mention_count: 0,
 *   follower_count: 10,
 *   verified: false
 * }
 */
export async function predict(payload) {
  const res = await axiosInstance.post(API.PREDICT, payload);
  return res.data;
}

export default axiosInstance;
