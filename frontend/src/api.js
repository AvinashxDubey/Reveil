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

export const API = {
  LOGIN: "/auth/login",       
  REGISTER: "/auth/register",  
};

export default axiosInstance;
