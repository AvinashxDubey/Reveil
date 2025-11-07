// src/App.jsx
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
 import Navbar from './components/Navbar'; // you said you already have it
import Homepage from './pages/Homepage';
import Login from './pages/Login';
import Signup from './pages/Signup';
import LandingPage from './pages/Landing';

export default function App(){
  return (
    <BrowserRouter>
      <Navbar /> 
      <Routes>
        <Route path="/" element={<Homepage/>} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/dashboard" element={<LandingPage/>} />
      </Routes>
    </BrowserRouter>
  );
}
