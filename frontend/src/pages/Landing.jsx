import React from "react";
import AccountForm from "../components/AccountForm";
import "../styles/Landing.css";

export default function LandingPage() {
  return (
    <div className="wrap">
      <header className="hero">
        <h1 className="title">Fake Account Detection System</h1>
        <p className="subtitle">
          Protecting ITBP and India’s social media landscape from fake accounts and misinformation.
        </p>
        <a className="cta" href="#form">
          View Demo
        </a>
      </header>

      <main className="main">
        <section id="form" className="formCard">
          <h2 className="formTitle">Check Account Details</h2>
          <AccountForm />
        </section>
      </main>
    </div>
  );
}
