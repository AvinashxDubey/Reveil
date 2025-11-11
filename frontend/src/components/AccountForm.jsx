// src/components/AccountForm.jsx
import React, { useState } from "react";
import "../styles/AccountForm.css"; // keep your styles
import { predict } from "../api";   // your axios wrapper

export default function AccountForm() {
  const [form, setForm] = useState({
    created_at_local: "",
    username: "",
    tweet: "",
    hashtags: "",
    retweet_count: "",
    mention_count: "",
    follower_count: "",
    verified: false,
  });

  // UI state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null); // will hold PredictionResponse
  const [error, setError] = useState(null);

  function toIsoZ(datetimeLocal) {
    if (!datetimeLocal) return null;
    const d = new Date(datetimeLocal);
    if (isNaN(d.getTime())) return null;
    return d.toISOString();
  }

  function handleChange(e) {
    const { name, value, type, checked } = e.target;
    setForm((prev) => ({ ...prev, [name]: type === "checkbox" ? checked : value }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);

    const payload = {
      created_at: toIsoZ(form.created_at_local) || new Date().toISOString(),
      username: form.username.trim(),
      tweet: form.tweet,
      hashtags: form.hashtags.trim(),
      retweet_count: Number.parseInt(form.retweet_count || "0", 10),
      mention_count: Number.parseInt(form.mention_count || "0", 10),
      follower_count: Number.parseInt(form.follower_count || "0", 10),
      verified: Boolean(form.verified),
    };

    if (!payload.username || !payload.tweet) {
      setError("username and tweet are required.");
      return;
    }
    if (
      Number.isNaN(payload.retweet_count) ||
      Number.isNaN(payload.mention_count) ||
      Number.isNaN(payload.follower_count)
    ) {
      setError("Please enter valid numbers for counts.");
      return;
    }

    setLoading(true);
    try {
      const data = await predict(payload);
      setResult(data);
    } catch (err) {
      if (err?.response) {
        setError(`Server ${err.response.status}: ${JSON.stringify(err.response.data)}`);
      } else if (err?.request) {
        setError("No response from server. Is the backend running?");
      } else {
        setError(err?.message || "Unknown error");
      }
    } finally {
      setLoading(false);
    }
  }

  // Small inline styles for the result card so it looks okay without extra CSS
  const cardStyle = {
    borderRadius: 10,
    padding: "1rem",
    marginTop: "1rem",
    boxShadow: "0 6px 18px rgba(16,24,40,0.06)",
    background: "#fff",
    border: "1px solid rgba(16,24,40,0.04)",
  };

  const labelStyle = { fontWeight: 700, marginRight: 8 };

  return (
    <>
      <form className="form" onSubmit={handleSubmit}>
        <div className="row">
          <input
            name="username"
            value={form.username}
            onChange={handleChange}
            className="input"
            placeholder="Username"
          />
          <input
            name="created_at_local"
            value={form.created_at_local}
            onChange={handleChange}
            className="input"
            type="datetime-local"
            title="Account creation date/time (local)"
          />
        </div>

        <div className="row">
          <textarea
            name="tweet"
            value={form.tweet}
            onChange={handleChange}
            className="input"
            placeholder="Tweet text"
            rows={3}
          />
        </div>

        <div className="rowSmall">
          <input
            name="hashtags"
            value={form.hashtags}
            onChange={handleChange}
            className="inputSmall"
            placeholder="Hashtags (space-separated, e.g. #foo #bar)"
          />
          <input
            name="retweet_count"
            value={form.retweet_count}
            onChange={handleChange}
            className="inputSmall"
            placeholder="Retweet count"
            type="number"
            min="0"
          />
          <input
            name="mention_count"
            value={form.mention_count}
            onChange={handleChange}
            className="inputSmall"
            placeholder="Mention count"
            type="number"
            min="0"
          />
          <input
            name="follower_count"
            value={form.follower_count}
            onChange={handleChange}
            className="inputSmall"
            placeholder="Follower count"
            type="number"
            min="0"
          />
        </div>

        <div className="flagsGrid">
          <label className="check">
            <input
              type="checkbox"
              name="verified"
              checked={form.verified}
              onChange={handleChange}
            />
            verified
          </label>
        </div>

        <div className="actions">
          <button type="submit" className="submit" disabled={loading}>
            {loading ? "Checking..." : "Check Account"}
          </button>
        </div>
      </form>

      {/* Error */}
      {error && (
        <div style={{ ...cardStyle, borderColor: "rgba(255, 50, 50, 0.12)" }}>
          <div style={{ color: "#b00020", fontWeight: 700 }}>Error</div>
          <div style={{ marginTop: 6 }}>{error}</div>
        </div>
      )}

      {/* Result */}
      {result && (
        <div style={cardStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <div>
              <div style={{ fontSize: "1.1rem", fontWeight: 800 }}>
                Prediction: <span style={{ color: result.prediction === "human" ? "#0b8a3e" : "#c0392b" }}>{result.prediction}</span>
              </div>
              <div style={{ marginTop: 4, color: "#374151" }}>
                Confidence: <strong>{(result.confidence ?? 0).toFixed(4)}</strong>
              </div>
            </div>
            <div style={{ textAlign: "right", color: "#6b7280" }}>
              <div>User ID: {result.user_id}</div>
              <div style={{ fontSize: 12 }}>{new Date(result.timestamp).toLocaleString()}</div>
            </div>
          </div>

          <hr style={{ margin: "12px 0", border: "none", borderTop: "1px solid rgba(16,24,40,0.06)" }} />

          <div>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>Features calculated</div>
            <div style={{ fontSize: 14, color: "#111827" }}>
              {/* render features as a list */}
              {result.features_calculated && typeof result.features_calculated === "object" ? (
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {Object.entries(result.features_calculated).map(([k, v]) => (
                    <li key={k}>
                      <span style={labelStyle}>{k}:</span> {String(v)}
                    </li>
                  ))}
                </ul>
              ) : (
                <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(result.features_calculated, null, 2)}</pre>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
