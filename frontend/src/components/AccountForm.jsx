import React, { useState } from "react";
import "../styles/AccountForm.css";

export default function AccountForm() {
  const [form, setForm] = useState({
    username: "",
    fullName: "",
    followers: "",
    following: "",
    joined: "",
    flags: {
      isPrivate: false,
      hasChannel: false,
      isBusiness: false,
      hasGuides: false,
      hasUrl: false,
    },
  });

  function handleChange(e) {
    const { name, value, type, checked } = e.target;
    if (name.startsWith("flag_")) {
      const key = name.replace("flag_", "");
      setForm((prev) => ({
        ...prev,
        flags: { ...prev.flags, [key]: checked },
      }));
    } else {
      setForm((prev) => ({ ...prev, [name]: value }));
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    console.log("Form submitted:", form);
    alert("Form submitted (check console)");
  }

  return (
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
          name="fullName"
          value={form.fullName}
          onChange={handleChange}
          className="input"
          placeholder="Full Name"
        />
      </div>

      <div className="rowSmall">
        <input
          name="followers"
          value={form.followers}
          onChange={handleChange}
          className="inputSmall"
          placeholder="Number of Followers"
          type="number"
        />
        <input
          name="following"
          value={form.following}
          onChange={handleChange}
          className="inputSmall"
          placeholder="Number of Following"
          type="number"
        />
        <input
          name="joined"
          value={form.joined}
          onChange={handleChange}
          className="inputSmall"
          placeholder="Joined Date (Month/Year)"
        />
      </div>

      <div className="flagsGrid">
        <label className="check">
          <input
            type="checkbox"
            name="flag_isPrivate"
            checked={form.flags.isPrivate}
            onChange={handleChange}
          />{" "}
          is private
        </label>
        <label className="check">
          <input
            type="checkbox"
            name="flag_hasChannel"
            checked={form.flags.hasChannel}
            onChange={handleChange}
          />{" "}
          has channel
        </label>
        <label className="check">
          <input
            type="checkbox"
            name="flag_isBusiness"
            checked={form.flags.isBusiness}
            onChange={handleChange}
          />{" "}
          is business account
        </label>
        <label className="check">
          <input
            type="checkbox"
            name="flag_hasGuides"
            checked={form.flags.hasGuides}
            onChange={handleChange}
          />{" "}
          has guides
        </label>
        <label className="check">
          <input
            type="checkbox"
            name="flag_hasUrl"
            checked={form.flags.hasUrl}
            onChange={handleChange}
          />{" "}
          has external url
        </label>
      </div>

      <div className="actions">
        <button className="submit">Check Account</button>
      </div>
    </form>
  );
}
