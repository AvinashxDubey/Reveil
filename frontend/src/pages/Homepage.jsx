import React, { useEffect } from "react";
import "../styles/Homepage.css";

import card1 from "../assets/card1.jpg";
import card2 from "../assets/card2.jpg";
import card3 from "../assets/card3.jpg";
import about1 from "../assets/about1.jpg";
import about2 from "../assets/about2.jpg";
import about3 from "../assets/about3.jpg";
import about4 from "../assets/about4.jpg";
import Navbar from "../components/Navbar";

function Homepage() {
  useEffect(() => {
    const waitForImages = (elements, timeout = 200000) =>
      new Promise((resolve) => {
        const imgs = Array.from(elements)
          .map((el) => Array.from(el.querySelectorAll("img")))
          .flat();
        if (!imgs.length) return resolve();

        let remaining = imgs.length;
        const done = () => {
          remaining -= 1;
          if (remaining <= 0) resolve();
        };

        imgs.forEach((img) => {
          if (img.complete && img.naturalWidth !== 0) {
            done();
          } else {
            img.addEventListener("load", done);
            img.addEventListener("error", done);
          }
        });

        setTimeout(resolve, timeout);
      });

const NAV_H = 70; // match navbar height in px
const obsOptions = {
  threshold: 0.15,
  root: null,
  rootMargin: `-${NAV_H}px 0px -8% 0px`
};

    const allSections = Array.from(document.querySelectorAll(".content-section"));

    let cardObserver = null;
    let sectionObserver = null;
    let textObserver = null;
    let aboutObserver = null;
    let aboutTimers = [];

    waitForImages(allSections, 2500).then(() => {
      cardObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("show");
          } else {
            entry.target.classList.remove("show");
          }
        });
      }, obsOptions);

      const cards = document.querySelectorAll(".card");
      cards.forEach((c) => cardObserver.observe(c));

      sectionObserver = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              document.body.style.backgroundColor = entry.target.dataset.bg || "";
              document.body.style.transition = "background-color 0.8s ease-in-out";
            }
          });
        },
        { threshold: 0.5 }
      );
      allSections.forEach((sec) => sectionObserver.observe(sec));

      textObserver = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            const textEl = entry.target.querySelector(".text");
            if (!textEl) return;

            if (entry.isIntersecting) {
              textEl.classList.add("in-view");
            } else {
              textEl.classList.remove("in-view");
            }
          });
        },
        { threshold: 0.15, root: null, rootMargin: "0px 0px -12% 0px" }
      );
      allSections.forEach((sec) => textObserver.observe(sec));

      const about = document.querySelector(".about-section");
      if (about) {
        const aboutCards = Array.from(about.querySelectorAll(".cards-row .card"));
        if (aboutCards.length) {
          aboutObserver = new IntersectionObserver(
            (entries) => {
              entries.forEach((entry) => {
                if (entry.isIntersecting) {
                  aboutCards.forEach((card, i) => {
                    const t = setTimeout(() => card.classList.add("show"), i * 120);
                    aboutTimers.push(t);
                  });
                } else {
                  aboutTimers.forEach((id) => clearTimeout(id));
                  aboutTimers = [];
                  aboutCards.forEach((card) => card.classList.remove("show"));
                }
              });
            },
            { threshold: 0.2, rootMargin: "0px 0px -10% 0px" }
          );

          aboutObserver.observe(about);
        }
      }
    });

    return () => {
      if (cardObserver) cardObserver.disconnect();
      if (sectionObserver) sectionObserver.disconnect();
      if (textObserver) textObserver.disconnect();
      if (aboutObserver) aboutObserver.disconnect();
      aboutTimers.forEach((id) => clearTimeout(id));
      aboutTimers = [];
    };
  }, []);

  return (
    <React.Fragment>
              
      <div className="Homepage">
        <Navbar/>
        <section className="content-section hero1" data-bg="rgba(12, 82, 73, 1)">
          <div className="text">
            <h1>Fake Social Media Detection</h1>
            <p>Discover fake profiles, posts, and content.</p>
          </div>
        </section>
      </div>

      <div>
        <section className="content-section about-section" data-bg="#2f525aff">
          <div className="text">
            <h2>About Us</h2>
            <p>
              We are a creative platform dedicated to showcasing authentic content and
              connecting people.
            </p>
          </div>

          <div className="cards-row">
            <div className="card">
              <img src={about1} alt="About 1" />
              <p className="card-name">Alice Johnson</p>
            </div>
            <div className="card">
              <img src={about2} alt="About 2" />
              <p className="card-name">David Smith</p>
            </div>
            <div className="card">
              <img src={about3} alt="About 3" />
              <p className="card-name">Sophia Lee</p>
            </div>
            <div className="card">
              <img src={about4} alt="About 4" />
              <p className="card-name">Michael Brown</p>
            </div>
          </div>
        </section>

        <section className="content-section reverse" data-bg="#1e3c72">
          <div className="card">
            <img className="card1" src={card1} alt="Detection 2" />
          </div>
          <div className="text">
            <h2>Profile Analysis</h2>
            <p>Check the authenticity of user profiles and their content quickly and reliably.</p>
          </div>
        </section>

        <section className="content-section" data-bg="#406750ff">
          <div className="card">
            <img className="card2" src={card2} alt="Detection 3" />
          </div>
          <div className="text">
            <h2>Post Verification</h2>
            <p>Detect fake posts, misleading information, and suspicious content with our AI algorithms.</p>
          </div>
        </section>

        <section className="content-section reverse" data-bg="#4e4376">
          <div className="card">
            <img className="card3" src={card3} alt="Detection 4" />
          </div>
          <div className="text">
            <h2>Real-Time Monitoring</h2>
            <p>Keep your social media feeds safe by identifying fake activity in real time.</p>
          </div>
        </section>
      </div>
    </React.Fragment>
  );
}

export default Homepage;
