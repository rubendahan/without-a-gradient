import TrafficLab from "./components/TrafficLab";

export default function App() {
  return (
    <>
      <a className="skip-link" href="#lab">
        Skip to the experiment
      </a>
      <div className="page-shell">
        <nav className="site-nav" aria-label="Main navigation">
          <a className="wordmark" href="../">
            without a gradient
          </a>
          <a href="../">Optimisation methods ↗</a>
        </nav>
        <header className="traffic-hero">
          <div>
            <p className="eyebrow">Delta 2026 / Interactive application</p>
            <h1>
              Traffic-signal
              <br />
              optimisation
            </h1>
          </div>
          <div className="hero-description">
            <p>
              Compare signal-timing strategies on a synthetic road network using
              an analytical delay model and particle swarm optimisation.
            </p>
            <p className="small-copy">
              Change the demand, run the swarm and compare the delay. This is a
              synthetic teaching model inspired by Delta 2026, not the
              competition simulator.
            </p>
            <a className="text-link" href="#lab">
              Try the experiment ↓
            </a>
          </div>
        </header>
        <main>
          <section id="lab" className="chapter">
            <div className="chapter-top">
              <span className="eyebrow">01 / Experiment</span>
              <span className="small-copy">
                36 junctions · 108 parameters · fixed seed
              </span>
            </div>
            <h2>Signal plans and estimated delay</h2>
            <p className="section-lead">
              Start with equal green times, compare a plan based on demand, then
              let the swarm search from that baseline. Lower delay is better.
            </p>
            <TrafficLab />
            <p className="model-caption">
              The cars illustrate the signal timings. The score comes from a
              separate analytical delay model, not from counting the animated
              cars. Neither the map nor the model includes turning traffic,
              spillback or clearance intervals.
            </p>
          </section>
          <section id="how" className="chapter">
            <span className="eyebrow">02 / Inside the model</span>
            <h2>Model assumptions and implementation</h2>
            <div className="model-grid">
              <article>
                <span className="model-number">A</span>
                <h3>The score</h3>
                <p>
                  A four-hour estimate of accumulated delay, in vehicle-seconds.
                  Each movement contributes a uniform red-light delay and an
                  incremental delay term. A separate, heuristic coordination
                  term rewards compatible offsets.
                </p>
                <p>
                  It follows a Webster / HCM-style formulation. It is not a
                  calibrated prediction for a real city.
                </p>
              </article>
              <article>
                <span className="model-number">B</span>
                <h3>The plan</h3>
                <p>
                  Every junction has two phases. The plan allocates green time
                  between them and chooses an offset within the cycle. Both
                  phases retain a minimum green time.
                </p>
                <p>
                  “Equal split” means each phase gets half the cycle. It does
                  not mean that conflicting movements are green together.
                </p>
              </article>
              <article>
                <span className="model-number">C</span>
                <h3>The search</h3>
                <p>
                  Three groups of 16 particles share their best plan. The search
                  starts with the demand-based baseline included, then runs for
                  60 generations: 2,928 objective evaluations including
                  initialisation.
                </p>
                <p>
                  The best result found is not a proof of optimality. The fixed
                  seed makes each run reproducible.
                </p>
              </article>
            </div>
          </section>
        </main>
        <footer className="site-footer">
          <a className="wordmark" href="../">
            without a gradient
          </a>
          <p>
            Inspired by the Delta 2026 traffic challenge.
            <br />
            Analytical delay model and reproducible optimisation experiments.
          </p>
          <a href="https://github.com/rubendahan/without-a-gradient/tree/main/delta">
            Model & source ↗
          </a>
        </footer>
      </div>
    </>
  );
}
