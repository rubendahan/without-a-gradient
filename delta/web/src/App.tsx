import TrafficLab from './components/TrafficLab'
import FlatCeilingChart from './components/FlatCeilingChart'

export default function App() {
  return (
    <div className="mx-auto max-w-5xl px-5 pb-24">
      <Hero />

      <Section
        id="lab"
        eyebrow="Try it"
        title="A city, and an optimiser you can run"
        lead="Drag the demand slider, then hit Optimise. The swarm starts from a sensible plan and tunes it. Keep an eye on the gain over the sane plan: at low demand it stays near zero no matter how long it runs."
      >
        <div className="rounded-2xl bg-[var(--color-card)] p-4 shadow-sm ring-1 ring-[var(--color-line)] sm:p-6">
          <TrafficLab />
        </div>
        <p className="mt-3 text-sm text-[var(--color-muted)]">
          Cars drive at a steady speed and stop at red lights, so a queue building up behind a
          signal is congestion you can see. The lights cycle from the current plan: each junction is
          green for the main street for a slice of its cycle, shifted by its offset. When you
          optimise, the green slices shift toward the busier streets and the offsets line up, so the
          queues drain and a platoon can catch a run of greens.
        </p>
      </Section>

      <Section
        id="finding"
        eyebrow="The finding"
        title="The optimiser only earns its keep near capacity"
        lead="We ran the swarm against the sane plan across a sweep of demand levels. Below the 1% line the optimiser buys essentially nothing: the network clears its queues every cycle and there is no delay left to remove. Push demand past capacity and tuning the lights finally pays off. The real Delta network lived on the left of this chart."
      >
        <div className="rounded-2xl bg-[var(--color-card)] p-4 shadow-sm ring-1 ring-[var(--color-line)] sm:p-6">
          <FlatCeilingChart />
        </div>
        <p className="mt-3 text-sm text-[var(--color-muted)]">
          Bars are the optimiser's improvement over the demand-proportional plan, computed live in
          your browser. Blue bars are a flat ceiling, red bars are where timing matters.
        </p>
      </Section>

      <Section id="how" eyebrow="How it works" title="What you are actually running">
        <div className="grid gap-6">
          <Block title="The objective is a black box">
            <p>
              Delta 2026 was an international applied-mathematics competition (sponsored by Mireo).
              The challenge was to retime a whole city's traffic
              signals so a fleet of vehicles loses as little total time as possible over four hours.
              You submit a full plan and a traffic simulator hands back a single number, the total
              delay. There is no gradient and no formula, and every query is slow, so the evaluation
              budget is the thing you have to spend wisely.
            </p>
            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Fact n="~341" l="signals to set" />
              <Fact n="5,215" l="vehicles" />
              <Fact n="4 h" l="simulated per query" />
              <Fact n="~5 s" l="wall-clock per query" />
            </div>
          </Block>

          <Block title="The model running on this page is a stand-in">
            <p>
              The real objective is Delta's mesoscopic simulator, which we do not have. So this page
              runs a transparent replacement: a Webster and HCM intersection-delay model, the
              textbook physics of how long cars wait at a light. Its delay stays low and flat while a
              movement clears its queue each cycle and rises sharply as the movement approaches
              capacity, which is the shape that produced the finding. The interface is the same, one
              number in and one number out, so swapping in the real engine changes nothing
              downstream.
            </p>
          </Block>

          <Block title="The method is a swarm with a warm start">
            <p>
              Three particle swarms with different temperaments share one global best: one explores
              widely, one balances, one refines. They start from the demand-proportional plan rather
              than from noise, because warm starting from a good heuristic is where most of the real
              improvement comes from, and they tune the green splits and offsets from there. The same
              optimisers, written in plain NumPy, sit in the{' '}
              <code className="rounded bg-[var(--color-paper)] px-1.5 py-0.5 text-[0.85em]">delta</code>{' '}
              Python package in this repository.
            </p>
          </Block>
        </div>
      </Section>

      <Footer />
    </div>
  )
}

function Hero() {
  return (
    <header className="pt-10 pb-12 sm:pt-14">
      <a
        href="../"
        className="mb-6 inline-flex items-center gap-1.5 text-sm text-[var(--color-muted)] transition hover:text-[var(--color-ink)]"
      >
        ← the methods explainer
      </a>
      <div className="mb-5 inline-flex items-center gap-2 rounded-full bg-[var(--color-accent-soft)] px-3 py-1 text-xs font-medium text-[var(--color-accent)]">
        Delta 2026 · sponsored by Mireo
      </div>
      <h1 className="text-4xl font-bold leading-[1.08] tracking-tight sm:text-5xl">
        Retiming a city's traffic lights,
        <br />
        <span className="text-[var(--color-muted)]">
          and learning the optimiser was barely worth it.
        </span>
      </h1>
      <p className="mt-6 max-w-2xl text-lg leading-relaxed text-[var(--color-muted)]">
        You submit a full signal plan, a simulator runs four hours of city traffic, and you get back
        one number: the total time everyone lost. No gradient, no formula, and every evaluation is
        slow. Here is the whole problem running in your browser, optimiser included.
      </p>
      <div className="mt-7 rounded-xl border-l-4 border-[var(--color-accent)] bg-[var(--color-card)] p-4 text-[0.95rem] leading-relaxed shadow-sm ring-1 ring-[var(--color-line)]">
        <span className="font-semibold">The twist.</span> For the demand we were handed, the network
        was so far from saturation that a sensible plan written in five minutes was within 1% of
        anything our optimisers found. The result worth keeping was understanding the objective, not
        beating it.
      </div>
    </header>
  )
}

function Section({
  id,
  eyebrow,
  title,
  lead,
  children,
}: {
  id: string
  eyebrow: string
  title: string
  lead?: string
  children: React.ReactNode
}) {
  return (
    <section id={id} className="mt-20 scroll-mt-8">
      <div className="text-sm font-semibold uppercase tracking-wide text-[var(--color-accent)]">
        {eyebrow}
      </div>
      <h2 className="mt-2 text-2xl font-bold tracking-tight sm:text-3xl">{title}</h2>
      {lead && <p className="mt-3 max-w-3xl leading-relaxed text-[var(--color-muted)]">{lead}</p>}
      <div className="mt-6">{children}</div>
    </section>
  )
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl bg-[var(--color-card)] p-5 shadow-sm ring-1 ring-[var(--color-line)] sm:p-6">
      <h3 className="text-lg font-semibold">{title}</h3>
      <div className="mt-2 max-w-3xl leading-relaxed text-[var(--color-muted)]">{children}</div>
    </div>
  )
}

function Fact({ n, l }: { n: string; l: string }) {
  return (
    <div className="rounded-xl bg-[var(--color-paper)] p-3 text-center">
      <div className="tnum text-xl font-bold text-[var(--color-ink)]">{n}</div>
      <div className="mt-0.5 text-xs text-[var(--color-muted)]">{l}</div>
    </div>
  )
}

function Footer() {
  return (
    <footer className="mt-20 border-t border-[var(--color-line)] pt-8 text-sm text-[var(--color-muted)]">
      <p>
        Built from the Delta 2026 traffic-optimisation challenge (Delta was sponsored by Mireo). The optimisers come
        from a small NumPy library; the delay model on this page is an honest stand-in for the real
        simulator, with the same query interface so the real engine drops in unchanged.
      </p>
      <p className="mt-3">Everything runs client-side. No backend, no tracking. MIT licensed.</p>
    </footer>
  )
}
