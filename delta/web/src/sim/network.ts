// The static description of a city: signalised intersections, the traffic
// movements that pass through each one, and the demand on those movements.
// No optimisation and no delay maths live here, only the structure everything
// else is built on. This is a direct port of delta/network.py.
//
// Vocabulary:
//   intersection  a junction controlled by a signal; its timing is what we tune
//   phase         a set of movements that get green together (they do not conflict)
//   cycle length  C, the time for one full pass through all phases (seconds)
//   green split   the fraction of the cycle a phase is green; splits sum to 1
//   offset        how far a junction's cycle is shifted vs a global clock
//   saturation s  the discharge rate of a movement on green (veh/s)
//   demand q      how fast cars arrive at a movement (veh/s)
//
// The ratio x = q / (s * g) is a movement's degree of saturation. Below 1 the
// movement clears its queue every cycle; as it approaches 1 the queue stops
// clearing and delay blows up. The whole Delta finding turns on this number.

import { rngFrom, uniform } from './rng'

export interface Movement {
  name: string
  q: number // arrival rate, veh/s
  s: number // saturation flow, veh/s
}

export interface Phase {
  name: string
  movements: Movement[]
}

export interface Intersection {
  name: string
  phases: Phase[]
  cycleLength: number // seconds
  minGreen: number // seconds, the safety floor per phase
  row: number
  col: number
}

export interface RoadNetwork {
  intersections: Intersection[]
  corridors: number[][] // each is a list of intersection indices forming an arterial
  freeFlowTravelTime: number // seconds between adjacent junctions, sets the ideal offset
  width: number
  height: number
}

export function totalArrival(phase: Phase): number {
  return phase.movements.reduce((sum, m) => sum + m.q, 0)
}

// The critical q/s of a phase: the share of the cycle it needs just to keep up
// with its busiest movement.
export function maxFlowRatio(phase: Phase): number {
  return phase.movements.reduce((best, m) => Math.max(best, m.q / m.s), 0)
}

export interface CityOptions {
  nIntersections?: number
  seed?: number
  load?: number // target mean degree of saturation; 0.55 is comfortably undersaturated
}

// Build a small, reproducible grid city as a stand-in for Delta's. The real
// Delta network had about 341 signals; this stays small enough to optimise in
// the browser in a fraction of a second while exercising every term in the
// delay model.
export function buildCity(opts: CityOptions = {}): RoadNetwork {
  const n = opts.nIntersections ?? 36
  const seed = opts.seed ?? 0
  const load = opts.load ?? 0.55
  const rand = rngFrom(seed)

  const width = Math.max(1, Math.round(Math.sqrt(n)))
  const equalGreen = 0.5 // a 2-phase signal at an equal split
  const sat = 0.5 // veh/s per movement, about 1800 veh/h, a standard lane

  const intersections: Intersection[] = []
  for (let k = 0; k < n; k++) {
    const phases: Phase[] = []
    for (const side of ['main', 'cross'] as const) {
      // Demand chosen so the main street's degree of saturation at an equal
      // split equals `load`: x = q / (s * g) = load, so q = load * s * g.
      // Cross streets carry less than the main arterial.
      const base = load * sat * equalGreen * (side === 'main' ? 1 : 0.6)
      const movements: Movement[] = (['fwd', 'rev'] as const).map((d) => ({
        name: `int${k}-${side}-${d}`,
        q: Math.max(0, base * uniform(rand, 0.7, 1.3)),
        s: sat,
      }))
      phases.push({ name: `int${k}-${side}`, movements })
    }
    intersections.push({
      name: `int${k}`,
      phases,
      cycleLength: 90,
      minGreen: 7,
      row: Math.floor(k / width),
      col: k % width,
    })
  }

  // Each grid row is a coordinated arterial corridor, a green-wave candidate.
  const corridors: number[][] = []
  for (let start = 0; start < n; start += width) {
    const corridor: number[] = []
    for (let j = start; j < Math.min(start + width, n); j++) corridor.push(j)
    if (corridor.length >= 2) corridors.push(corridor)
  }

  const height = Math.ceil(n / width)
  return { intersections, corridors, freeFlowTravelTime: 25, width, height }
}

// Mean degree of saturation under an equal-split plan: the single number that
// explains the ceiling. Well below 1 means almost no delay to optimise away.
export function meanSaturation(net: RoadNetwork): number {
  let sum = 0
  let count = 0
  for (const inter of net.intersections) {
    const g = 1 / inter.phases.length
    for (const phase of inter.phases) {
      for (const mv of phase.movements) {
        if (mv.q > 0) {
          sum += mv.q / (mv.s * g)
          count++
        }
      }
    }
  }
  return count ? sum / count : 0
}
