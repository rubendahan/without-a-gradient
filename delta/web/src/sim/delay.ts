// The delay model: a transparent stand-in for Delta's mesoscopic simulator.
//
// THIS IS A PROXY, NOT THE REAL SIMULATOR. The real Delta objective was a
// proprietary mesoscopic traffic simulator that ran thousands of vehicles over
// four hours and returned one number, the total vehicle delay. We do not have
// it, so this is a classic Webster / HCM-style intersection delay model with
// the same one-vector-in, one-scalar-out interface. Port of delta/simulator.py.
//
// Per movement the delay has two physically meaningful parts:
//   1. Uniform delay: the unavoidable wait on red under steady demand.
//   2. Overflow delay: the HCM incremental term, near zero while below capacity
//      and exploding as the degree of saturation x approaches and passes 1.
// On top of that, a corridor coordination term rewards offsets that form a good
// green wave and penalises badly coordinated ones, which is what makes offsets
// matter at all.

import type { RoadNetwork } from './network'
import { totalArrival } from './network'
import type { IntersectionTiming, SignalPlan } from './plan'

const ANALYSIS_PERIOD_H = 4.0 // the Delta horizon
const OVERFLOW_K = 0.5 // HCM incremental-delay constant for fixed-time signals

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))
// Python-style modulo that is always non-negative for a positive divisor.
const pmod = (a: number, b: number) => ((a % b) + b) % b

// Total delay for one movement in vehicle-seconds, summed over every vehicle
// arriving in the horizon, so plans are compared on the same scale.
export function movementDelay(q: number, s: number, g: number, C: number): number {
  if (q <= 0) return 0
  g = clamp(g, 1e-6, 1)
  const capacity = s * g
  const x = q / capacity

  // Uniform delay per vehicle. min(x, 1) in the denominator keeps it finite at
  // and beyond saturation; the overflow term then carries the queue growth.
  const xEff = Math.min(x, 1)
  const denom = Math.max(1 - xEff * g, 1e-6)
  const dUniform = (0.5 * C * (1 - g) ** 2) / denom

  // Overflow / random delay per vehicle, HCM incremental form.
  const T = ANALYSIS_PERIOD_H
  const cPerH = capacity * 3600
  const root = (x - 1) ** 2 + (8 * OVERFLOW_K * x) / (cPerH * T)
  const dOverflow = 900 * T * (x - 1 + Math.sqrt(root))

  const perVehicle = Math.max(dUniform + dOverflow, 0)
  const nVehicles = q * (T * 3600)
  return perVehicle * nVehicles
}

// Green-wave reward and penalty summed over corridor links, in vehicle-seconds.
// For each adjacent pair we compare the actual offset difference to the ideal
// one (the free-flow travel time). A near-ideal offset earns a rebate, a
// worst-case one a penalty, scaled by the platoon carried on the link.
export function coordinationDelay(net: RoadNetwork, timings: IntersectionTiming[]): number {
  if (net.corridors.length === 0) return 0
  const ideal = net.freeFlowTravelTime
  let total = 0
  for (const corridor of net.corridors) {
    for (let i = 0; i + 1 < corridor.length; i++) {
      const a = corridor[i]
      const b = corridor[i + 1]
      const ta = timings[a]
      const tb = timings[b]
      const C = 0.5 * (ta.cycleLength + tb.cycleLength)
      let err = pmod(tb.offset - ta.offset - ideal, C)
      if (err > C / 2) err -= C
      const factor = (Math.abs(err) / (C / 2)) * 2 - 1
      const platoon = totalArrival(net.intersections[a].phases[0]) * 3600
      total += factor * platoon * 0.5
    }
  }
  return total
}

// The objective handed to the optimiser: one decision vector in, one scalar
// (total vehicle delay) out. Lower is better.
export function evaluate(
  net: RoadNetwork,
  plan: SignalPlan,
  x: number[],
  coordinationWeight = 1,
): number {
  const timings = plan.fromVector(x)
  let total = 0
  net.intersections.forEach((inter, k) => {
    const t = timings[k]
    inter.phases.forEach((phase, p) => {
      const g = t.splits[p]
      for (const mv of phase.movements) total += movementDelay(mv.q, mv.s, g, t.cycleLength)
    })
  })
  total += coordinationWeight * coordinationDelay(net, timings)
  return total
}

export interface IntersectionStat {
  delay: number // total delay at this junction, veh-s
  sat: number // worst movement degree of saturation under the current splits
}

// Per-intersection delay and worst-case saturation, used to colour the city map.
export function intersectionStats(
  net: RoadNetwork,
  plan: SignalPlan,
  x: number[],
): IntersectionStat[] {
  const timings = plan.fromVector(x)
  return net.intersections.map((inter, k) => {
    const t = timings[k]
    let delay = 0
    let sat = 0
    inter.phases.forEach((phase, p) => {
      const g = t.splits[p]
      for (const mv of phase.movements) {
        delay += movementDelay(mv.q, mv.s, g, t.cycleLength)
        if (mv.q > 0) sat = Math.max(sat, mv.q / (mv.s * g))
      }
    })
    return { delay, sat }
  })
}
