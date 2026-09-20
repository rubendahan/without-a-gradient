import test from 'node:test';
import assert from 'node:assert/strict';
import { createPSO, createGA, createDE, createSA, createCMAES } from '../web/js/algos.js';
import { FUNCS } from '../web/js/landscape.js';

const methods = [
  ['PSO', createPSO, {n:22,w:.72,c1:1.5,c2:1.5}],
  ['GA', createGA, {n:36,mut:.08,k:3}],
  ['DE', createDE, {n:28,F:.7,CR:.9}],
  ['SA', createSA, {T0:4,cool:.985}],
  ['CMA-ES', createCMAES, {n:12,sig:.3}],
];
for (const [name, factory, params] of methods) {
  test(`${name}: deterministic, finite, non-increasing incumbent on every landscape`, () => {
    for (const objective of Object.values(FUNCS)) {
      const first = factory(objective, {...params, seed:42});
      const replay = factory(objective, {...params, seed:42});
      let best = first.best;
      for (let i=0;i<100;i++) {
        first.step(); replay.step();
        assert.ok(Number.isFinite(first.best), objective.label);
        assert.ok(first.best <= best + 1e-10, `${objective.label}: best-so-far worsened`);
        assert.equal(first.best,replay.best, `${objective.label}: seeded replay diverged`);
        best=first.best;
      }
    }
  });
}
test('DE: crossover always includes a donor coordinate, including CR=0', () => {
  const calls=[];
  const objective={...FUNCS.sphere,f:(x,y)=>{calls.push([x,y]);return x*x+y*y;}};
  const algorithm=createDE(objective,{n:12,F:.7,CR:0,seed:42});
  const population=calls.slice();
  algorithm.step();
  assert.equal(calls.length,24);
  for(let i=0;i<12;i++) assert.notDeepEqual(calls[12+i],population[i],`Trial ${i} copied its target unchanged`);
});
test('SA: one iteration makes exactly one objective evaluation and records its proposal', () => {
  let calls=0;
  const objective={...FUNCS.sphere,f:(x,y)=>{calls++;return x*x+y*y;}};
  const algorithm=createSA(objective,{T0:4,cool:.985,seed:42});
  assert.equal(calls,1); algorithm.step(); assert.equal(calls,2);
  assert.equal(algorithm.iter,1); assert.equal(algorithm.frame().links.length,1);
});
test('PSO: operation detail uses the actual highlighted particle transition', () => {
  const algorithm=createPSO(FUNCS.sphere,{n:22,w:.72,c1:1.5,c2:1.5,seed:42});
  const before=algorithm.frame().dots[0]; algorithm.step();
  const after=algorithm.frame();
  assert.equal(after.operation.x,before.x); assert.equal(after.operation.y,before.y);
  assert.equal(after.operation.nx,after.dots[0].x); assert.equal(after.operation.ny,after.dots[0].y);
});
import { descend, derivative, objective, pointAt, STARTS, STEP_SIZE, MINIMUM } from '../web/js/descent.js';

test('Local descent: every displayed iterate follows the fixed gradient update', () => {
  for (const start of STARTS) {
    const run = descend(start);
    assert.equal(run.path[0], start);
    assert.ok(run.converged);
    for (let i=1;i<run.path.length;i++) {
      assert.ok(Math.abs(run.path[i] - (run.path[i-1] - STEP_SIZE * derivative(run.path[i-1]))) < 1e-12);
      assert.ok(objective(run.path[i]) <= objective(run.path[i-1]) + 1e-10);
    }
    assert.ok(Math.abs(derivative(run.rest)) < 1e-4);
  }
});
test('Local descent: multiple basins and interpolation preserve the actual endpoints', () => {
  const runs = STARTS.map(descend);
  assert.ok(new Set(runs.map(r=>r.rest.toFixed(2))).size > 3);
  const global = runs.filter(r=>Math.abs(r.rest-MINIMUM.x)<.02);
  assert.ok(global.length>0 && global.length<runs.length);
  for(const run of runs) {
    assert.equal(pointAt(run,0),run.start);
    assert.equal(pointAt(run,1000),run.rest);
    assert.equal(pointAt(run,.5),(run.path[0]+run.path[1])/2);
  }
});
