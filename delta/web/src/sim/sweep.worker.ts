import { flatCeilingSweep } from "./analysis";
self.onmessage = () => {
  try {
    for (const load of [0.4, 0.55, 0.7, 0.85, 0.95, 1.1, 1.25]) {
      self.postMessage({
        row: flatCeilingSweep([load], { iters: 60, seed: 0 })[0],
      });
    }
    self.postMessage({ done: true });
  } catch (error) {
    self.postMessage({
      error: error instanceof Error ? error.message : "Calculation failed.",
    });
  }
};
