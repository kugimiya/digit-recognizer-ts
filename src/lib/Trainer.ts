import { readFileSync } from "node:fs";
import { deep_clone, Layer, Network } from "./Network";
import { Workers } from "worker-farm";

export type TrainSet = {
  input: number[],
  output: number[],
};

export type DataSet = { digit: number, inputs: number[] }[];

export class Trainer {
  dataset: DataSet = [];
  farm: Workers;

  constructor(path_to_set: string, farm: Workers) {
    this.farm = farm;
    const dataset_raw = readFileSync(path_to_set).toString();
    dataset_raw.split('\n').forEach((line, i) => {
      if (i === 0) {
        return;
      }

      const [digit, ...inputs] = JSON.parse(`[${line}]`);

      if (digit === undefined) {
        return;
      }

      this.dataset.push({ digit, inputs });
    });
  }

  async train_network(epoch_count: number, generations_count: number, origin_network: Network): Promise<Network> {
    console.log('INFO: training network');

    const train_sets = this.dataset.map(({ digit, inputs }) => {
      return {
        input: inputs,
        output: [
          digit === 0 ? 1 : 0,
          digit === 1 ? 1 : 0,
          digit === 2 ? 1 : 0,
          digit === 3 ? 1 : 0,
          digit === 4 ? 1 : 0,
          digit === 5 ? 1 : 0,
          digit === 6 ? 1 : 0,
          digit === 7 ? 1 : 0,
          digit === 8 ? 1 : 0,
          digit === 9 ? 1 : 0,
        ],
      }
    });

    let cloned = Network.clone(origin_network.layers);
    let coolest_error = Number.MAX_SAFE_INTEGER;

    for (let i = 0; i < epoch_count; i++) {
      console.log(`  INFO: run iteration ${i}`);

      for (let j = 0; j < train_sets.length; j++) {
        console.log(`    INFO: run train set ${j}`);
        const train_set = train_sets[j];
        let mutations: { network: Network, error: number }[] = [];
        let promises: Promise<number>[] = [];

        for (let x = 0; x < generations_count; x++) {
          promises.push(new Promise(resolve => {
            this.farm(
              { network: cloned, train_set, i, j, x },
              (_err: null, { network, error }: { network: Network, error: number }) => {
                mutations.push({ network, error });
                resolve(error);
              }
            );
          }));
        }

        await Promise.all(promises);

        let coolest = mutations.sort((a, b) => a.error - b.error).at(0);
        if (coolest && coolest_error > coolest.error && coolest.error !== 0) {
          cloned.layers = coolest.network.layers;
          coolest_error = coolest.error;
        }

        console.log(`    INFO: run train set ${j} ends, error value: ${coolest_error}`);
      }
    }

    return cloned;
  }
}
