import { readFileSync } from 'node:fs';
import path from "node:path";
import { Network } from "./lib/Network";

// Инит тредов для параллельной работы
// const worker_farm = require('worker-farm');
// const farm = worker_farm({ maxConcurrentCallsPerWorker: 1, autoStart: false }, path.resolve(__dirname, 'lib/Worker'));

const dataset = readFileSync(path.resolve(__dirname, '..', 'mnist_digits_dataset.csv'))
  .toString()
  .split('\n')
  .slice(2, -2)
  .map((line) => JSON.parse(`[${line}]`) as Array<number>)
  .map(([digit, ...array]) => ({
    digit,
    input: array.map(v => v / 255),
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
    ]
  }));

function run_dataset(origin_network: Network, dataset: { digit: number, input: number[], output: number[] }[]) {
  let errors: number[] = [];
  let networks: Network[] = [];
  let common_error = 0;

  dataset.forEach(({ input, output }) => {
    const network = origin_network.clone();
    network.run(input);
    const error = network.get_error(output);
    common_error = common_error + error;
    networks.push(network);
    errors.push(error);
  });

  common_error = common_error / errors.length;

  return { common_error, errors, networks };
}

let network = new Network([784, 16, 16, 10]);
network.randomize();
const result = run_dataset(network, dataset);

console.log(result.common_error);
