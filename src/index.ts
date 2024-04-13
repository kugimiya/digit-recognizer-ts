import { rmSync, writeFileSync } from 'node:fs';
import { Trainer } from "./lib/Trainer";
import { Network } from "./lib/Network";
import path from "node:path";

const worker_farm = require('worker-farm');

const farm = worker_farm({ maxConcurrentCallsPerWorker: 1, autoStart: true }, path.resolve(__dirname, 'lib/Worker'));

const MAX_EPOCHS = 2;
const MAX_MUTATIONS_VARIANTS = 16;

console.log('INFO: reading dataset');
const trainer = new Trainer(path.resolve(__dirname, '../mnist_digits_dataset.csv'), farm);

console.log('INFO: making network');
const network = new Network();
network.make([784, 784 / 2, 784 / 4, 784 / 8, 784 / 16, 10]);

console.log('INFO: training network');
trainer.train_network(MAX_EPOCHS, MAX_MUTATIONS_VARIANTS, network).then(coolest_network => {
  try {
    rmSync(path.resolve(__dirname, '../trained_weights.json'));
  } catch {}

  try {
    writeFileSync(path.resolve(__dirname, '../trained_weights.json'), JSON.stringify(coolest_network.layers));
  } catch {}

  trainer.dataset.forEach(({ digit, inputs }) => {
    let runned = Network.run(coolest_network.layers, inputs);
    console.log(digit, runned.layers.at(-1)?.neurons_out_values.map(i => Math.round(i * 10000) / 10000));
  });
});
