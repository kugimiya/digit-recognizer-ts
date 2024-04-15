import { readFileSync, writeFileSync } from 'node:fs';
import path from "node:path";
import { Network } from "./lib/Network";

// Инит тредов для параллельной работы
// const worker_farm = require('worker-farm');
// const farm = worker_farm({ maxConcurrentCallsPerWorker: 1, autoStart: false }, path.resolve(__dirname, 'lib/Worker'));

const getShuffledArr = <T>(arr: T[]): T[] => {
  const newArr = arr.slice(0);

  for (let i = newArr.length - 1; i > 0; i--) {
    const rand = Math.floor(Math.random() * (i + 1));
    [newArr[i], newArr[rand]] = [newArr[rand], newArr[i]];
  }

  return newArr;
};

const test_datasets_file_names = [
  'test_mnist_aa', 'test_mnist_ab', 'test_mnist_ac', 'test_mnist_ad', 'test_mnist_ae', 'test_mnist_af', 'test_mnist_ag'
];

const train_datasets_file_names = [
  'train_mnist_a0', 'train_mnist_aa', 'train_mnist_ab', 'train_mnist_ac', 'train_mnist_ad', 'train_mnist_ae', 'train_mnist_af', 'train_mnist_ag', 'train_mnist_ah', 'train_mnist_ai', 'train_mnist_aj', 'train_mnist_ak', 'train_mnist_al', 'train_mnist_am', 'train_mnist_an', 'train_mnist_ao', 'train_mnist_ap', 'train_mnist_aq', 'train_mnist_ar', 'train_mnist_as', 'train_mnist_at', 'train_mnist_au', 'train_mnist_av', 'train_mnist_aw', 'train_mnist_ax', 'train_mnist_ay', 'train_mnist_az', 'train_mnist_ba', 'train_mnist_bb', 'train_mnist_bc', 'train_mnist_bd', 'train_mnist_be', 'train_mnist_bf', 'train_mnist_bg', 'train_mnist_bh', 'train_mnist_bi', 'train_mnist_bj', 'train_mnist_bk', 'train_mnist_bl', 'train_mnist_bm', 'train_mnist_bn'
];


const load_datasets = (filenames: string[]) => filenames.map((file_name) => readFileSync(path.resolve(__dirname, '..', 'datasets', file_name))
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
  }))
).reduce((acc, cur) => [...acc, ...cur],[]);

const train_dataset = load_datasets(train_datasets_file_names);
const test_dataset = load_datasets(test_datasets_file_names);

function run_dataset(origin_network: Network, dataset: { digit: number, input: number[], output: number[] }[], learn_rate: number) {
  let errors: number[] = [];
  let common_error = 0;
  let network = origin_network.clone();

  let false_predictions = 0;
  let true_predictions = 0;

  dataset.forEach(({ input, output, digit }) => {
    network.feed_forward(input);
    const error = network.get_error(output);
    common_error = common_error + error;
    errors.push(error);

    const result = network.get_result_softed().map((v) => v > 0.5 ? 1 as number : 0 as number);

    if (result[digit] === 1 && result.reduce((a, c) => a + c, 0) === 1) {
      true_predictions += 1;
    } else {
      false_predictions += 1;
    }

    network.back_propagation(learn_rate, output);
  });

  common_error = common_error / errors.length;

  return { common_error, errors, network, false_predictions, true_predictions };
}

let network = new Network([784, 784 / 2, 784 / 4, 784 / 8, 784 / 16, 10]);
network.randomize();

let epoch_count = 100;
let batch_size = 100;
let learn_rate = 0.001;
let nice_ratio = 0.955;
let prev_ratio = 0;

try {
  const trained = JSON.parse( readFileSync(path.resolve(__dirname, '..', 'weights.json')).toString() ) as Network;
  network.layers = trained.layers;
} catch (e) {
  console.log('Looks like first run :thinking:');
}

while (prev_ratio < nice_ratio) {
  console.log('Start training');

  for (let i = 0; i < epoch_count; i++) {
    console.log(`Run epoch: ${i + 1} of ${epoch_count}, with batch_size=${batch_size} ...`);
    const result = run_dataset(network, getShuffledArr(train_dataset).slice(0, batch_size), learn_rate);
    network = result.network;
    console.log(`... completed, with avg_error=${result.common_error}, true_pred=${result.true_predictions} / ${batch_size}\n`);
  }

  writeFileSync(path.resolve(__dirname, '..', 'weights.json'), JSON.stringify(network.clone().clear()));

  console.log('Run test dataset');

  let true_predictions = 0;
  test_dataset.forEach(({ input, digit }) => {
    network.feed_forward(input);
    const result = network.get_result_softed().map((v) => v > 0.5 ? 1 as number : 0 as number);
    if (result[digit] === 1 && result.reduce((a, c) => a + c, 0) === 1) {
      true_predictions += 1;
    }
  });

  prev_ratio = true_predictions / test_dataset.length;
  console.log(`Test done! True predictions: ${true_predictions} of ${test_dataset.length} ratio=${prev_ratio}`);
  if (prev_ratio < nice_ratio) {
    console.log(`Ratio (${prev_ratio} is not nice (${nice_ratio}), start training again...`);
  } else {
    console.log(`Ratio is good, terminate training`);
  }
}
