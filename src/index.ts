import { appendFileSync, readFileSync, writeFileSync } from 'node:fs';
import path from "node:path";
import { ETP_Params, Network } from "./lib/Network";
import { ETP } from "etp-ts";
import { cpus } from "node:os";

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

const mnist_train_set = [
  'train_mnist_a0', 'train_mnist_aa', 'train_mnist_ab', 'train_mnist_ac', 'train_mnist_ad', 'train_mnist_ae', 'train_mnist_af', 'train_mnist_ag', 'train_mnist_ah', 'train_mnist_ai', 'train_mnist_aj', 'train_mnist_ak', 'train_mnist_al', 'train_mnist_am', 'train_mnist_an', 'train_mnist_ao', 'train_mnist_ap', 'train_mnist_aq', 'train_mnist_ar', 'train_mnist_as', 'train_mnist_at', 'train_mnist_au', 'train_mnist_av', 'train_mnist_aw', 'train_mnist_ax', 'train_mnist_ay', 'train_mnist_az', 'train_mnist_ba', 'train_mnist_bb', 'train_mnist_bc', 'train_mnist_bd', 'train_mnist_be', 'train_mnist_bf', 'train_mnist_bg', 'train_mnist_bh', 'train_mnist_bi', 'train_mnist_bj', 'train_mnist_bk', 'train_mnist_bl', 'train_mnist_bm', 'train_mnist_bn'
];

const dirty_train_set = [
  'dataset_v2_aa', 'dataset_v2_ab', 'dataset_v2_ac', 'dataset_v2_ad', 'dataset_v2_ae', 'dataset_v2_af', 'dataset_v2_ag', 'dataset_v2_ah', 'dataset_v2_ai', 'dataset_v2_aj', 'dataset_v2_ak', 'dataset_v2_al', 'dataset_v2_am', 'dataset_v2_an', 'dataset_v2_ao', 'dataset_v2_ap', 'dataset_v2_aq', 'dataset_v2_ar', 'dataset_v2_as', 'dataset_v2_at', 'dataset_v2_au', 'dataset_v2_av', 'dataset_v2_aw', 'dataset_v2_ax', 'dataset_v2_ay', 'dataset_v2_az', 'dataset_v2_ba', 'dataset_v2_bb', 'dataset_v2_bc', 'dataset_v2_bd', 'dataset_v2_be', 'dataset_v2_bf', 'dataset_v2_bg', 'dataset_v2_bh', 'dataset_v2_bi', 'dataset_v2_bj', 'dataset_v2_bk', 'dataset_v2_bl', 'dataset_v2_bm', 'dataset_v2_bn', 'dataset_v2_bo', 'dataset_v2_bp', 'dataset_v2_bq', 'dataset_v2_br', 'dataset_v2_bs', 'dataset_v2_bt', 'dataset_v2_bu', 'dataset_v2_bv', 'dataset_v2_bw', 'dataset_v2_bx', 'dataset_v2_by', 'dataset_v2_bz', 'dataset_v2_ca', 'dataset_v2_cb', 'dataset_v2_cc', 'dataset_v2_cd', 'dataset_v2_ce', 'dataset_v2_cf', 'dataset_v2_cg', 'dataset_v2_ch', 'dataset_v2_ci', 'dataset_v2_cj', 'dataset_v2_ck', 'dataset_v2_cl', 'dataset_v2_cm', 'dataset_v2_cn', 'dataset_v2_co', 'dataset_v2_cp', 'dataset_v2_cq', 'dataset_v2_cr', 'dataset_v2_cs', 'dataset_v2_ct', 'dataset_v2_cu', 'dataset_v2_cv', 'dataset_v2_cw', 'dataset_v2_cx', 'dataset_v2_cy', 'dataset_v2_cz', 'dataset_v2_da', 'dataset_v2_db', 'dataset_v2_dc', 'dataset_v2_dd', 'dataset_v2_de', 'dataset_v2_df', 'dataset_v2_dg', 'dataset_v2_dh', 'dataset_v2_di', 'dataset_v2_dj', 'dataset_v2_dk', 'dataset_v2_dl', 'dataset_v2_dm', 'dataset_v2_dn', 'dataset_v2_do', 'dataset_v2_dp', 'dataset_v2_dq', 'dataset_v2_dr', 'dataset_v2_ds', 'dataset_v2_dt', 'dataset_v2_du', 'dataset_v2_dv', 'dataset_v2_dw', 'dataset_v2_dx', 'dataset_v2_dy', 'dataset_v2_dz', 'dataset_v2_ea', 'dataset_v2_eb', 'dataset_v2_ec', 'dataset_v2_ed', 'dataset_v2_ee', 'dataset_v2_ef', 'dataset_v2_eg', 'dataset_v2_eh', 'dataset_v2_ei', 'dataset_v2_ej', 'dataset_v2_ek', 'dataset_v2_el', 'dataset_v2_em', 'dataset_v2_en', 'dataset_v2_eo', 'dataset_v2_ep', 'dataset_v2_eq', 'dataset_v2_er', 'dataset_v2_es', 'dataset_v2_et', 'dataset_v2_eu', 'dataset_v2_ev', 'dataset_v2_ew', 'dataset_v2_ex', 'dataset_v2_ey', 'dataset_v2_ez', 'dataset_v2_fa', 'dataset_v2_fb', 'dataset_v2_fc', 'dataset_v2_fd', 'dataset_v2_fe', 'dataset_v2_ff', 'dataset_v2_fg', 'dataset_v2_fh', 'dataset_v2_fi', 'dataset_v2_fj', 'dataset_v2_fk', 'dataset_v2_fl', 'dataset_v2_fm', 'dataset_v2_fn', 'dataset_v2_fo', 'dataset_v2_fp', 'dataset_v2_fq', 'dataset_v2_fr', 'dataset_v2_fs', 'dataset_v2_ft', 'dataset_v2_fu', 'dataset_v2_fv', 'dataset_v2_fw', 'dataset_v2_fx', 'dataset_v2_fy', 'dataset_v2_fz', 'dataset_v2_ga', 'dataset_v2_gb', 'dataset_v2_gc', 'dataset_v2_gd', 'dataset_v2_ge', 'dataset_v2_gf', 'dataset_v2_gg', 'dataset_v2_gh', 'dataset_v2_gi', 'dataset_v2_gj', 'dataset_v2_gk', 'dataset_v2_gl', 'dataset_v2_gm',
];

const train_datasets_file_names = [
  ...mnist_train_set,
  // ...dirty_train_set,
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

console.log('Load datasets...');

const train_dataset = load_datasets(train_datasets_file_names);
const test_dataset = load_datasets(test_datasets_file_names);

async function main([ action_type ]: ETP_Params) {
  console.log(action_type);
  return 0;
}

const etp = new ETP(cpus().length, main);

async function run_dataset(origin_network: Network, dataset: { digit: number, input: number[], output: number[] }[], learn_rate: number, true_predicate: number) {
  let time = {
    run: Date.now(),
    run_delta: 0,
    train: Date.now(),
    train_delta: 0,
  };

  let errors: number[] = [];
  let common_error = 0;
  let network = origin_network.clone();

  let false_predictions = 0;
  let true_predictions = 0;

  for (let { input, output, digit } of dataset) {
    time.run = Date.now();
    await network.feed_forward(input);
    time.run_delta += Date.now() - time.run;

    const error = network.get_error(output);
    common_error = common_error + error;
    errors.push(error);

    const result = network.get_result_softed().map((v) => v > true_predicate ? 1 as number : 0 as number);

    if (result[digit] === 1 && result.reduce((a, c) => a + c, 0) === 1) {
      true_predictions += 1;
    } else {
      false_predictions += 1;
    }

    time.train = Date.now();
    network.back_propagation(learn_rate, output, etp);
    time.train_delta += Date.now() - time.train;
  }

  time.run_delta = time.run_delta / dataset.length;
  time.train_delta = time.train_delta / dataset.length;

  common_error = common_error / errors.length;

  return { common_error, errors, network, false_predictions, true_predictions,
    run_delta: time.run_delta,
    train_delta: time.train_delta,
  };
}

console.log('Load/create network...');

let net_conf = [784, 256, 64, 10];
let network = new Network(net_conf);
network.randomize();

let epoch_count = 100;
let batch_size = 100;
let learn_rate = 0.001;
let nice_ratio = 0.985;
let prev_ratio = 0;
let true_predicate = 0.5;

try {
  network.asJSON = readFileSync(path.resolve(__dirname, '..', `weights_${net_conf.join('_')}.json`)).toString();
} catch (e) {
  console.log(`Failed loading "weights_${net_conf.join('_')}.json". Looks like first run :thinking:`);
}

let train_count = 0;
let test_count = 0;
let train_stat_name = `${Date.now()}_stat_train_${net_conf.join('_')}.csv`;
let test_stat_name = `${Date.now()}_stat_test_${net_conf.join('_')}.csv`;

writeFileSync(train_stat_name, 'run,err,ratio\n');
writeFileSync(test_stat_name, 'run,ratio\n');

const runner = async() => {
  await etp.init();

  while (prev_ratio < nice_ratio) {
    console.log('Start training');

    for (let i = 0; i < epoch_count; i++) {
      // batch_size = Math.round(train_dataset.length / epoch_count * (i + 1));
      console.log(`Run epoch: ${i + 1} of ${epoch_count}, with batch_size=${batch_size} ...`);
      const result = await run_dataset(network, getShuffledArr(train_dataset).slice(0, batch_size), learn_rate, true_predicate);
      network = result.network;
      console.log(`sample train time avg: ${result.train_delta}ms, sample run time avg: ${result.run_delta}ms`);
      console.log(`... completed, with avg_error=${result.common_error}, ratio: ${result.true_predictions / batch_size}, true_pred=${result.true_predictions} / ${batch_size}\n`);

      train_count += 1;
      appendFileSync(train_stat_name, `${train_count},${result.common_error},${result.true_predictions / batch_size}\n`);
    }

    writeFileSync(path.resolve(__dirname, '..', `weights_${net_conf.join('_')}.json`), network.asJSON);

    console.log('Run test dataset');

    let true_predictions = 0;
    test_dataset.forEach(({ input, digit }) => {
      network.feed_forward(input);
      const result = network.get_result_softed().map((v) => v > true_predicate ? 1 as number : 0 as number);
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

    test_count += 1;
    appendFileSync(test_stat_name, `${test_count},${prev_ratio}\n`);
  }
}

runner();
