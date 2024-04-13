import { Network } from "./Network";

module.exports = function (
  data: { network: Network, train_set: { input: number[], output: number[] }, i: number, j: number, x: number },
  callback: (err: Error | null, data: { network: Network, error: number }) => void
) {
  console.log(`      INFO: mutate generation ${data.i}.${data.j}.${data.x}`);
  // console.log(data);
  let next_data = {
    network: Network.mutate(data.network.layers),
    error: 0,
  };

  console.log(`      INFO: run generation ${data.i}.${data.j}.${data.x}`);
  let state = Network.run(next_data.network.layers, data.train_set.input);
  let error_sum = state.layers.at(-1)?.neurons_out_values
    .map((value, index) => Math.abs(data.train_set.output[index] - value))
    .reduce((acc, cur) => acc + cur) || 0;

  next_data.network = state;
  next_data.error = error_sum;

  callback(null, next_data);
}
