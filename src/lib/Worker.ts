const sigmoid = (x: number) => {
  return 1 / (1 + Math.exp(-x));
};

module.exports = function (
  data: [action: 'calc_layer_forward', prev_layer_activations: number[], prev_layer_weights: number[], curr_layer_biases: number[], prev_layer_size: number, j: number]
    | [action: 'calc_layer_backward'],
  callback: (...args: any) => void
) {
  if (data[0] === 'calc_layer_forward') {
    const [_action, prev_layer_activations, prev_layer_weights, curr_layer_biases, prev_layer_size, j] = data;
    let curr_layer_activation = 0;

    for (let k = 0; k < prev_layer_size; k++) {
      curr_layer_activation += prev_layer_activations[k] * prev_layer_weights[k];
    }

    curr_layer_activation += curr_layer_biases[j];
    curr_layer_activation = sigmoid(curr_layer_activation);
    callback(null, curr_layer_activation);
  }
}
