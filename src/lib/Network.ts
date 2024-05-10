import { ETP } from "etp-ts";

export type Layer = {
  size: number;
  biases: Float64Array;
  weights: Float64Array;
  activations: Float64Array;
}

export type ETP_Params = [
  action_type: 'gradient_calc' | 'delta_calc' | 'next_error_calc' | 'next_weights_calc',
  prev_layer_activations: Float64Array,
  prev_layer_size: number,
  current_layer_weights: Float64Array,
  current_layer_size: number,
  errors: Float64Array,
  gradients: Float64Array,
  deltas: Float64Array,
  next_errors: Float64Array
];

export const sigmoid = (x: number) => {
  return 1 / (1 + Math.exp(-x));
};

export const sigmoid_prime = (x: number) => {
  return x * (1 - x);
};

export class Network {
  layers: Layer[] = [];
  next_sizes: number[] = [];

  constructor(neurons_counts: number[]) {
    for (let i = 0; i < neurons_counts.length; i++) {
      let next_size = 0;

      if (i < neurons_counts.length - 1) {
        next_size = neurons_counts[i + 1];
      }

      this.layers[i] = {
        size: neurons_counts[i],
        activations: new Float64Array(new SharedArrayBuffer(Float64Array.BYTES_PER_ELEMENT * neurons_counts[i])).fill(0.0),
        biases: new Float64Array(new SharedArrayBuffer(Float64Array.BYTES_PER_ELEMENT * neurons_counts[i])).fill(0.0),
        weights: new Float64Array(new SharedArrayBuffer(Float64Array.BYTES_PER_ELEMENT * neurons_counts[i] * next_size)).fill(0.0),
      };

      this.next_sizes[i] = next_size;
    }
  }

  randomize() {
    for (let i = 0; i < this.layers.length; i++) {
      let nextSize = 0;

      if (i < this.layers.length - 1) {
        nextSize = this.layers[i + 1].size;
      }

      for (let j = 0; j < this.layers[i].size; j++) {
        const offset = this.next_sizes[i] * j;
        this.layers[i].biases[j] = Math.random() * 2.0 - 1.0;

        for (let k = 0; k < nextSize; k++) {
          this.layers[i].weights[offset + k] = Math.random() * 2.0 - 1.0;
        }
      }
    }
  }

  get asJSON() {
    return JSON.stringify(
      this.layers.map(layer => [layer.size, [...layer.biases], [...layer.weights]])
    );
  }

  set asJSON(json: string) {
    const raw: [number, number[], number[]][] = JSON.parse(json);
    raw.forEach(([size, biases, weights], layer_index) => {
      this.layers[layer_index].size = size;

      biases.forEach((bias, index) => {
        this.layers[layer_index].biases[index] = bias;
      });

      weights.forEach((neuron_weights, index) => {
        this.layers[layer_index].weights[index] = neuron_weights;
      });
    });
  }

  get last_layer(): Layer {
    return this.layers.at(-1) as Layer;
  }

  async feed_forward(input: number[]) {
    // set input as activations of first layer
    input.forEach((value, index) => {
      this.layers[0].activations[index] = value;
    });

    // feed layers :^)
    let next = 0;
    for (let i = 1; i < this.layers.length; i++)  {
      let prev_layer = this.layers[i - 1];
      let curr_layer = this.layers[i];

      for (let j = 0; j < curr_layer.size; j++) {
        next = curr_layer.biases[j];
        for (let k = 0; k < prev_layer.size; k++) {
          next += prev_layer.activations[k] * prev_layer.weights[this.next_sizes[i - 1] * k + j];
        }
        curr_layer.activations[j] = sigmoid(next);
      }
    }
  }

  back_propagation(learn_rate: number, output: number[], etp: ETP<ETP_Params, number>): Network {
    // calc errors
    let errors = new Float64Array(new SharedArrayBuffer(Float64Array.BYTES_PER_ELEMENT * this.last_layer.size)).fill(0.0);
    for (let i = 0; i < this.last_layer.size; i++) {
      errors[i] = output[i] - this.last_layer.activations[i];
    }

    // iterate over layers and calc/apply gradients, errors, weights, biases; skip last layer from iteration
    for (let layer_index = this.layers.length - 2; layer_index >= 0; layer_index--) {
      let current_layer = this.layers[layer_index];
      let prev_layer = this.layers[layer_index + 1];

      let gradients = new Float64Array(new SharedArrayBuffer(Float64Array.BYTES_PER_ELEMENT * prev_layer.size)).fill(0.0);
      let new_deltas = new Float64Array(new SharedArrayBuffer(Float64Array.BYTES_PER_ELEMENT * prev_layer.size * current_layer.size)).fill(0.0);
      let next_errors = new Float64Array(new SharedArrayBuffer(Float64Array.BYTES_PER_ELEMENT * current_layer.size)).fill(0.0);

      // calc gradients
      for (let i = 0; i < prev_layer.size; i++) {
        gradients[i] = errors[i] * sigmoid_prime(prev_layer.activations[i]) * learn_rate;
      }

      // calc deltas
      let offset = 0;
      for (let i = 0; i < prev_layer.size; i++) {
        offset = i * current_layer.size;
        for (let j = 0; j < current_layer.size; j++) {
          new_deltas[offset + j] = gradients[i] * current_layer.activations[j];
        }
      }

      // calc next errors
      for (let i = 0; i < current_layer.size; i++) {
        offset = this.next_sizes[layer_index] * i;
        for (let j = 0; j < prev_layer.size; j++) {
          next_errors[i] += current_layer.weights[offset + j] * errors[j];
        }
      }

      // calc/apply new weights
      for (let i = 0; i < prev_layer.size; i++) {
        offset = i * current_layer.size;
        for (let j = 0; j < current_layer.size; j++) {
          current_layer.weights[this.next_sizes[layer_index] * j + i] += new_deltas[offset + j];
        }
      }

      // apply gradients to biases
      for (let i = 0; i < prev_layer.size; i++) {
        prev_layer.biases[i] += gradients[i];
      }

      // swap errors
      errors = next_errors.slice(0);
    }

    return this;
  }

  get_result_softed() {
    return (this.last_layer.activations || []).map(v => Math.floor(v * 100) / 100);
  }

  get_error(output: number[]) {
    return (this.last_layer.activations || [] as number[])
      .map((value, index) => {
        return (value + output[index]) * (value + output[index]);
      })
      .reduce((acc, val) => acc + val, 0);
  }

  clone(): Network {
    const next_net = new Network(this.layers.map(_ => _.size));
    next_net.layers = this.layers.map((l) => ({
      biases: l.biases.slice(0),
      weights: l.weights.slice(0),
      activations: l.activations.slice(0),
      size: l.size,
    }));
    return next_net;
  }
}
