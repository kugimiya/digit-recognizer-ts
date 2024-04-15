export type Layer = {
  neuron_biases: number[];
  out_weights: number[];
}

export type LayerResult = {
  input_sums: number[];
  activations: number[];
}

export const sigmoid = (x: number) => {
  return Math.exp(x) / (Math.exp(x) + 1)
};

export class Network {
  layers: Layer[] = [];
  layers_results: LayerResult[] = [];

  constructor(neurons_counts: number[]) {
    for (let i = 0; i < neurons_counts.length; i++) {
      const weights_count = i === neurons_counts.length - 1
        ? 0
        : neurons_counts[i] * neurons_counts[i + 1];

      this.layers.push({
        neuron_biases: new Array(neurons_counts[i]).fill(0),
        out_weights: new Array(weights_count).fill(0)
      });

      this.layers_results.push({
        input_sums: new Array(neurons_counts[i]).fill(0),
        activations: new Array(neurons_counts[i]).fill(0)
      });
    }
  }

  randomize() {
    this.layers.forEach((layer) => {
      layer.out_weights = layer.out_weights.map(() => Math.random() - 0.5);
      layer.neuron_biases = layer.neuron_biases.map(() => Math.random() - 0.5);
    });
  }

  run(input: number[]) {
    // set input as activations of first layer
    input.forEach((value, index) => {
      this.layers_results[0].activations[index] = value;
    });

    // feed layers :^)
    for (let layer_index = 1; layer_index < this.layers.length; layer_index++) {
      let previous_activations = this.layers_results[layer_index - 1].activations;

      for (let neuron_index = 0; neuron_index < this.layers[layer_index].neuron_biases.length; neuron_index++) {
        const weight_offset = neuron_index * this.layers[layer_index - 1].neuron_biases.length;
        const bias = this.layers[layer_index].neuron_biases[neuron_index];
        const input_sum = previous_activations
          .map((value, index) => value * this.layers[layer_index - 1].out_weights[weight_offset + index])
          .reduce((acc, val) => acc + val, 0);

        const pre_activation = input_sum + bias;
        this.layers_results[layer_index].input_sums[neuron_index] = pre_activation;
        this.layers_results[layer_index].activations[neuron_index] = sigmoid(pre_activation);
      }
    }
  }

  get_result_softed() {
    return (this.layers_results.at(-1)?.activations || []).map(v => Math.floor(v * 10) / 10);
  }

  get_error(output: number[]) {
    return (this.layers_results.at(-1)?.activations || [] as number[])
      .map((value, index) => {
        return (value + output[index]) * (value + output[index]);
      })
      .reduce((acc, val) => acc + val, 0);
  }

  clone(): Network {
    const next_net = new Network(this.layers.map(_ => _.neuron_biases.length));
    next_net.layers = this.layers.map((l) => ({ neuron_biases: l.neuron_biases.slice(0), out_weights: l.out_weights.slice(0) }));
    next_net.layers_results = this.layers_results.map((r) => ({ input_sums: r.input_sums.slice(0), activations: r.activations.slice(0) }));
    return next_net;
  }
}
