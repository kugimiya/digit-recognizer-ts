export type Weight = {
  value: number;
  layer: number;
  from_neuron_index: number;
  to_neuron_index: number;
}

export enum LayerType {
  Input,
  Output,
  Hidden
}

export type Layer = {
  value: number;
  neurons_count: number;
  neurons_out_values: number[];
  type: LayerType;
  weights: Weight[];
}

export function deep_clone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

export class Network {
  layers: Layer[] = [];

  make(layers_neurons_counts: number[]) {
    console.log('INFO: making network');

    const layers_count = layers_neurons_counts.length;

    this.layers = layers_neurons_counts.map((neurons_count: number, index) => {
      const type: Layer['type'] = index === 0 ? LayerType.Input : index !== layers_count - 1 ? LayerType.Output : LayerType.Hidden;
      const weights: Weight[] = [];
      const neurons_out_values: number[] = new Array<number>(neurons_count).fill(0);

      if (type !== LayerType.Input) {
        const prev_layer_neurons_counts = layers_neurons_counts[index - 1];
        for (let to_neuron_index = 0; to_neuron_index < neurons_count; to_neuron_index++) {
          for (let from_neuron_index = 0; from_neuron_index < prev_layer_neurons_counts; from_neuron_index++) {
            weights.push({ value: 0, layer: index, from_neuron_index, to_neuron_index });
          }
        }
      }

      return ({
        value: index,
        type,
        neurons_count,
        neurons_out_values,
        weights,
      }) as Layer;
    });
  }

  static mutate(layers: Layer[]): Network {
    const network = new Network();
    let cloned = deep_clone(layers);

    for (let layer_index = 0; layer_index < cloned.length; layer_index++) {
      cloned[layer_index].weights = cloned[layer_index].weights.map(w => ({ ...w, value: w.value + ((Math.random() / 100) - 0.005) }));
    }

    network.layers = cloned;

    return network;
  }

  static activation_function_sigmoid(x: number) {
    return 1 / (1 + Math.pow(Math.E, -1 * x));
  }

  static clone(layers: Layer[]): Network {
    const network = new Network();
    let cloned = deep_clone(layers);
    network.layers = cloned;
    return network;
  }

  static run(layers: Layer[], input_values: number[]): Network {
    const network = new Network();
    let cloned = deep_clone(layers);
    cloned[0].neurons_out_values = cloned[0].neurons_out_values.map((_, index) => input_values[index]);

    for (let i = 1; i < cloned.length; i++) {
      let layer = cloned[i];
      let prev_layer = cloned[i - 1];

      for (let neuron_index = 0; neuron_index < layer.neurons_count; neuron_index++) {
        const related_weights = layer.weights.filter(weight => weight.to_neuron_index === neuron_index);
        const input_value_summa = related_weights.reduce((acc, cur_weight) => {
          acc += prev_layer.neurons_out_values[cur_weight.from_neuron_index] * cur_weight.value;
          return acc;
        }, 0);
        layer.neurons_out_values[neuron_index] = Network.activation_function_sigmoid(input_value_summa);
      }
    }

    network.layers = cloned;
    return network;
  }
}
