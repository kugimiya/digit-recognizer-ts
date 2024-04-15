declare module "lib/Network" {
    export type Layer = {
        size: number;
        biases: number[];
        weights: number[][];
        activations: number[];
    };
    export const sigmoid: (x: number) => number;
    export const sigmoid_prime: (x: number) => number;
    export class Network {
        layers: Layer[];
        constructor(neurons_counts: number[]);
        randomize(): void;
        get last_layer(): Layer;
        feed_forward(input: number[]): void;
        back_propagation(learn_rate: number, output: number[]): Network;
        get_result_softed(): number[];
        get_error(output: number[]): number;
        clone(): Network;
        clear(): Network;
    }
}
declare module "front" { }
//# sourceMappingURL=front.d.ts.map