define("lib/Network", ["require", "exports"], function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.Network = exports.sigmoid_prime = exports.sigmoid = void 0;
    const sigmoid = (x) => {
        return 1 / (1 + Math.exp(-x));
    };
    exports.sigmoid = sigmoid;
    const sigmoid_prime = (x) => {
        return x * (1 - x);
    };
    exports.sigmoid_prime = sigmoid_prime;
    class Network {
        layers = [];
        constructor(neurons_counts) {
            for (let i = 0; i < neurons_counts.length; i++) {
                let nextSize = 0;
                if (i < neurons_counts.length - 1) {
                    nextSize = neurons_counts[i + 1];
                }
                this.layers[i] = {
                    size: neurons_counts[i],
                    biases: new Array(neurons_counts[i]).fill(0),
                    weights: new Array(neurons_counts[i]).fill(0).map(() => new Array(nextSize).fill(0)),
                    activations: new Array(neurons_counts[i]).fill(0)
                };
            }
        }
        randomize() {
            for (let i = 0; i < this.layers.length; i++) {
                let nextSize = 0;
                if (i < this.layers.length - 1) {
                    nextSize = this.layers[i + 1].size;
                }
                for (let j = 0; j < this.layers[i].size; j++) {
                    this.layers[i].biases[j] = Math.random() * 2.0 - 1.0;
                    for (let k = 0; k < nextSize; k++) {
                        this.layers[i].weights[j][k] = Math.random() * 2.0 - 1.0;
                    }
                }
            }
        }
        get last_layer() {
            return this.layers.at(-1);
        }
        feed_forward(input) {
            // set input as activations of first layer
            input.forEach((value, index) => {
                this.layers[0].activations[index] = value;
            });
            // feed layers :^)
            for (let i = 1; i < this.layers.length; i++) {
                let prev_layer = this.layers[i - 1];
                let curr_layer = this.layers[i];
                for (let j = 0; j < curr_layer.size; j++) {
                    curr_layer.activations[j] = 0;
                    for (let k = 0; k < prev_layer.size; k++) {
                        curr_layer.activations[j] += prev_layer.activations[k] * prev_layer.weights[k][j];
                    }
                    curr_layer.activations[j] += curr_layer.biases[j];
                    curr_layer.activations[j] = (0, exports.sigmoid)(curr_layer.activations[j]);
                }
            }
        }
        back_propagation(learn_rate, output) {
            // calc errors
            let errors = new Array(this.last_layer.activations.length).fill(0);
            for (let i = 0; i < this.last_layer.activations.length; i++) {
                errors[i] = output[i] - this.last_layer.activations[i];
            }
            // iterate over layers and calc/apply gradients, errors, weights, biases; skip last layer from iteration
            for (let layer_index = this.layers.length - 2; layer_index >= 0; layer_index--) {
                const current_layer = this.layers[layer_index];
                const prev_layer = this.layers[layer_index + 1];
                // calc gradients
                const gradients = new Array(prev_layer.size).fill(0);
                for (let i = 0; i < prev_layer.size; i++) {
                    gradients[i] = errors[i] * (0, exports.sigmoid_prime)(prev_layer.activations[i]);
                    gradients[i] *= learn_rate;
                }
                // calc deltas
                const deltas = new Array(prev_layer.size).fill(0).map(() => new Array(current_layer.size).fill(0));
                for (let i = 0; i < prev_layer.size; i++) {
                    for (let j = 0; j < current_layer.size; j++) {
                        deltas[i][j] = gradients[i] * current_layer.activations[j];
                    }
                }
                // calc next errors
                const next_errors = new Array(current_layer.size).fill(0);
                for (let i = 0; i < current_layer.size; i++) {
                    for (let j = 0; j < prev_layer.size; j++) {
                        next_errors[i] += current_layer.weights[i][j] * errors[j];
                    }
                }
                // swap errors
                errors = next_errors.slice(0);
                // calc/apply new weights
                let next_weights = new Array(current_layer.weights.length).fill(0).map(() => new Array(current_layer.weights[0].length).fill(0));
                for (let i = 0; i < prev_layer.size; i++) {
                    for (let j = 0; j < current_layer.size; j++) {
                        next_weights[j][i] = current_layer.weights[j][i] + deltas[i][j];
                    }
                }
                current_layer.weights = next_weights.slice(0).map(_ => _.slice(0));
                // apply gradients to biases
                for (let i = 0; i < prev_layer.size; i++) {
                    prev_layer.biases[i] += gradients[i];
                }
            }
            return this;
        }
        get_result_softed() {
            return (this.last_layer.activations || []).map(v => Math.floor(v * 10) / 10);
        }
        get_error(output) {
            return (this.last_layer.activations || [])
                .map((value, index) => {
                return (value + output[index]) * (value + output[index]);
            })
                .reduce((acc, val) => acc + val, 0);
        }
        clone() {
            const next_net = new Network(this.layers.map(_ => _.size));
            next_net.layers = this.layers.map((l) => ({
                biases: l.biases.slice(0),
                weights: l.weights.slice(0).map(_ => _.slice(0)),
                activations: l.activations.slice(0),
                size: l.size,
            }));
            return next_net;
        }
        clear() {
            this.layers = this.layers.map((l) => ({
                biases: l.biases.slice(0),
                weights: l.weights.slice(0).map(_ => _.slice(0)),
                activations: new Array(l.activations.length).fill(0),
                size: l.size,
            }));
            return this;
        }
    }
    exports.Network = Network;
});
define("front", ["require", "exports", "lib/Network"], function (require, exports, Network_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style.width = '672px';
    canvas.style.height = '672px';
    canvas.style.border = '1px solid black';
    document.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    let ready = false;
    let draw = false;
    let cX = 0;
    let cY = 0;
    fetch('/weights.json').then(res => res.json()).then((data) => {
        const network = new Network_1.Network([784, 784 / 2, 784 / 4, 784 / 8, 784 / 16, 10]);
        network.layers = data.layers;
        ready = true;
        if (ctx) {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.closePath();
            ctx.fillStyle = 'black';
        }
        const doDraw = () => {
            if (ctx) {
                ctx.rect(cX - 1, cY - 1, 2, 2);
                ctx.fill();
                ctx.closePath();
            }
        };
        const recalc = () => {
            if (ctx) {
                const input = [];
                const image = ctx.getImageData(0, 0, 28, 28);
                for (let i = 0, n = image.data.length; i < n; i += 4) {
                    input.push(image.data[i] / 255);
                }
                network.feed_forward(input);
                const result = network.get_result_softed();
                for (let i = 0; i < 28; i++) {
                    console.log(input.slice(i * 28, 28).map(_ => Math.round(_).toString()).join(''));
                }
                result.forEach((v, i) => {
                    document.getElementById(`${i}`).value = (v * 10).toString();
                });
            }
        };
        canvas.addEventListener('mousemove', (ev) => {
            cX = ev.offsetX / 28 + 1;
            cY = ev.offsetY / 28 + 1;
            if (draw && ready) {
                requestAnimationFrame(doDraw);
            }
        });
        canvas.addEventListener('mousedown', () => {
            draw = true;
        });
        canvas.addEventListener('mouseup', () => {
            draw = false;
            recalc();
        });
    });
});
