import { Network } from "./lib/Network";

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

let net_conf = [784, 512, 256, 64, 10];

fetch(`weights_${net_conf.join('_')}.bin`).then(res => res.arrayBuffer()).then((data) => {
  const network = new Network(net_conf);
  network.set_as_bin(data, true);

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
  }

  const recalc = () => {
    if (ctx) {
      const input: number[] = [];
      const image = ctx.getImageData(0, 0, 28, 28);
      for (let i = 0, n = image.data.length; i < n; i += 4) {
        input.push(Math.abs(255 - image.data[i]) / 255);
      }

      network.feed_forward(input);
      const result = network.get_result_softed();

      for (let i = 0; i < 28; i++) {
        console.log(input.slice(i * 28, 28).map(_ => Math.round(_).toString()).join(''));
      }

      result.forEach((v, i) => {
        (document.getElementById(`${i}`) as HTMLInputElement).value = (v * 10).toString();
      });
    }
  }

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
