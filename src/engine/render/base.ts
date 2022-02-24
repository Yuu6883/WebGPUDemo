import { mat4, vec4 } from 'gl-matrix';
import Cloth from '../cloth/cloth';
import Engine from '../core/engine';
import Cube from '../primitives/cube';
import Camera from './camera';
import { DeferredPipeline } from './pipeline';
import PointLight from './pointlight';
import Scene from './scene';

export const GDevice: {
    readyState: 0 | 1 | 2;
    adapter: GPUAdapter;
    device: GPUDevice;
    format: GPUTextureFormat;
    screen: { width: number; height: number };
} = {
    readyState: 0,
    adapter: null,
    device: null,
    format: null,
    screen: null,
};

export const checkDevice = () => {
    if (GDevice.readyState !== 2) throw new Error('Device not ready');
};

export default class Renderer {
    private RAF = 0;
    private lastRAF = performance.now();

    private engine: Engine;
    private canvas: OffscreenCanvas;
    private ctx: GPUCanvasContext;

    private static DefaultDepthStencilTex: GPUTexture = null;
    static DefaultDepthStencilView: GPUTextureView = null;

    private readonly dimension: [number, number];

    private pipeline: DeferredPipeline;
    private scene: Scene;
    private mainCamera: Camera;

    private readonly cubes: Cube[] = [];
    private cloth: Cloth;

    constructor(engine: Engine) {
        this.engine = engine;
        this.canvas = engine.params.canvas;
        this.ctx = this.canvas.getContext('webgpu');
        this.dimension = [this.canvas.width, this.canvas.height];
    }

    async init() {
        if (!GDevice.readyState) {
            GDevice.readyState = 1;
            GDevice.adapter = await navigator.gpu.requestAdapter();
            GDevice.device = await GDevice.adapter.requestDevice();
            GDevice.readyState = 2;
        } else return;

        const p = this.engine.params;
        GDevice.screen = p.screen;
        this.resize(p.screen.width * p.dpr, p.screen.height * p.dpr);

        GDevice.format = this.ctx.getPreferredFormat(GDevice.adapter);
        this.ctx.configure({
            device: GDevice.device,
            format: GDevice.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.pipeline = new DeferredPipeline();

        this.scene = new Scene(this);
        this.mainCamera = new Camera(this.scene);

        const POS_RANGE = 10;
        const rng = (min: number, max: number) => Math.random() * (max - min) + min;

        for (let i = 0; i < 100; i++) {
            const cube = new Cube(this.pipeline);
            cube.transform.position = [
                rng(-POS_RANGE, POS_RANGE),
                rng(-POS_RANGE, POS_RANGE),
                rng(-POS_RANGE, POS_RANGE),
            ];
            const scale = Math.random() * 0.25 + 0.25;
            cube.transform.scale = [scale, scale, scale];
            this.pipeline.meshDrawList.push(cube);
            this.cubes.push(cube);
        }

        for (let i = 0; i < 250; i++) {
            const light = new PointLight();
            light.position = new Float32Array([
                rng(-POS_RANGE, POS_RANGE),
                rng(-POS_RANGE, POS_RANGE),
                rng(-POS_RANGE, POS_RANGE),
            ]);

            light.color = new Float32Array([
                rng(0.25, 0.75),
                rng(0.25, 0.75),
                rng(0.25, 0.75),
            ]);

            light.radius = 5;

            this.pipeline.lightDrawList.push(light);
        }

        this.pipeline.updateLight();

        this.cloth = new Cloth(this.pipeline, 10, 10, {
            mass: 1,
            rest_length: 1,
            springConstant: 1,
            dampingConstant: 1,
            wind: [0, 0, 0],
            gravity: [0, -9.81, 0],
        });

        this.pipeline.clothDrawList.push(this.cloth);

        this.start();

        console.log(GDevice.device.limits);
        // setTimeout(() => this.stop(), 100);
    }

    resize(w: number, h: number) {
        this.canvas.width = w;
        this.canvas.height = h;

        this.ctx.configure({
            device: GDevice.device,
            format: this.ctx.getPreferredFormat(GDevice.adapter),
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
            size: { width: w, height: h },
        });

        if (this.dimension[0] < w || this.dimension[1] < h) {
            Renderer.DefaultDepthStencilTex?.destroy();
            Renderer.DefaultDepthStencilTex = GDevice.device.createTexture({
                size: { width: w, height: h },
                mipLevelCount: 1,
                dimension: '2d',
                format: 'depth24plus-stencil8',
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
            });
            Renderer.DefaultDepthStencilView =
                Renderer.DefaultDepthStencilTex.createView();
        }

        this.dimension[0] = w;
        this.dimension[1] = h;
        GDevice.screen.width = w;
        GDevice.screen.height = h;
    }

    start() {
        if (this.RAF) return;
        const cb = (now: number) => {
            const t = now * 0.001;

            for (let i = 0; i < this.cubes.length; i++) {
                const cube = this.cubes[i];

                cube.transform.rotation = [Math.sin(t + i), Math.cos(t + i), 0, 0];
                cube.transform.update();
                cube.transform.updateInverse();
            }

            const dt = Math.max(0.1, (now - this.lastRAF) / 1000);

            this.pipeline.render(
                dt,
                this.ctx.getCurrentTexture().createView(),
                this.mainCamera.vp,
            );

            this.RAF = requestAnimationFrame(cb);
            this.lastRAF = now;
        };
        this.RAF = requestAnimationFrame(cb);
    }

    stop() {
        if (!this.RAF) return;
        cancelAnimationFrame(this.RAF);
        this.RAF = 0;
    }

    get aspectRatio() {
        return this.dimension[0] / this.dimension[1];
    }
}
