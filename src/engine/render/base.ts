import Cloth from '../cloth/cloth';
import Engine from '../core/engine';
import CameraRotator from '../input/input';
import Cube from '../primitives/cube';
import Camera from './camera';
import { DeferredPass } from './deferred-pass';
import PointLight from './pointlight';
import Scene from './scene';
import Stats from 'stats-js';
import { GUI } from 'dat.gui';
import Fluid from '../fluid/fluid';

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
    private canvas: HTMLCanvasElement | OffscreenCanvas;
    private ctx: GPUCanvasContext;

    private stats = new Stats();
    private gui = new GUI();

    private static DefaultDepthStencilTex: GPUTexture = null;
    static DefaultDepthStencilView: GPUTextureView = null;

    private readonly dimension: [number, number];

    private pass: DeferredPass;
    private scene: Scene;
    private mainCamera: Camera;
    private cameraCtrl: CameraRotator;

    private readonly cubes: Cube[] = [];
    private cloth: Cloth;
    private fluid: Fluid;

    constructor(engine: Engine) {
        this.engine = engine;
        this.canvas = engine.params.canvas;
        this.ctx = this.canvas.getContext('webgpu');
        this.dimension = [this.canvas.width, this.canvas.height];

        document.body.appendChild(this.stats.dom);
    }

    async init() {
        if (!GDevice.readyState) {
            GDevice.readyState = 1;
            GDevice.adapter = await navigator.gpu.requestAdapter();
            GDevice.device = await GDevice.adapter.requestDevice();
            GDevice.readyState = 2;
        } else return;

        GDevice.format = this.ctx.getPreferredFormat(GDevice.adapter);
        this.ctx.configure({
            device: GDevice.device,
            format: GDevice.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.scene = new Scene(this);
        this.mainCamera = new Camera(this.scene);
        this.cameraCtrl = new CameraRotator(this.mainCamera);

        const p = this.engine.params;
        GDevice.screen = p.screen;
        this.resize(p.screen.width * p.dpr, p.screen.height * p.dpr);

        this.pass = new DeferredPass();
        this.pass.init().then(() => this.setupFluid());

        this.setupLights();

        this.start();
    }

    setupCubes() {
        const POS_RANGE = 100;
        const rng = (min: number, max: number) => Math.random() * (max - min) + min;

        const CUBES = 250;
        for (let i = 0; i < CUBES; i++) {
            const cube = new Cube(this.pass);
            cube.transform.position = [
                rng(-POS_RANGE, POS_RANGE),
                rng(-POS_RANGE, POS_RANGE),
                rng(-POS_RANGE, POS_RANGE),
            ];
            const scale = Math.random() * 5 + 5;
            cube.transform.scale = [scale, scale, scale];
            this.pass.meshDrawList.push(cube);
            this.cubes.push(cube);
        }
    }

    setupLights() {
        const POS_RANGE = 100;
        const rng = (min: number, max: number) => Math.random() * (max - min) + min;

        const LIGHTS = 250;
        for (let i = 0; i < LIGHTS; i++) {
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

            light.radius = 64;

            this.pass.lightDrawList.push(light);
        }

        this.pass.updateLight();
    }

    setupCloth() {
        const gui = this.gui;
        const options = {
            'Debug Normal': false,
            'Wind X': 2,
            'Wind Y': -1,
            'Wind Z': 5,
            'Floor Y': 5,
            'Simulation Speed': 1,
            'Reset Wind': function () {
                options['Wind X'] = options['Wind Y'] = options['Wind Z'] = 0;
                updateWind();
                gui.updateDisplay();
            },
        };

        const updateWind = () => {
            this.cloth.setWindSpeed([
                options['Wind X'],
                options['Wind Y'],
                options['Wind Z'],
            ]);
        };

        const updateFloor = () => {
            this.cloth.setFloor(options['Floor Y']);
        };

        const DIM = 64;
        this.cloth = new Cloth(this.pass, DIM, DIM, {
            mass: 1,
            rest_length: 100 / DIM,
            springConstant: DIM * DIM,
            dampingConstant: 50,
            floor: options['Floor Y'],
            wind: [options['Wind X'], options['Wind Y'], options['Wind Z']],
            gravity: [0, -9.81, 0],
        });
        this.cloth.transform.position = [-DIM / 2, -DIM / 2, 0];
        this.cloth.transform.update();
        this.cloth.transform.updateInverse();

        this.pass.clothDrawList.push(this.cloth);

        console.log(GDevice.device.limits);
        // setTimeout(() => this.stop(), 100);

        // Wind speed range
        const WR = 10;

        gui.add(options, 'Debug Normal').onChange(v => (Cloth.debug = v));
        gui.add(options, 'Wind X', -WR, WR, 0.01).onChange(updateWind);
        gui.add(options, 'Wind Y', -WR, WR, 0.01).onChange(updateWind);
        gui.add(options, 'Wind Z', -WR, WR, 0.01).onChange(updateWind);
        gui.add(options, 'Floor Y', 0, 1.5 * DIM, 0.01).onChange(updateFloor);
        gui.add(options, 'Simulation Speed', 0.1, 25, 0.01).onChange(
            v => (Cloth.sampleRate = 1 / v),
        );
        gui.add(options, 'Reset Wind');

        this.cloth.fixedPoints.forEach(({ row, col, x, y }, i) => {
            console.log({ row, col, x, y });

            const coord = { x, y };
            const updateCoord = () =>
                this.cloth.setFixedPointPosition(row, col, coord.x, coord.y);
            const folder = gui.addFolder(`Fixed Point#${i}`);
            folder.add(coord, 'x', -DIM * 2, DIM * 2, 0.01).onChange(updateCoord);
            folder.add(coord, 'y', -DIM * 2, DIM * 2, 0.01).onChange(updateCoord);
        });
    }

    setupFluid() {
        this.fluid = new Fluid(this.pass, { max_num: 1 });
        this.pass.fluidDrawList.push(this.fluid);
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

        this.mainCamera.aspect = w / h;
    }

    start() {
        if (this.RAF) return;

        const cb = async (now: number) => {
            this.stats.begin();

            const t = now * 0.001;

            this.mainCamera.update();

            for (let i = 0; i < this.cubes.length; i++) {
                const cube = this.cubes[i];

                cube.transform.rotation = [Math.sin(t + i), Math.cos(t + i), 0, 0];
                cube.transform.update();
                cube.transform.updateInverse();
            }

            const dt = Math.min(1 / 60, (now - this.lastRAF) / 1000);

            await this.pass.render(
                dt,
                now,
                this.ctx.getCurrentTexture().createView(),
                this.mainCamera.view,
            );

            this.RAF = requestAnimationFrame(cb);
            this.lastRAF = now;

            this.stats.end();
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
