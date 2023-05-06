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
import Particles from '../particles/particles';
import { vec3 } from 'gl-matrix';

export const GDevice: {
    readyState: 0 | 1 | 2;
    adapter: GPUAdapter;
    device: GPUDevice;
    format: GPUTextureFormat;
    screen: { width: number; height: number };
    now: number;
} = {
    readyState: 0,
    adapter: null,
    device: null,
    format: null,
    screen: null,
    now: performance.now(),
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
    private particles: Particles;

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

        console.log(GDevice.device.limits);

        GDevice.format = navigator.gpu.getPreferredCanvasFormat();
        this.ctx.configure({
            device: GDevice.device,
            format: GDevice.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
            alphaMode: 'opaque',
        });

        this.scene = new Scene(this);
        this.mainCamera = new Camera(this.scene);
        this.cameraCtrl = new CameraRotator(this.mainCamera);

        const p = this.engine.params;
        GDevice.screen = p.screen;
        this.resize(p.screen.width * p.dpr, p.screen.height * p.dpr);

        this.pass = new DeferredPass();

        this.setupLights();
        this.start();
        await this.pass.init();
    }

    setupCubes() {
        const POS_RANGE = 250;
        const rng = (min: number, max: number) => Math.random() * (max - min) + min;

        const CUBES = 2500;
        for (let i = 0; i < CUBES; i++) {
            const cube = new Cube(this.pass);
            cube.transform.position = [
                rng(-POS_RANGE, POS_RANGE),
                rng(0, 2 * POS_RANGE),
                rng(-POS_RANGE, POS_RANGE),
            ];
            const scale = Math.random() * 10 + 1;
            cube.transform.rotation = [
                Math.random(),
                Math.random(),
                Math.random(),
                Math.random(),
            ];
            cube.transform.scale = [scale, scale, scale];
            this.pass.meshDrawList.push(cube);
            this.cubes.push(cube);
        }
    }

    setupLights() {
        const POS_RANGE = 500;
        const rng = (min: number, max: number) => Math.random() * (max - min) + min;

        const LIGHTS = 1024;
        for (let i = 0; i < LIGHTS; i++) {
            const light = new PointLight();
            light.position = new Float32Array([
                rng(-POS_RANGE, POS_RANGE),
                rng(0, POS_RANGE),
                rng(-POS_RANGE, POS_RANGE),
            ]);

            light.color = new Float32Array([
                rng(0.25, 0.75),
                rng(0.25, 0.75),
                rng(0.25, 0.75),
            ]);

            light.radius = 128;

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
            const coord = { x, y };
            const updateCoord = () =>
                this.cloth.setFixedPointPosition(row, col, coord.x, coord.y);
            const folder = gui.addFolder(`Fixed Point#${i}`);
            folder.add(coord, 'x', -DIM * 2, DIM * 2, 0.01).onChange(updateCoord);
            folder.add(coord, 'y', -DIM * 2, DIM * 2, 0.01).onChange(updateCoord);
        });
    }

    setupParticles() {
        const p = (this.particles = new Particles(this.pass, {
            max_num: 1000000,
            max_spawn_per_frame: 10000,
        }));
        this.pass.particlesDrawList.push(this.particles);

        const gui = this.gui;
        const options = {
            Pause: function () {
                p.pause = !p.pause;
            },
            Radius: p.radius,
        };
        const addVectorOption = (
            folder: GUI,
            vec: vec3,
            range: vec3,
            positiveOnly = false,
        ) => {
            const vecOp = {
                X: vec[0],
                Y: vec[1],
                Z: vec[2],
            };
            for (let i = 0; i < 3; i++)
                folder
                    .add(
                        vecOp,
                        'XYZ'.charAt(i),
                        positiveOnly ? 0 : vec[i] - range[i],
                        positiveOnly ? range[i] : vec[i] + range[i],
                        range[i] * 0.01,
                    )
                    .onChange(v => (vec[i] = v));
        };

        gui.add(options, 'Pause');
        const constOptions = {
            'Air Density': p.coeffients[0],
            Drag: p.coeffients[1],
            'Groud Elasticity': p.coeffients[2],
            'Groud Friction': p.coeffients[3],
        };
        const constants = gui.addFolder('Constants');
        for (let i = 0; i < 4; i++) {
            constants
                .add(
                    constOptions,
                    Object.keys(constOptions)[i],
                    0,
                    [2, 0.025, 1, 1][i],
                    0.001,
                )
                .onChange(v => (p.coeffients[i] = v));
        }

        const particle = gui.addFolder('Particle');

        particle.add(options, 'Radius', 0.01, 5, 0.001).onChange(v => (p.radius = v));

        const spawn = particle.addFolder('Spawn');
        const spawnOptions = {
            'Spawn Rate': p.spawn_rate,
            'Life Span': p.lifeSpan[0],
            'Life Variance': p.lifeSpan[1],
        };

        spawn
            .add(spawnOptions, 'Spawn Rate', 0, 100000, p.spawn_rate * 0.001)
            .onChange(v => (p.spawn_rate = v));
        spawn
            .add(spawnOptions, 'Life Span', 5000, 25000, 100)
            .onChange(v => (p.lifeSpan[0] = v));
        spawn
            .add(spawnOptions, 'Life Variance', 0, 5000, 100)
            .onChange(v => (p.lifeSpan[1] = v));

        addVectorOption(gui.addFolder('Wind'), p.wind, [100, 100, 100]);
        addVectorOption(
            particle.addFolder('Initial Position'),
            p.initPos,
            [1000, 1000, 1000],
        );
        addVectorOption(
            particle.addFolder('Position Variance'),
            p.variPos,
            [500, 500, 500],
            true,
        );
        addVectorOption(
            particle.addFolder('Initial Velocity'),
            p.initVel,
            [100, 100, 100],
        );
        addVectorOption(
            particle.addFolder('Velocity Variance'),
            p.variVel,
            [50, 50, 50],
            true,
        );
    }

    resize(w: number, h: number) {
        this.canvas.width = w;
        this.canvas.height = h;

        this.ctx.configure({
            device: GDevice.device,
            format: GDevice.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
            alphaMode: 'opaque',
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
            GDevice.now = now;
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
