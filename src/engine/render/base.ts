import Engine from '../core/engine';
import Camera2DController from '../input/2dcam';
import Camera from './camera';
import Scene from './scene';
import Stats from 'stats-js';
import { GUI } from 'dat.gui';
import TextureAtlas from './texture';
import { RenderPass } from './interfaces';
import { PBRSpritePass } from './pbr-sprite-pass';
import SpriteStore from './sprite-store';
import { FlatSpritePass } from './flat-sprite-pass';

export const RESOLUTION = [2560, 1440];

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
    private canvas: HTMLCanvasElement;
    private ctx: GPUCanvasContext;

    private stats = new Stats();
    private gui = new GUI();

    private static DefaultDepthStencilTex: GPUTexture = null;
    static DefaultDepthStencilView: GPUTextureView = null;

    public readonly viewport: [number, number] = [0, 0];

    public store: SpriteStore;
    public readonly passes: RenderPass[] = [];
    private scene: Scene;
    private mainCamera: Camera;
    private cameraCtrl: Camera2DController;
    private textures: TextureAtlas;

    public postRenderHooks: Function[] = [];
    public ready = new Promise<void>(resolve => (this.resolveReady = resolve));
    private resolveReady: () => void;

    constructor(engine: Engine) {
        this.engine = engine;
        this.canvas = engine.params.canvas;
        this.ctx = this.canvas.getContext('webgpu');
        this.textures = new TextureAtlas();

        this.canvas.width = RESOLUTION[0] * engine.params.dpr;
        this.canvas.height = RESOLUTION[1] * engine.params.dpr;

        document.body.appendChild(this.stats.dom);
    }

    public readonly options = {
        'Brush Fill': false,
        'Brush Radius': 0.1,
        'Always Spawn': true,
        'Max Asteroids': 250_000,
        'Platform Velocity': 1.0 / 30.0,
        'Random X Offset': 0,
        'Random Y Offset': 20,
        'Random X Range': 100,
        'Random Y Range': 10,
        'Random Velocity': 0.1 / 30,
    };

    async init() {
        console.log('Loading textures...');
        const loading = this.textures.load();

        console.log('Initializing WebGPU...');
        if (!GDevice.readyState) {
            GDevice.readyState = 1;
            GDevice.adapter = await navigator.gpu.requestAdapter();
            GDevice.device = await GDevice.adapter.requestDevice({
                // requiredLimits: {
                //     maxColorAttachmentBytesPerSample: 64,
                // },
            });
            GDevice.readyState = 2;
        } else return;

        console.log(GDevice.device.limits);

        GDevice.format = navigator.gpu.getPreferredCanvasFormat();
        this.ctx.configure({
            device: GDevice.device,
            format: GDevice.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
            alphaMode: 'premultiplied',
        });

        this.scene = new Scene(this);
        this.mainCamera = new Camera(this.scene);
        this.cameraCtrl = new Camera2DController(this, this.mainCamera);

        const p = this.engine.params;
        GDevice.screen = p.screen;
        this.resize(window.innerWidth * p.dpr, window.innerHeight * p.dpr);
        window.addEventListener('resize', () =>
            this.resize(window.innerWidth * p.dpr, window.innerHeight * p.dpr),
        );

        this.store = new SpriteStore();
        this.passes.push(new PBRSpritePass(this.store), new FlatSpritePass(this.store));

        this.start();
        await Promise.all(this.passes.map(pass => pass.init()).concat([loading]));

        this.store.updateSpriteTexture(this.textures);

        const gui = this.gui;
        const options = this.options;
        const brush = gui.addFolder('Brush');
        brush.add(options, 'Brush Fill');
        brush.add(options, 'Brush Radius', 0.1, 32, 0.1);
        const asteroid = gui.addFolder('Asteroid');
        asteroid.add(options, 'Always Spawn');
        asteroid.add(options, 'Max Asteroids', 0, 2000000, 1000);
        asteroid.add(options, 'Platform Velocity', 0, 0.5, 0.001);
        asteroid.add(options, 'Random X Offset', -1000, 1000, 10);
        asteroid.add(options, 'Random Y Offset', -1000, 1000, 10);
        asteroid.add(options, 'Random X Range', 0, 1000, 10);
        asteroid.add(options, 'Random Y Range', 0, 1000, 10);
        asteroid.add(options, 'Random Velocity', 0, 0.025, 0.0001);

        // this.test();
        this.resolveReady();
    }

    // test() {
    //     const pass = this.getPass(FlatSpritePass);
    //     const chunk = pass.allocChunk();

    //     const ELEM = 10 * 10;
    //     const posX = new Int32Array(ELEM);
    //     const posY = new Int32Array(ELEM);
    //     const state = new Uint32Array(ELEM);
    //     const HALF_TILE = 1 << 10;

    //     for (let x = 0; x < 10; x++) {
    //         for (let y = 0; y < 10; y++) {
    //             posX[x + y * 10] = ((x - 5) << 11) + HALF_TILE;
    //             posY[x + y * 10] = ((y - 5) << 11) + HALF_TILE;
    //             state[x + y * 10] = 4 << 16;
    //         }
    //     }
    //     pass.updateChunk(chunk, posX, posY, state);
    // }

    getPass<T extends RenderPass>(ctor: new (...args: any[]) => T): T {
        return this.passes.find(p => p instanceof ctor) as T;
    }

    brush(x: number, y: number) {
        this.engine.sim.call(
            'brush',
            x,
            y,
            this.options['Brush Radius'],
            0,
            this.options['Brush Fill'],
        );
    }

    resize(w: number, h: number) {
        if (!Renderer.DefaultDepthStencilTex) {
            const ct = this.ctx.getCurrentTexture();
            Renderer.DefaultDepthStencilTex = GDevice.device.createTexture({
                size: { width: ct.width, height: ct.height },
                mipLevelCount: 1,
                dimension: '2d',
                format: 'depth24plus-stencil8',
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
            });
            Renderer.DefaultDepthStencilView =
                Renderer.DefaultDepthStencilTex.createView();
            for (const pass of this.passes) pass.resize(w, h);
        }

        const DPR = this.engine.params.dpr;
        this.viewport[0] = w * DPR;
        const ratio =
            window.innerWidth / window.innerHeight / (RESOLUTION[0] / RESOLUTION[1]);
        if (ratio < 1) {
            this.viewport[1] = ratio * this.canvas.clientHeight * DPR;
        } else {
            this.viewport[1] = this.canvas.clientHeight * DPR;
        }
    }

    start() {
        if (this.RAF) return;

        const cb = async (now: number) => {
            GDevice.now = now;
            this.stats.begin();

            this.mainCamera.update(this.viewport);

            const dt = Math.min(1 / 60, (now - this.lastRAF) / 1000);

            this.store.updateCam(this.mainCamera.view);
            for (const pass of this.passes) {
                const clear = pass === this.passes[0];
                pass.render(dt, now, this.ctx.getCurrentTexture(), clear);
            }

            this.RAF = requestAnimationFrame(cb);
            this.lastRAF = now;

            this.postRenderHooks.forEach(h => h());
            this.stats.end();
        };
        console.log('Starting animation loop');
        this.RAF = requestAnimationFrame(cb);
    }

    stop() {
        if (!this.RAF) return;
        cancelAnimationFrame(this.RAF);
        this.RAF = 0;
    }
}
