import Engine from '../core/engine';
import { FlatSpritePass, RenderChunk } from '../render/flat-sprite-pass';
import { PBRSpritePass } from '../render/pbr-sprite-pass';

export default class SimWorker {
    public ready = false;
    public tick = 0;

    private readonly engine: Engine;
    private readonly worker: Worker;

    private readonly buffers: Record<string, ArrayBuffer> = {
        posX: null,
        posY: null,
        state: null,
    };

    private readonly chunkMap: Map<string, RenderChunk> = new Map();

    constructor(engine: Engine) {
        this.engine = engine;
        this.worker = new Worker(new URL('./worker.ts', import.meta.url));
        this.worker.addEventListener('message', ({ data }) => {
            if (data.event === 'ready') {
                this.ready = true;
                this.engine.renderer.ready.then(() => this.start());
            } else if (data.event === 'tick') {
                this.tick++;
                if (this.buffers.posX || this.buffers.posY || this.buffers.state)
                    return console.error('Buffers not cleared???');

                const size = data.size;
                this.buffers.posX = data.buffers.posX;
                this.buffers.posY = data.buffers.posY;
                this.buffers.state = data.buffers.state;

                const pass = this.engine.renderer.getPass(PBRSpritePass);

                pass.updateSprites(
                    new Int32Array(this.buffers.posX, 0, size),
                    new Int32Array(this.buffers.posY, 0, size),
                    new Uint32Array(this.buffers.state, 0, size),
                );
                this.engine.renderer.store.updateTick(this.tick);
            } else if (data.event === 'chunk') {
                const pass = this.engine.renderer.getPass(FlatSpritePass);
                const key = `${data.chunkX}:${data.chunkY}`;
                let chunk = this.chunkMap.get(key);

                if (data.empty) {
                    if (chunk) chunk.free();
                    this.chunkMap.delete(key);
                    return;
                }

                if (!chunk) {
                    chunk = pass.allocChunk();
                    this.chunkMap.set(key, chunk);
                }

                pass.updateChunk(
                    chunk,
                    new Int32Array(data.posX),
                    new Int32Array(data.posY),
                    new Uint32Array(data.state),
                );
            }
        });

        this.engine.renderer.postRenderHooks.push(() => this.sim());
    }

    start() {
        this.call('start', this.engine.renderer.options['Max Asteroids']);
    }

    sim() {
        if (!this.ready) return;
        if (!this.buffers.posX || !this.buffers.posY || !this.buffers.state) return;
        const options = this.engine.renderer.options;

        this.worker.postMessage(
            {
                call: 'sim',
                args: [-Math.abs(options['Platform Velocity'])],
                buffers: this.buffers,
                fill: options['Always Spawn'] ? options['Max Asteroids'] : 0,
                rng: [
                    options['Random X Offset'],
                    options['Random Y Offset'],
                    options['Random X Range'],
                    options['Random Y Range'],
                    options['Random Velocity'],
                ],
            },
            Object.values(this.buffers),
        );
        this.buffers.posX = null;
        this.buffers.posY = null;
        this.buffers.state = null;
    }

    call(call: string, ...args: any[]) {
        this.worker.postMessage({ call, args });
    }
}
