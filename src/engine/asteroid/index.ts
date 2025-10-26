import Engine from '../core/engine';
import { SpritePass } from '../render/sprite-pass';

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

    public velocity = -1 / 30;

    constructor(engine: Engine) {
        this.engine = engine;
        this.worker = new Worker(new URL('./worker.ts', import.meta.url));
        this.worker.addEventListener('message', ({ data }) => {
            if (data.event === 'ready') {
                this.ready = true;
                this.start();
            } else if (data.event === 'tick') {
                this.tick++;
                if (this.buffers.posX || this.buffers.posY || this.buffers.state)
                    return console.error('Buffers not cleared???');

                const size = data.size;
                this.buffers.posX = data.buffers.posX;
                this.buffers.posY = data.buffers.posY;
                this.buffers.state = data.buffers.state;

                const pass = this.engine.renderer.pass;

                if (pass instanceof SpritePass) {
                    pass.updateSprites(
                        new Int32Array(this.buffers.posX, 0, size),
                        new Int32Array(this.buffers.posY, 0, size),
                        new Uint32Array(this.buffers.state, 0, size),
                    );
                    pass.updateTick(this.tick);
                }
            }
        });

        this.engine.renderer.postRenderHooks.push(() => this.sim());
    }

    start() {
        this.call('start');
    }

    sim() {
        if (!this.ready) return;
        if (!this.buffers.posX || !this.buffers.posY || !this.buffers.state) return;

        this.worker.postMessage(
            { call: 'sim', args: [this.velocity], buffers: this.buffers },
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
