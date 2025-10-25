import AsteroidASM from '../asteroid';
import Renderer from '../render/base';
import { SpritePass } from '../render/sprite-pass';
import { EngineParam } from '../types';

export default class Engine {
    public readonly renderer: Renderer;
    public readonly params: EngineParam;
    public readonly asm: AsteroidASM;
    private tick = 0;

    constructor(params: EngineParam) {
        this.params = params;
        this.renderer = new Renderer(this);
        this.asm = new AsteroidASM();
    }

    async init() {
        await Promise.all([
            this.asm.init(),
            this.renderer.init().catch(e => console.error(e)),
        ]);
        this.syncAsteroids();
    }

    syncAsteroids() {
        if (!this.asm.ready) return;
        const pass = this.renderer.pass;
        if (!(pass instanceof SpritePass)) return;
        this.tick++;

        const vel = -1 / 30;
        this.asm.module.tick(vel);

        const size = this.asm.module.get_asteroid_size();
        const posXPtr = this.asm.module.get_asteroid_pos_x();
        const posYPtr = this.asm.module.get_asteroid_pos_y();
        const statePtr = this.asm.module.get_asteroid_state();

        const posX = new Int32Array(this.asm.memory.buffer, posXPtr, size);
        const posY = new Int32Array(this.asm.memory.buffer, posYPtr, size);
        const state = new Uint32Array(this.asm.memory.buffer, statePtr, size);

        pass.updateSprites(posX, posY, state);
        pass.updateTick(this.tick);
    }
}
