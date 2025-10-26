import SimWorker from '../asteroid';
import Renderer from '../render/base';
import { SpritePass } from '../render/sprite-pass';
import { EngineParam } from '../types';

export default class Engine {
    public readonly renderer: Renderer;
    public readonly params: EngineParam;
    public readonly sim: SimWorker;

    constructor(params: EngineParam) {
        this.params = params;
        this.renderer = new Renderer(this);
        this.sim = new SimWorker(this);
    }

    init() {
        return this.renderer.init().catch(e => console.error(e));
    }
}
